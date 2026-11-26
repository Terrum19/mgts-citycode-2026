"""Grounding DINO proposals with an independent Qwen inspection for every crop."""
import hashlib
import inspect
import json
import threading
import time

from detector import crop_box
from vlm import ROOT, VLM, parse_json, to_jpeg

TYPES = {
    "rsh": "распределительный щит (РЩ)",
    "tksh": "телекоммуникационный климатический шкаф (ТКШ)",
}
QUERIES = {
    "rsh": "electrical distribution box. small electrical cabinet.",
    "tksh": "outdoor telecom cabinet. telecommunications equipment cabinet. cabinet with ventilation grille.",
}
DAMAGE = {"mechanical", "paint"}
VERDICTS = DAMAGE | {"ok", "review", "not_target"}
PROMPT = """На фото вырез с контекстом. Оцени только корпус в рамке {target_box} в шкале 0–1000
по ширине и высоте выреза. Детектор искал «{hint}». Тип определяй по самому объекту.
Соседние корпуса, опора и кабели за рамкой нужны только как контекст.

- tksh: телекоммуникационный климатический шкаф, обычно глубокий корпус с климатической
  пристройкой, вентиляционной панелью или жалюзи. Гладкий бок не исключает ТКШ.
- rsh: распределительный щит, обычно компактный короб с простой дверцей. В открытом
  щите могут быть видны автоматы и клеммы. Цвет, высота и знак опасности бывают у обоих.

- mechanical: смятая или порванная панель, пролом, сорванная или отсутствующая дверь,
  повреждённое крепление, смятый климатический отсек. Штатные швы и целая дверца,
  открытая при обслуживании, сами по себе не подтверждают повреждение.
- paint: отслоение или отсутствие краски, оголённый металл, коррозия. Грязь, тень и блики
  сами по себе не дефект. При отдельной механической деформации и повреждении краски
  выбери mechanical.
- ok: корпус различим, видимой части достаточно для оценки, явных дефектов нет.
- review: корпус или нужная деталь слишком мелкая, скрыта, обрезана либо тип неясен.
- not_target: в рамке фон, опора, провода, отдельная решётка без корпуса или другой объект.

Укажи конкретный видимый признак и его место. Ответь только JSON:
{{"type":"tksh|rsh" или null,"verdict":"mechanical|paint|ok|review|not_target","evidence":"короткая причина"}}."""


def parse_inspection(text):
    answer = parse_json(text)
    if not answer or not isinstance(answer.get("verdict"), str) or answer["verdict"] not in VERDICTS:
        return None
    verdict = answer["verdict"]
    kind = answer.get("type")
    if not isinstance(kind, str) or kind not in TYPES:
        if verdict in DAMAGE or verdict == "ok":
            return None
        kind = None
    evidence = answer.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    return {"type": kind, "verdict": verdict, "evidence": evidence.strip()}


def overlap(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def detector_proposals(raw, width, height, min_side, dedup_iou):
    proposals = []
    for candidate in sorted(raw, key=lambda row: row["score"], reverse=True):
        x1, y1, x2, y2 = candidate["bbox"]
        box = [max(0, round(x1)), max(0, round(y1)), min(width, round(x2)), min(height, round(y2))]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if min(bw, bh) < min_side or bw * bh > 0.88 * width * height:
            continue
        if any(overlap(box, other["bbox"]) >= dedup_iou for other in proposals):
            continue
        proposals.append({**candidate, "bbox": box})
    return proposals


class GroundingDetector:
    def __init__(self, settings):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = settings.get("device", "cpu")
        self.settings = settings
        self.lock = threading.Lock()
        weights = ROOT / settings["model"]
        if not weights.is_dir():
            raise FileNotFoundError(f"Нет локальных весов Grounding DINO: {weights}")
        self.processor = AutoProcessor.from_pretrained(str(weights), local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(weights), local_files_only=True,
        ).eval().to(self.device)
        self.post = self.processor.post_process_grounded_object_detection
        threshold = "box_threshold" if "box_threshold" in inspect.signature(self.post).parameters else "threshold"
        self.thresholds = {threshold: settings.get("min_score", 0.22),
                           "text_threshold": settings.get("text_threshold", 0.20)}

    def detect(self, image):
        width, height = image.size
        found = []
        with self.lock, self.torch.inference_mode():
            for kind, query in QUERIES.items():
                tokens = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
                output = self.post(self.model(**tokens), tokens.input_ids,
                                   target_sizes=[(height, width)], **self.thresholds)[0]
                for box, score in zip(output["boxes"].tolist(), output["scores"].tolist()):
                    found.append({"type": kind, "bbox": box, "score": round(score, 4)})
        return detector_proposals(found, width, height, self.settings.get("min_side", 20),
                                  self.settings.get("dedup_iou", 0.50))


class GroundingQwen:
    def __init__(self, settings, config):
        self.settings = settings
        self.vlm = VLM(config["variants"][settings["vlm"]], config["taxonomy"])
        if not self.vlm.key:
            raise ValueError("Для Grounding DINO + Qwen нужен OPENROUTER_API_KEY")
        self.detector = GroundingDetector(settings)
        self.workers = settings.get("workers", 4)

    def fingerprint(self):
        material = json.dumps([self.settings, self.vlm.fingerprint(), QUERIES, PROMPT, self.vlm.max_side],
                              sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode()).hexdigest()

    def predict(self, image):
        started = time.monotonic()
        image = image.convert("RGB")
        found = self.detector.detect(image)
        det_seconds = time.monotonic() - started
        result = {"objects": [], "detections": found, "crop_results": [], "bad_json": False,
                  "error": None, "cost": 0.0, "det_seconds": round(det_seconds, 2)}
        for index, detection in enumerate(found, 1):
            box = crop_box(detection["bbox"], *image.size, self.settings.get("pad", 0.30),
                           self.settings.get("crop_min_side", 96))
            crop = image.crop(box)
            offset_x, offset_y = box[:2]
            x1, y1, x2, y2 = detection["bbox"]
            inside = [round(value * 1000 / side) for value, side in zip(
                (x1 - offset_x, y1 - offset_y, x2 - offset_x, y2 - offset_y), crop.size * 2,
            )]
            prompt = PROMPT.format(target_box=inside, hint=TYPES[detection["type"]])
            answer = self.vlm._ask(to_jpeg(crop, self.vlm.max_side), prompt, parse_inspection)
            item = answer["value"]
            verdict = item["verdict"] if item else "review"
            result["cost"] += answer["cost"]
            result["error"] = result["error"] or answer["error"]
            result["bad_json"] |= answer["bad_json"]
            result["crop_results"].append({"proposal_id": index, "bbox": detection["bbox"], "crop_bbox": box,
                                           "answer": item, "error": answer["error"], "bad_json": answer["bad_json"]})
            if verdict == "not_target":
                continue
            status = "bad" if verdict in DAMAGE else verdict
            result["objects"].append({
                "type": (item or {}).get("type") or "cabinet", "bbox": detection["bbox"], "status": status,
                "damage": {"paint": "coating_corrosion", "mechanical": "mechanical_damage"}.get(verdict),
                "description": item["evidence"] if item else "Qwen не вернул валидный ответ",
                "det_score": detection["score"], "proposal_id": index,
            })
        result["seconds"] = round(time.monotonic() - started, 2)
        return result
