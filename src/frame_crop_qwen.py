import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from detector import crop_box
from grounding_qwen import GroundingQwen, TYPES
from vlm import ROOT, parse_json, to_jpeg


CLASSES = ("rs_mech", "rs_ok", "rs_paint", "tk_mech", "tk_ok", "tk_paint")
TYPE_CODES = {"rsh": "rs", "tksh": "tk"}
STATES = {"mechanical": "mech", "paint": "paint", "ok": "ok"}
CLASS_LABELS = dict(zip(CLASSES, (
    "РЩ · механическое повреждение", "РЩ · норма", "РЩ · повреждение покрытия",
    "ТКШ · механическое повреждение", "ТКШ · норма", "ТКШ · повреждение покрытия",
)))


def parse_frame(text):
    answer = parse_json(text)
    if not isinstance(answer, dict) or type(answer.get("class")) is not int:
        return None
    if not 1 <= answer["class"] <= len(CLASSES):
        return None
    if isinstance(answer.get("description"), str) and answer["description"].strip():
        return answer
    return None


def compare_checks(frame, primary, detections):
    if frame is None:
        return "review", "Проверка целого фото не вернула допустимый класс."
    if primary is None:
        reason = "Все кандидаты определены как фон." if detections else "Детектор не нашёл подходящей рамки."
        return "review", reason
    answer = primary["answer"]
    if not answer or answer["verdict"] == "review":
        return "review", "Не удалось уверенно оценить корпус в основной рамке."
    crop_class = TYPE_CODES[answer["type"]] + "_" + STATES[answer["verdict"]]
    if CLASSES[frame["class"] - 1] != crop_class:
        return "review", "Оценки целого фото и основной рамки расходятся."
    return ("ok" if answer["verdict"] == "ok" else "bad"), ""


class FrameCropQwen(GroundingQwen):
    def __init__(self, settings, config):
        vlm_name = settings["vlm"]
        model_config = dict(config["variants"][vlm_name], max_tokens=1000, timeout=75)
        config = dict(config, variants={**config["variants"], vlm_name: model_config})
        super().__init__(settings, config)
        self.frame_prompt = (ROOT / settings["frame_prompt"]).read_text(encoding="utf-8").rstrip()
        self.encoder = None
        self.head = None
        self.head_digest = ""
        self.dino_note = ""
        self.dino_lock = threading.Lock()
        encoder_path = ROOT / settings["encoder"]
        head_path = ROOT / settings["crop_head"]
        if not encoder_path.is_dir() or not head_path.is_file():
            self.dino_note = "Диагностика DINOv2 недоступна: отсутствуют локальные веса или классификатор вырезов."
            return
        import joblib
        from embed import Embedder

        self.head = joblib.load(head_path)
        if set(self.head.classes_) != set(CLASSES):
            raise ValueError("Классы головы DINOv2 не соответствуют шести классам РЩ и ТКШ")
        self.head_digest = hashlib.sha256(head_path.read_bytes()).hexdigest()
        self.encoder = Embedder(str(encoder_path), device=settings.get("device", "cpu"), size=224)

    def fingerprint(self):
        signature = super().fingerprint() + self.frame_prompt + self.head_digest
        return hashlib.sha256(signature.encode()).hexdigest()

    def diagnose(self, image, detections):
        if self.encoder is None:
            return {"available": False, "reason": self.dino_note, "objects": []}
        from crop_dino import features

        objects = []
        with self.dino_lock:
            for offset in range(0, len(detections), 8):
                group = detections[offset:offset + 8]
                images = [image.crop(crop_box(item["bbox"], *image.size, 0.2, 96)) for item in group]
                labels = self.head.predict(features(self.encoder, images))
                for index, (label, item) in enumerate(zip(labels, group), offset + 1):
                    objects.append({"proposal_id": index, "bbox": item["bbox"], "class": str(label)})
        return {"available": True, "objects": objects}

    def predict(self, image):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            full = pool.submit(self.vlm._ask, to_jpeg(image, self.vlm.max_side), self.frame_prompt, parse_frame)
            result = super().predict(image)
            frame_call = full.result()
        frame = frame_call["value"]
        retained = result["objects"]
        selected = max(retained, key=lambda item: item["det_score"]) if retained else None
        primary = result["crop_results"][selected["proposal_id"] - 1] if selected else None
        answer = (primary or {}).get("answer") or {}
        frame_class = CLASSES[frame["class"] - 1] if frame else "review"
        crop_class = "review"
        if answer.get("verdict") in STATES:
            crop_class = TYPE_CODES[answer["type"]] + "_" + STATES[answer["verdict"]]
        status, reason = compare_checks(frame, primary, result["detections"])
        frame_description = (frame or {}).get("description", "")
        crop_description = answer.get("evidence", "")
        kind = answer.get("type")
        if kind not in TYPES:
            kind = {"rs":"rsh", "tk":"tksh"}.get(frame_class.split("_")[0], "cabinet")
        final_object = {
            "type": kind, "status": status, "bbox": selected["bbox"] if selected else None,
            "damage": selected["damage"] if selected and status == "bad" else None,
            "description": reason or crop_description,
        }
        result.update(
            objects=[final_object], frame_class=frame_class, crop_class=crop_class,
            frame_description=frame_description, crop_description=crop_description, review_reason=reason,
            primary_proposal_id=primary["proposal_id"] if primary else None,
            frame_result=frame_call, dinov2=self.diagnose(image, result["detections"]),
            error=frame_call["error"] or (primary or {}).get("error"),
            bad_json=bool(frame_call["bad_json"] or (primary or {}).get("bad_json")),
            cost=result["cost"] + frame_call["cost"],
            sub_opinions=[
                {"model": "Qwen235 · целое фото", "label": CLASS_LABELS.get(frame_class, "На проверку"), "description": frame_description},
                {"model": "Qwen235 · основная рамка", "label": CLASS_LABELS.get(crop_class, "На проверку"), "description": crop_description},
            ],
        )
        result["seconds"] = round(time.monotonic() - started, 2)
        return result
