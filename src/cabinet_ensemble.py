"""Один проход Grounding DINO и два независимых промта Qwen на одних и тех же вырезках.

    [variants.dss_cabinet_ensemble] kind = "cabinet_ensemble", vlm = "qwen235"

Модель всегда одна — `qwen235`. Второе мнение даёт не другая модель, а другой промт:
первый осматривает рамку детектора с подсказкой типа, второй — вырез без координат и подсказки.
Решение по вырезу принимает `consensus.combine_opinions`: единогласие подтверждает статус,
любое расхождение, ошибка API или битый JSON отправляют вырез на ручную проверку.
"""
import hashlib
import json
import time

from consensus import combine_opinions
from detector import crop_box
from grounding_qwen import PROMPT, TYPES, GroundingDetector, parse_inspection
from vlm import VLM, to_jpeg

FRAME_HINT = "ТКШ или РЩ"
FRAME_TARGET = [0, 0, 1000, 1000]

# Второй промт намеренно не содержит координат рамки и подсказки детектора: модель должна
# прийти к выводу сама, поэтому расхождение с первым промтом — содержательный сигнал.
PROMPT_ALT = """На фото вырез с контекстом. Определи, что находится в центре кадра, и оцени состояние.
- tksh: телекоммуникационный климатический шкаф — глубокий уличный корпус, вентиляционная панель,
  жалюзи или климатический отсек.
- rsh: распределительный щит — компактный короб с дверцей, часто на опоре или стене.
- mechanical: вмятины, пролом, сорванная или отсутствующая дверь, повреждённое крепление, наклон.
- paint: отслоение краски, ржавчина, коррозия, оголённый металл.
- ok: корпус цел, видимых дефектов нет.
- review: корпус мелкий, скрыт или обрезан, тип неясен.
- not_target: в центре не шкаф и не щит.
Ответь только JSON:
{{"type":"tksh|rsh" или null,"verdict":"mechanical|paint|ok|review|not_target","evidence":"короткая причина"}}"""


class CabinetEnsemble:
    def __init__(self, settings: dict, config: dict):
        self.settings = settings
        self.config = config
        vlm_name = settings.get("vlm") or next(iter(settings.get("vlms") or ["qwen235"]), "qwen235")
        if vlm_name not in config["variants"]:
            raise ValueError(f"Нет варианта VLM {vlm_name!r} в config.toml")
        vcfg = dict(config["variants"][vlm_name])
        vcfg["timeout"] = settings.get("vlm_timeout", vcfg.get("timeout", 180))
        vcfg["retries"] = settings.get("vlm_retries", vcfg.get("retries", 5))
        self.vlm = VLM(vcfg, config["taxonomy"])
        if not self.vlm.key:
            raise ValueError(f"Для модели {vlm_name} не найден API-ключ")
        self.vlm_name = vlm_name
        self.workers = settings.get("workers", 2)
        self.frame_only = bool(settings.get("frame_only", False))
        self.detector = None if self.frame_only else GroundingDetector(settings)

    def fingerprint(self) -> str:
        material = json.dumps(
            [self.settings, self.vlm_name, self.vlm.fingerprint(), PROMPT, PROMPT_ALT, TYPES,
             self.vlm.max_side],
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def _ask_both(self, jpeg: bytes, framed_prompt: str, result: dict) -> list[dict]:
        opinions = []
        for label, prompt in (("промт 1 (рамка)", framed_prompt), ("промт 2 (без рамки)", PROMPT_ALT)):
            answer = self.vlm._ask(jpeg, prompt, parse_inspection)
            opinions.append({
                "variant": f"{self.vlm_name} · {label}",
                "model": self.vlm.cfg.get("model", ""),
                "answer": answer["value"],
                "error": answer["error"],
                "bad_json": bool(answer["bad_json"]),
                "seconds": answer["seconds"],
                "cost": answer["cost"],
                "attempts": answer.get("attempts", 0),
                "raw": answer.get("raw", ""),
            })
            result["cost"] += answer["cost"]
            result["bad_json"] |= bool(answer["bad_json"])
        return opinions

    def predict(self, image) -> dict:
        started = time.monotonic()
        image = image.convert("RGB")
        width, height = image.size
        result = {
            "objects": [], "detections": [], "crop_results": [], "bad_json": False,
            "error": None, "cost": 0.0, "seconds": 0.0, "det_seconds": 0.0,
            "mode": "frame" if self.frame_only else "detector",
            "truncated_proposals": 0, "review_reasons": [],
        }

        if self.frame_only:
            proposals = [{"type": "cabinet", "bbox": None, "score": None, "synthetic": True}]
            result["detections"] = proposals
        else:
            try:
                proposals = self.detector.detect(image)
            except Exception as exc:  # noqa: BLE001 — сбой детектора не должен ронять весь прогон
                result["error"] = f"детектор: {exc}"[:300]
                result["review_reasons"].append("detector_error")
                result["seconds"] = round(time.monotonic() - started, 2)
                return result
            result["detections"] = proposals
            result["det_seconds"] = round(time.monotonic() - started, 2)
            if not proposals:
                result["review_reasons"].append("no_detections")

        max_crops = int(self.settings.get("max_crops", 4))
        ordered = sorted(proposals, key=lambda row: row.get("score") or 0.0, reverse=True)
        inspected = ordered[:max_crops]
        result["truncated_proposals"] = max(0, len(ordered) - len(inspected))
        if result["truncated_proposals"]:
            result["review_reasons"].append("crop_limit")

        confirmed = 0
        for index, proposal in enumerate(inspected, 1):
            if self.frame_only:
                crop = image
                box = [0, 0, width, height]
                target = FRAME_TARGET
                hint = FRAME_HINT
            else:
                box = crop_box(proposal["bbox"], width, height, self.settings.get("pad", 0.30),
                               self.settings.get("crop_min_side", 96))
                crop = image.crop(box)
                offset_x, offset_y = box[:2]
                x1, y1, x2, y2 = proposal["bbox"]
                target = [round(value * 1000 / side) for value, side in zip(
                    (x1 - offset_x, y1 - offset_y, x2 - offset_x, y2 - offset_y), crop.size * 2,
                )]
                hint = TYPES.get(proposal.get("type"), FRAME_HINT)
            framed_prompt = PROMPT.format(target_box=target, hint=hint)
            opinions = self._ask_both(to_jpeg(crop, self.vlm.max_side), framed_prompt, result)
            consensus = combine_opinions(opinions, proposal.get("score"),
                                         self.settings.get("review_det_score", 0.30))
            result["crop_results"].append({
                "proposal_id": index,
                "bbox": proposal.get("bbox"),
                "crop_bbox": box,
                "opinions": opinions,
                "consensus": consensus,
            })
            if consensus["status"] == "not_target":
                continue
            confirmed += 1
            result["objects"].append({
                "proposal_id": index,
                "bbox": None if self.frame_only else proposal["bbox"],
                "type": consensus["type"] or "cabinet",
                "status": consensus["status"],
                "damage": consensus["damage"],
                "description": consensus["description"],
                "det_score": proposal.get("score"),
                "review_reasons": consensus["review_reasons"],
                "full_agreement": consensus["full_agreement"],
            })

        if inspected and not confirmed:
            result["review_reasons"].append("no_confirmed_target")
        result["seconds"] = round(time.monotonic() - started, 2)
        return result
