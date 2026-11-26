"""Варианты системы за одним интерфейсом: make(name, cfg).predict(img) -> {"objects": [...], ...}.

Вариант описывается в config.toml, секция [variants.<имя>]:
  kind = "vlm"      (по умолчанию) — только VLM: base_url + model;
  kind = "det_vlm"  — детектор [detector] находит объекты, VLM из варианта `vlm = "<имя>"` оценивает вырезы.

Объект в ответе: {"type", "bbox": [x1,y1,x2,y2] px, "status": "ok|bad", "damage", "description"}.
"""
import time

from PIL import Image

from detector import crop_box
from vlm import VLM, clean_state

_detector = None  # тяжёлая модель: одна на процесс


def get_detector(cfg: dict):
    global _detector
    if _detector is None:
        from detector import Detector
        _detector = Detector(cfg)
    return _detector


class VLMOnly:
    def __init__(self, vcfg: dict, cfg: dict):
        self.vlm = VLM(vcfg, cfg["taxonomy"])
        self.workers = vcfg.get("workers", 2)

    def fingerprint(self) -> str:
        return self.vlm.fingerprint()

    def predict(self, img: Image.Image) -> dict:
        return self.vlm.predict(img)


class DetVLM:
    def __init__(self, vcfg: dict, cfg: dict):
        self.dcfg = cfg["detector"]
        self.tax = cfg["taxonomy"]
        self.det = get_detector(self.dcfg)
        self.vlm = VLM(cfg["variants"][vcfg["vlm"]], self.tax)
        self.workers = vcfg.get("workers", 2)

    def fingerprint(self) -> str:
        d = self.dcfg
        return f"{self.vlm.fingerprint()}-{d['model'].split('/')[-1]}-{d.get('min_score')}-{d.get('pad')}"

    def predict(self, img: Image.Image) -> dict:
        t0 = time.time()
        img = img.convert("RGB")
        w, h = img.size
        found = self.det.detect(img)
        det_sec = time.time() - t0
        out = {"objects": [], "bad_json": False, "error": None, "cost": 0.0, "detections": found}
        for d in found:
            box = crop_box(d["bbox"], w, h, self.dcfg.get("pad", 0.2), self.dcfg.get("crop_min_side", 96))
            r = self.vlm.classify_crop(img.crop(box), d["type"])
            out["cost"] += r["cost"]
            out["error"] = out["error"] or r["error"]
            a = r["answer"]
            if a is None:                   # VLM не ответила — объект оставляем, статус не знаем
                out["bad_json"] |= r["bad_json"]
                out["objects"].append({**d, "status": "review", "damage": None, "description": "", "vlm_failed": True})
                continue
            if a.get("is_object") is False:  # VLM отклонила находку детектора (знак, ливнёвка и т.п.)
                continue
            status, damage = clean_state(a, self.tax)
            out["objects"].append({"type": a.get("type") if a.get("type") in self.tax["object_types"] else d["type"],
                                   "bbox": d["bbox"], "status": status, "damage": damage,
                                   "description": str(a.get("description") or ""), "det_score": d["score"]})
        out["seconds"] = round(time.time() - t0, 2)
        out["det_seconds"] = round(det_sec, 2)
        return out


def make(name: str, cfg: dict):
    vcfg = cfg["variants"][name]
    kind = vcfg.get("kind", "vlm")
    if kind == "vlm":
        return VLMOnly(vcfg, cfg)
    if kind == "photo_dino":
        from photo_dino import PhotoDino
        return PhotoDino(vcfg)
    if kind in ("det_dino", "boxes_dino"):
        from crop_dino import CropDino
        return CropDino(vcfg, cfg["detector"] if kind == "det_dino" else None)
    if kind == "grounding_qwen":
        from grounding_qwen import GroundingQwen
        return GroundingQwen(vcfg, cfg)
    if kind == "frame_crop_qwen":
        from frame_crop_qwen import FrameCropQwen
        return FrameCropQwen(vcfg, cfg)
    if kind == "cabinet_ensemble":
        from cabinet_ensemble import CabinetEnsemble
        return CabinetEnsemble(vcfg, cfg)
    if kind == "det_vlm":
        return DetVLM(vcfg, cfg)
    raise ValueError(f"неизвестный kind={kind!r} у варианта {name}")
