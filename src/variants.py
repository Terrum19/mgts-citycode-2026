"""Варианты системы за одним интерфейсом: make(name, cfg).predict(img) -> {"objects": [...], ...}.

Вариант описывается в config.toml, секция [variants.<имя>]:
  kind = "vlm"       — только VLM: base_url + model;
  kind = "det_vlm"   — детектор [detector] находит объекты, VLM из варианта `vlm` оценивает вырезы;
  kind = "boxes_vlm" — рамки организатора уже есть, VLM оценивает каждый вырез;
  kind = "det_dino" / "boxes_dino" / "photo_dino" — обученный DINOv2.

Каждый predict возвращает единый контракт results.make_result: frame + objects.
"""
from __future__ import annotations

import hashlib
import json
import time

from PIL import Image

from config_utils import (check_offline_vlm, detector_config, is_offline, is_local_url,
                          resolve_variant)
from detector import crop_box
from draw import highlight_box
from results import make_result, normalize_reasons
from vlm import VLM

_detector = None  # тяжёлая модель: одна на процесс


def get_detector(cfg: dict):
    global _detector
    if _detector is None:
        from detector import Detector
        _detector = Detector(cfg)
    return _detector


class VLMOnly:
    def __init__(self, vcfg: dict, cfg: dict):
        check_offline_vlm(cfg, vcfg, is_offline(vcfg, cfg))
        self.vcfg = vcfg
        self.vlm = VLM(vcfg, cfg["taxonomy"])
        self.workers = vcfg.get("workers", 2)

    def fingerprint(self) -> str:
        return self.vlm.fingerprint()

    def predict(self, img: Image.Image) -> dict:
        return self.vlm.predict(img)


CONTEXT_MODES = ("crop", "context", "both")


class DetVLM:
    def __init__(self, vcfg: dict, cfg: dict, *, use_detector: bool = True):
        self.vcfg = vcfg
        self.dcfg = detector_config(cfg, vcfg)
        self.tax = cfg["taxonomy"]
        self.context_mode = vcfg.get("context_mode", "crop")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"неизвестный context_mode={self.context_mode!r}, допустимы {CONTEXT_MODES}")
        self.not_target = vcfg.get("not_target", "drop")
        if self.not_target not in ("drop", "review"):
            raise ValueError("not_target должен быть drop или review")
        self.offline = is_offline(vcfg, cfg)
        check_offline_vlm(cfg, vcfg, self.offline)
        self.det = get_detector(self.dcfg) if use_detector else None
        self.vlm = VLM(resolve_variant(cfg, vcfg["vlm"]), self.tax)
        self.workers = vcfg.get("workers", 2)
        self.highlight_local = bool(vcfg.get("highlight_local", False))

    def fingerprint(self) -> str:
        payload = {
            "vlm": self.vlm.fingerprint(),
            "detector": self.det.fingerprint() if self.det is not None else None,
            "context_mode": self.context_mode,
            "not_target": self.not_target,
            "highlight_local": self.highlight_local,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def _handle(self, found: dict, call: dict) -> dict | None:
        """Ответ VLM по одной находке -> объект или None (только для not_target=drop)."""
        typ = found.get("type")
        base = {k: found[k] for k in ("bbox", "object_id") if k in found}
        if call.get("error"):
            return {**base, "type": typ, "status": "review", "damage": None,
                    "description": "", "review_reasons": ["vlm_error"], "vlm_error": call["error"][:300]}
        answer = call.get("answer")
        if not isinstance(answer, dict):
            reason = "bad_json" if call.get("bad_json") else "vlm_error"
            return {**base, "type": typ, "status": "review", "damage": None,
                    "description": "", "review_reasons": [reason]}
        description = str(answer.get("description") or "")
        if answer.get("is_object") is False:
            if self.not_target == "drop":
                return None
            return {**base, "type": typ, "status": "review", "damage": None,
                    "description": description, "review_reasons": ["not_target"]}
        from cascade import answer_error, clean_state
        why = answer_error(answer)
        if why:
            return {**base, "type": typ, "status": "review", "damage": None,
                    "description": description, "review_reasons": ["vlm_error"], "vlm_error": why}
        status, damage = clean_state(answer, self.tax)
        ans_type = answer.get("type")
        typ_out = ans_type if ans_type in self.tax.get("object_types", {}) else typ
        if status in ("ok", "bad") and typ_out not in self.tax.get("object_types", {}):
            return {**base, "type": typ, "status": "review", "damage": None,
                    "description": description, "review_reasons": ["taxonomy_mismatch"]}
        return {**base, "type": typ_out, "status": status, "damage": damage,
                "description": description,
                "review_reasons": ["low_confidence"] if status == "review" else []}

    def _classify_box(self, img: Image.Image, d: dict) -> dict:
        w, h = img.size
        box = crop_box(d["bbox"], w, h, self.dcfg.get("pad", 0.2), self.dcfg.get("crop_min_side", 96))
        if self.context_mode == "crop":
            return self.vlm.classify_crop(img.crop(box), d.get("type") or "")
        crop_img = img.crop(box) if self.context_mode == "both" else None
        context = highlight_box(img, d["bbox"]) if not self.highlight_local else img.crop(box)
        return self.vlm.classify_crop_in_context(context, crop_img, d.get("type") or "")

    def predict(self, img: Image.Image, boxes: list[dict] | None = None) -> dict:
        t0 = time.perf_counter()
        img = img.convert("RGB")
        if self.det is None:
            if boxes is None:
                raise ValueError("Передайте файл рамок через --boxes")
            found = list(boxes)
        else:
            found = self.det.detect(img)
        det_sec = time.perf_counter() - t0
        objects, invalid = [], []
        cost, bad_json, error = 0.0, False, None
        for d in found:
            call = self._classify_box(img, d)
            cost += float(call.get("cost") or 0.0)
            bad_json = bad_json or bool(call.get("bad_json"))
            error = error or call.get("error")
            obj = self._handle(d, call)
            if obj is not None:
                objects.append(obj)
        return make_result(objects=objects, scope="mixed", invalid_bbox_objects=invalid,
                           seconds=time.perf_counter() - t0, cost=cost, bad_json=bad_json, error=error,
                           detections=found, det_seconds=det_sec)


class BoxesVLM(DetVLM):
    """Рамки организатора без детектора: VLM не имеет права молча удалить находку."""

    def __init__(self, vcfg: dict, cfg: dict):
        super().__init__(vcfg, cfg, use_detector=False)
        self.not_target = vcfg.get("not_target", "review")
        if self.not_target not in ("drop", "review"):
            raise ValueError("not_target должен быть drop или review")

    def predict(self, img: Image.Image, boxes: list[dict] | None = None) -> dict:
        if boxes is None:
            raise ValueError("Для boxes_vlm нужны рамки организатора (--boxes)")
        return super().predict(img, boxes=boxes)


def make(name: str, cfg: dict):
    vcfg = resolve_variant(cfg, name)
    kind = vcfg.get("kind", "vlm")
    if kind == "vlm":
        return VLMOnly(vcfg, cfg)
    if kind == "photo_dino":
        from photo_dino import PhotoDino
        return PhotoDino(vcfg, cfg)
    if kind in ("det_dino", "boxes_dino"):
        from crop_dino import CropDino
        return CropDino(vcfg, detector_config(cfg, vcfg) if kind == "det_dino" else None, cfg)
    if kind == "det_vlm":
        return DetVLM(vcfg, cfg)
    if kind == "boxes_vlm":
        return BoxesVLM(vcfg, cfg)
    raise ValueError(f"неизвестный kind={kind!r} у варианта {name}")
