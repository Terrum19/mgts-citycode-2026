"""Детектор объектов: подключаемые бэкенды (backend) за одним интерфейсом detect(img).

Бэкенды (ключ `backend` в [detector] config.toml):
  * "mask2former" (по умолчанию) — panoptic Mask2Former на Mapillary Vistas: люк, опора, шкаф.
    Проверенный вариант: не путает люк и ливнёвку, но тяжёлый (Swin-L, секунды на CPU),
    не видит `aerial_equipment` и имеет некоммерческую лицензию весов (CC BY-NC-SA 4.0);
  * "grounding_dino" — open-vocabulary по текстовым промптам `prompts` (единственный, кто может
    найти оборудование на опоре), но чаще путает люк и ливнёвку: нецелевое отсекается на 2-м
    уровне каскада (`is_object=false` в VLM);
  * "rtdetr" — локальные веса обученного детектора, `classes` маппит id2label -> наш тип.

Контракт у всех один: [{"type", "bbox": [x1,y1,x2,y2] px, "score", "label"}].
Выбор бэкенда — только через конфиг: молчаливой подмены модели нет, ошибка выбранного
бэкенда должна быть явной (иначе результат и кэш `runs/*/preds.json` станут невалидными).

    python src/detector.py data/raw/panoramas/osm_123.jpg     # проверка: что нашёл и за сколько

Нужны torch и transformers (requirements-detector.txt).
"""
import hashlib
import json
import sys
import threading
import time

from PIL import Image

BACKENDS = ("mask2former", "grounding_dino", "rtdetr")


def backend_name(cfg: dict | None) -> str:
    """Имя бэкенда из конфига; по умолчанию — проверенный mask2former."""
    name = (cfg or {}).get("backend", "mask2former")
    if name not in BACKENDS:
        raise ValueError(f"неизвестный backend={name!r}; допустимо: {', '.join(BACKENDS)}")
    return name


def config_fingerprint(cfg: dict | None) -> str:
    """Отпечаток конфига детектора: смена модели/порогов инвалидирует кэш предсказаний."""
    text = json.dumps(cfg or {}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def box_iou(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedup(found: list[dict], threshold: float = 0.5) -> list[dict]:
    """Один объект, найденный по нескольким промптам: оставляем рамку с большим score."""
    kept = []
    for obj in sorted(found, key=lambda o: -o["score"]):
        if not any(box_iou(obj["bbox"], other["bbox"]) >= threshold for other in kept):
            kept.append(obj)
    return kept


def _box(values, w: int, h: int, min_side: float) -> list[int] | None:
    """Рамка из float -> целые пиксели внутри кадра; слишком мелкая отбрасывается."""
    x1, y1, x2, y2 = (float(v) for v in values)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    box = [max(0, int(round(x1))), max(0, int(round(y1))), min(w, int(round(x2))), min(h, int(round(y2)))]
    if min(box[2] - box[0], box[3] - box[1]) < min_side:
        return None
    return box


class _Mask2FormerBackend:
    """Проверенный бэкенд: panoptic-сегментация Vistas, из маски берём только bbox."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch, self.cfg = torch, cfg
        self.device = cfg.get("device", "cpu")
        self.proc = AutoImageProcessor.from_pretrained(cfg["model"])
        # по умолчанию процессор растягивает кадр до 2048 по короткой стороне — на CPU это медленно
        self.size = ({"shortest_edge": cfg["shortest_edge"], "longest_edge": cfg.get("longest_edge", 2048)}
                     if cfg.get("shortest_edge") else None)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(cfg["model"]).eval().to(self.device)
        names = self.model.config.id2label
        self.keep = {int(i): cfg["classes"][n] for i, n in names.items() if n in cfg["classes"]}
        missing = set(cfg["classes"]) - set(names.values())
        if missing:
            print(f"! классов нет в модели: {missing}", file=sys.stderr)

    def detect(self, img: Image.Image) -> list[dict]:
        torch, cfg = self.torch, self.cfg
        img = img.convert("RGB")
        w, h = img.size
        with torch.inference_mode():
            inputs = self.proc(images=img, return_tensors="pt", **({"size": self.size} if self.size else {})).to(self.device)
            out = self.model(**inputs)
            res = self.proc.post_process_panoptic_segmentation(
                out, threshold=cfg.get("min_score", 0.5), target_sizes=[(h, w)], label_ids_to_fuse=set())[0]
        seg, found = res["segmentation"], []
        names = self.model.config.id2label
        for s in res["segments_info"]:
            if s["label_id"] not in self.keep:
                continue
            ys, xs = torch.where(seg == s["id"])
            if len(xs) == 0:
                continue
            box = _box([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], w, h, cfg.get("min_side", 12))
            if box is None:
                continue
            found.append({"type": self.keep[s["label_id"]], "bbox": box,
                          "score": round(float(s["score"]), 3), "label": names[s["label_id"]]})
        return found


class _GroundingDinoBackend:
    """Open-vocabulary: отдельный текстовый запрос на каждый наш тип (однозначный маппинг prompt -> type)."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch, self.cfg = torch, cfg
        self.device = cfg.get("device", "cpu")
        self.prompts = {str(k): str(v) for k, v in (cfg.get("prompts") or {}).items()}
        if not self.prompts:
            raise ValueError("[detector].prompts нужен для backend=grounding_dino: {наш_тип = \"текст. ещё текст.\"}")
        self.proc = AutoProcessor.from_pretrained(cfg["model"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg["model"]).eval().to(self.device)

    def _post(self, outputs, input_ids, w: int, h: int) -> dict:
        kwargs = {"text_threshold": self.cfg.get("text_threshold", 0.25), "target_sizes": [(h, w)]}
        threshold = self.cfg.get("min_score", 0.3)
        try:  # transformers >= 4.51
            return self.proc.post_process_grounded_object_detection(
                outputs, input_ids, threshold=threshold, **kwargs)[0]
        except TypeError:  # старее: параметр назывался box_threshold
            return self.proc.post_process_grounded_object_detection(
                outputs, input_ids, box_threshold=threshold, **kwargs)[0]

    def detect(self, img: Image.Image) -> list[dict]:
        cfg = self.cfg
        img = img.convert("RGB")
        w, h = img.size
        found = []
        for typ, text in self.prompts.items():
            inputs = self.proc(images=img, text=text, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            result = self._post(outputs, inputs["input_ids"], w, h)
            labels = result.get("text_labels")
            if labels is None:
                labels = result.get("labels")
            labels = [] if labels is None else labels
            for box, score, label in zip(result["boxes"], result["scores"], labels):
                prepared = _box(box.tolist(), w, h, cfg.get("min_side", 12))
                if prepared is None:
                    continue
                found.append({"type": typ, "bbox": prepared, "score": round(float(score), 3), "label": str(label)})
        return dedup(found, cfg.get("dedup_iou", 0.5))


class _RTDETRBackend:
    """Локальные веса обученного детектора (RT-DETR / совместимая архитектура из transformers)."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.torch, self.cfg = torch, cfg
        self.device = cfg.get("device", "cpu")
        self.proc = AutoImageProcessor.from_pretrained(cfg["model"])
        self.model = AutoModelForObjectDetection.from_pretrained(cfg["model"]).eval().to(self.device)
        mapping = cfg.get("classes") or {}
        self.keep = {int(i): mapping[name] for i, name in self.model.config.id2label.items() if name in mapping}
        missing = set(mapping) - set(self.model.config.id2label.values())
        if missing:
            print(f"! классов нет в модели: {missing}", file=sys.stderr)

    def detect(self, img: Image.Image) -> list[dict]:
        cfg = self.cfg
        img = img.convert("RGB")
        w, h = img.size
        names = self.model.config.id2label
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.proc.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=cfg.get("min_score", 0.5))[0]
        found = []
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            label_id = int(label)
            if label_id not in self.keep:
                continue
            prepared = _box(box.tolist(), w, h, cfg.get("min_side", 12))
            if prepared is None:
                continue
            found.append({"type": self.keep[label_id], "bbox": prepared,
                          "score": round(float(score), 3), "label": names[label_id]})
        return found


_BACKENDS = {"mask2former": _Mask2FormerBackend, "grounding_dino": _GroundingDinoBackend,
             "rtdetr": _RTDETRBackend}


class Detector:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.name = backend_name(self.cfg)
        self.impl = _BACKENDS[self.name](self.cfg)
        self.lock = threading.Lock()  # одна модель на все потоки: прогоняем по одному кадру

    def detect(self, img: Image.Image) -> list[dict]:
        """-> [{type, bbox [x1,y1,x2,y2] px, score, label}] для нужных классов."""
        with self.lock:
            return self.impl.detect(img)

    def fingerprint(self) -> str:
        return f"det-{self.name}-{config_fingerprint(self.cfg)}"


def crop_box(bbox: list[int], w: int, h: int, pad: float, min_side: int) -> tuple[int, int, int, int]:
    """Рамка с запасом pad вокруг объекта и не меньше min_side по каждой стороне (VLM нужен контекст)."""
    x1, y1, x2, y2 = bbox
    cw = max((x2 - x1) * (1 + 2 * pad), min_side)
    ch = max((y2 - y1) * (1 + 2 * pad), min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (max(0, round(cx - cw / 2)), max(0, round(cy - ch / 2)),
            min(w, round(cx + cw / 2)), min(h, round(cy + ch / 2)))


if __name__ == "__main__":
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "config.toml").read_text())["detector"]
    det = Detector(cfg)
    print(f"backend: {det.name}")
    for p in sys.argv[1:]:
        t0 = time.time()
        found = det.detect(Image.open(p))
        print(f"{p}: {time.time() - t0:.1f}s")
        for o in found:
            print("  ", o)
