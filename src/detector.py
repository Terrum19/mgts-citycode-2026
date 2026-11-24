"""Детектор для варианта 1: Mask2Former, обученный на Mapillary Vistas (уличные кадры, есть классы
«люк», «опора», «распределительный шкаф»). Работает без интернета, если веса скачаны заранее.

    python src/detector.py data/raw/panoramas/osm_123.jpg     # проверка: что нашёл и за сколько

Нужны torch и transformers (requirements-detector.txt).
"""
import sys
import threading
import time

from PIL import Image


class Detector:
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
        self.lock = threading.Lock()  # одна модель на все потоки: прогоняем по одному кадру

    def detect(self, img: Image.Image) -> list[dict]:
        """-> [{type, bbox [x1,y1,x2,y2] px, score, label}] для нужных классов."""
        torch, cfg = self.torch, self.cfg
        img = img.convert("RGB")
        w, h = img.size
        with self.lock, torch.inference_mode():
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
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            if min(x2 - x1, y2 - y1) < cfg.get("min_side", 12):
                continue
            found.append({"type": self.keep[s["label_id"]], "bbox": [x1, y1, x2, y2],
                          "score": round(float(s["score"]), 3), "label": names[s["label_id"]]})
        return found


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
    det = Detector(tomllib.loads((root / "config.toml").read_text())["detector"])
    for p in sys.argv[1:]:
        t0 = time.time()
        found = det.detect(Image.open(p))
        print(f"{p}: {time.time() - t0:.1f}s")
        for o in found:
            print("  ", o)
