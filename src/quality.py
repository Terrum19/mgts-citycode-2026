"""Некритичный Quality Gate: яркость/контраст/резкость/пересвет.

По умолчанию выключен: пороги не откалиброваны, а плохой кадр с видимым дефектом нельзя
блокировать. Модуль считает признаки на downsampled grayscale без OpenCV; PIL нужен
только адаптеру frame_quality.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

DEFAULT = {
    "enabled": False,
    "max_side": 320,
    "min_mean": 25.0,
    "max_mean": 235.0,
    "min_std": 18.0,
    "min_sharpness": 40.0,
    "max_overexposed": 0.50,
}


def _rows(gray: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row) for row in gray if row]


def quality_metrics(gray: Sequence[Sequence[float]]) -> dict:
    rows = _rows(gray)
    if not rows or not rows[0]:
        return {"mean": 0.0, "std": 0.0, "sharpness": 0.0, "overexposed": 1.0, "pixels": 0}
    h, w = len(rows), len(rows[0])
    values = [float(v) for row in rows for v in row]
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    over = sum(1 for v in values if v >= 250) / n
    # дисперсия лапласиана 3x3; края не считаем, чтобы не выдумывать резкость
    lap = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            v = (4 * rows[y][x] - rows[y - 1][x] - rows[y + 1][x] - rows[y][x - 1] - rows[y][x + 1])
            lap.append(v)
    sharp = (sum(v * v for v in lap) / len(lap)) if lap else 0.0
    return {"mean": mean, "std": std, "sharpness": sharp, "overexposed": over, "pixels": n,
            "width": w, "height": h}


def assess_quality(gray: Sequence[Sequence[float]], cfg: Mapping[str, Any] | None = None) -> dict:
    cfg = {**DEFAULT, **dict(cfg or {})}
    metrics = quality_metrics(gray)
    if not cfg.get("enabled"):
        return {"passed": True, "metrics": metrics, "review_reasons": []}
    reasons = []
    if metrics["pixels"] == 0:
        reasons.append("quality")
    elif metrics["mean"] < float(cfg["min_mean"]) or metrics["mean"] > float(cfg["max_mean"]):
        reasons.append("quality")
    if metrics["pixels"] and metrics["std"] < float(cfg["min_std"]):
        reasons.append("quality")
    if metrics["pixels"] and metrics["sharpness"] < float(cfg["min_sharpness"]):
        reasons.append("quality")
    if metrics["overexposed"] > float(cfg["max_overexposed"]):
        reasons.append("quality")
    return {"passed": not reasons, "metrics": metrics, "review_reasons": sorted(set(reasons))}


def frame_quality(img, cfg: Mapping[str, Any] | None = None) -> dict:
    """PIL-адаптер: уменьшить кадр и посчитать признаки на grayscale."""
    cfg = {**DEFAULT, **dict(cfg or {})}
    size = int(cfg.get("max_side", 320) or 320)
    small = img.convert("L")
    if max(small.size) > size:
        small = small.copy()
        small.thumbnail((size, size))
    width, height = small.size
    data = list(small.getdata())
    rows = [data[y * width:(y + 1) * width] for y in range(height)]
    return assess_quality(rows, cfg)
