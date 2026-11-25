"""Единый контракт результата: объектный, кадровый и смешанный.

Ключевые правила:
  * объект всегда имеет корректный bbox; кадровый результат живёт в ``frame`` и не
    имитируется объектом с ``bbox=None``;
  * статус ``review`` — полноценный результат, а не ошибка: неизвестное состояние не
    превращается в ``ok``;
  * ``review`` отделён от ``status``: в кадре может быть подтверждённый ``bad`` и ещё
    один спорный объект. Тогда ``status="bad"``, ``review=True``, а маршрут — review;
  * кадровый агрегат, выведенный из объектов, честно помечается ``source="objects"``.

Модуль не тянет torch/PIL: его можно проверять stdlib-тестами.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 2
OUTPUT_STATUSES = ("ok", "bad", "review")
CLASS_STATUSES = ("ok", "bad")
SCOPES = ("frame", "objects", "mixed")

# Нормализованные причины review. CSV и UI показывают только эти значения.
REVIEW_REASONS = frozenset({
    "ood",
    "quality",
    "low_confidence",
    "vlm_error",
    "vlm_unavailable",
    "bad_json",
    "taxonomy_mismatch",
    "missing_prediction",
    "not_target",
    "low_support",
    "missing_status",
    "low_status_confidence",
    "low_margin",
    "grey_zone",
    "invalid_bbox",
})
_REASON_ALIASES = {
    "error": "vlm_error",
    "api_error": "vlm_error",
    "vlm_review": "low_confidence",
    "review": "low_confidence",
    "invalid_bbox_objects": "invalid_bbox",
    "missing": "missing_prediction",
    "no_status": "missing_status",
    "unsupported_taxonomy": "taxonomy_mismatch",
    "not_object": "not_target",
}


def normalize_reason(reason: Any) -> str:
    text = str(reason or "").strip().lower().replace(" ", "_").replace("-", "_")
    text = _REASON_ALIASES.get(text, text)
    return text or "low_confidence"


def normalize_reasons(reasons: Iterable[Any] | None) -> list[str]:
    return sorted({normalize_reason(r) for r in (reasons or []) if str(r or "").strip()})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def valid_bbox(box: Any) -> tuple[float, float, float, float] | None:
    """Четыре конечных числа, x1 < x2, y1 < y2 -> кортеж; иначе None.

    Числовые строки допускаются (JSON/Excel иногда отдают "12.0"), bool — нет.
    Функция не бросает исключений: ею проверяются и GT, и предсказания.
    """
    if isinstance(box, (str, bytes)) or box is None:
        return None
    try:
        values = list(box)
    except TypeError:
        return None
    if len(values) != 4:
        return None
    out = []
    for value in values:
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None
        if not _is_number(value):
            return None
        out.append(float(value))
    x1, y1, x2, y2 = out
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def invalid_bbox_reason(box: Any) -> str:
    if box is None:
        return "missing"
    if isinstance(box, (str, bytes)):
        return "not_a_list"
    try:
        values = list(box)
    except TypeError:
        return "not_a_list"
    if len(values) != 4:
        return "not_four_numbers"
    for value in values:
        if isinstance(value, bool):
            return "not_a_number"
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return "not_a_number"
        if not _is_number(value):
            return "non_finite"
    x1, y1, x2, y2 = (float(v) for v in values)
    if x1 >= x2 or y1 >= y2:
        return "degenerate"
    return "unknown"


def sanitize_objects(objects: Sequence[Mapping[str, Any]] | None) -> tuple[list[dict], list[dict]]:
    """Разделить объекты на валидные и записи о некорректных рамках.

    Возвращает (objects, invalid): invalid — [{"index", "bbox", "reason"}]. Объект с
    некорректной рамкой не выбрасывает весь кадр: он лишь требует review.
    """
    good, invalid = [], []
    for index, raw in enumerate(objects or []):
        if not isinstance(raw, Mapping):
            invalid.append({"index": index, "bbox": raw, "reason": "not_an_object"})
            continue
        obj = dict(raw)
        box = valid_bbox(obj.get("bbox"))
        if box is None:
            invalid.append({"index": index, "bbox": obj.get("bbox"), "reason": invalid_bbox_reason(obj.get("bbox"))})
            continue
        obj["bbox"] = [round(v, 3) if not float(v).is_integer() else int(v) for v in box]
        good.append(obj)
    return good, invalid


def _object_review_reasons(objects: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for obj in objects:
        if str(obj.get("status")) == "review":
            reasons.extend(obj.get("review_reasons") or [])
    return normalize_reasons(reasons)


def derive_frame(
    objects: Sequence[Mapping[str, Any]],
    *,
    error: str | None = None,
    bad_json: bool = False,
    invalid_bbox_count: int = 0,
) -> dict:
    """Кадровый агрегат из объектов.

    bad побеждает review; review-объект не понижает подтверждённый bad, но означает
    обязательную ручную проверку. Ошибка API является review только если в кадре нет
    подтверждённого bad: сбой необязательного описания не должен скрывать дефект.
    """
    statuses = [str(o.get("status")) for o in objects]
    bad = statuses.count("bad")
    review_objects = [o for o in objects if str(o.get("status")) == "review"]
    uncertain = bool(review_objects or invalid_bbox_count or ((error or bad_json) and not bad))
    status = "bad" if bad else ("review" if uncertain else "ok")
    reasons = _object_review_reasons(review_objects)
    if invalid_bbox_count:
        reasons.append("invalid_bbox")
    if (error or bad_json) and not bad:
        reasons.append("vlm_error" if error else "bad_json")
    review = uncertain
    if status == "review" and not reasons:
        reasons.append("low_confidence")
    return {
        "status": status,
        "type": None,
        "damage": None,
        "description": "",
        "review": bool(review),
        "review_reasons": normalize_reasons(reasons),
        "source": "objects",
    }


def make_frame(
    status: str,
    *,
    type: str | None = None,
    damage: str | None = None,
    description: str = "",
    review: bool | None = None,
    review_reasons: Iterable[Any] | None = None,
    source: str = "native",
) -> dict:
    status = status if status in OUTPUT_STATUSES else "review"
    reasons = normalize_reasons(review_reasons)
    if review is None:
        review = status == "review"
    if status == "review":
        review = True
        damage = None
        if not reasons:
            reasons = ["low_confidence"]
    return {"status": status, "type": type, "damage": damage if status == "bad" else None,
            "description": description or "", "review": bool(review),
            "review_reasons": reasons, "source": source}


def make_result(
    *,
    objects: Sequence[Mapping[str, Any]] | None = None,
    frame: Mapping[str, Any] | None = None,
    scope: str | None = None,
    invalid_bbox_objects: Sequence[Mapping[str, Any]] | None = None,
    seconds: float = 0.0,
    cost: float = 0.0,
    bad_json: bool = False,
    error: str | None = None,
    **telemetry: Any,
) -> dict:
    """Собрать ответ модели по версии схемы 2.

    Если frame не задан, он выводится из objects (в том числе пустых: успешный пустой
    ответ — это ok, а не отсутствие результата). Некорректные рамки перечисляются в
    invalid_bbox_objects и требуют review, но не роняют кадр.
    """
    good, auto_invalid = sanitize_objects(objects)
    invalid = list(invalid_bbox_objects or []) + auto_invalid
    if frame is None:
        frame_out = derive_frame(good, error=error, bad_json=bad_json, invalid_bbox_count=len(invalid))
    else:
        frame_out = make_frame(
            str(frame.get("status", "review")),
            type=frame.get("type"),
            damage=frame.get("damage"),
            description=str(frame.get("description") or ""),
            review=frame.get("review"),
            review_reasons=frame.get("review_reasons"),
            source=str(frame.get("source") or "native"),
        )
        if invalid and not frame_out["review"]:
            frame_out["review"] = True
            frame_out["review_reasons"] = normalize_reasons(frame_out["review_reasons"] + ["invalid_bbox"])
    inferred = "mixed" if frame is not None and good else ("frame" if frame is not None else "objects")
    scope_out = scope or inferred
    if scope_out not in SCOPES:
        scope_out = inferred
    out = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope_out,
        "frame": frame_out,
        "objects": good,
        "invalid_bbox_objects": list(invalid),
        "seconds": float(seconds or 0.0),
        "cost": float(cost or 0.0),
        "bad_json": bool(bad_json),
        "error": error,
    }
    for key, value in telemetry.items():
        out.setdefault(key, value)
    return out


def triage_route(frame: Mapping[str, Any] | None) -> str:
    """Единственный операционный маршрут кадра: review | repair | none.

    Review сильнее repair: подтверждённый bad + спорный объект всё равно требует
    человека, но не считается одновременно FP и review.
    """
    if not frame:
        return "review"
    status = str(frame.get("status"))
    if frame.get("review") or status == "review":
        return "review"
    if status == "bad":
        return "repair"
    return "none"


def get_objects(pred: Mapping[str, Any] | None) -> list[dict]:
    if not isinstance(pred, Mapping):
        return []
    objects = pred.get("objects")
    return list(objects) if isinstance(objects, (list, tuple)) else []


def get_frame(pred: Mapping[str, Any] | None) -> dict | None:
    if not isinstance(pred, Mapping):
        return None
    frame = pred.get("frame")
    return dict(frame) if isinstance(frame, Mapping) else None
