"""CSV-строки для CLI и Streamlit: objects, frames, repair, review, triage.

Правила:
  * objects.csv — только object-level;
  * frames.csv — кадровый результат (включая выведенный из объектов);
  * repair.csv — только подтверждённые bad;
  * review.csv — только review;
  * triage.csv — bad + review, оба scope.
Один кадр не дублируется в repair и review. Для object/mixed источниками служат
объекты; кадровая строка добавляется только если другого представления нет.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from results import derive_frame, get_frame, get_objects, sanitize_objects, triage_route, valid_bbox

OURS = {"file", "scope", "object_n", "object_id", "type", "type_ru", "status", "damage", "damage_ru",
        "description", "bbox", "review", "review_reasons", "triage", "frame_status", "frame_source",
        "n_objects", "n_invalid_bbox", "error", "bad_json", "seconds", "cost"}


def _meta(meta: Mapping[str, Any] | None) -> dict:
    return {("meta_" + k if k in OURS else k): v for k, v in (meta or {}).items()}


def _joined_reasons(obj: Mapping[str, Any]) -> str:
    return ";".join(str(x) for x in (obj.get("review_reasons") or []))


def object_row(file: str, obj: Mapping[str, Any], pred: Mapping[str, Any], meta: Mapping[str, Any] | None,
               names: Mapping[str, Any] | None, index: int) -> dict:
    names = names or {}
    status = str(obj.get("status") or "review")
    damage = obj.get("damage") if status == "bad" else None
    box = valid_bbox(obj.get("bbox"))
    return {
        "file": file, "scope": "object", "object_n": index, "object_id": obj.get("object_id", ""),
        **_meta(meta),
        "type": obj.get("type") or "", "type_ru": names.get(obj.get("type"), obj.get("type") or ""),
        "status": status, "damage": damage or "",
        "damage_ru": names.get(damage, damage or "") if status == "bad" else "",
        "description": str(obj.get("description") or ""),
        "bbox": " ".join(str(v) for v in box) if box else "",
        "review": status == "review", "review_reasons": _joined_reasons(obj),
        "triage": "review" if status == "review" else ("repair" if status == "bad" else "none"),
        "frame_status": (get_frame(pred) or {}).get("status", ""),
        "frame_source": (get_frame(pred) or {}).get("source", ""),
        "n_objects": len(get_objects(pred)), "n_invalid_bbox": len(pred.get("invalid_bbox_objects") or []),
        "error": pred.get("error") or "", "bad_json": bool(pred.get("bad_json")),
        "seconds": pred.get("seconds", 0.0), "cost": pred.get("cost", 0.0),
    }


def frame_row(file: str, pred: Mapping[str, Any], meta: Mapping[str, Any] | None,
              names: Mapping[str, Any] | None) -> dict:
    names = names or {}
    objects = get_objects(pred)
    frame = get_frame(pred)
    if frame is None:
        frame = derive_frame(objects, error=pred.get("error"), bad_json=bool(pred.get("bad_json")),
                             invalid_bbox_count=len(pred.get("invalid_bbox_objects") or []))
    status = str(frame.get("status") or "review")
    damage = frame.get("damage") if status == "bad" else None
    clean, invalid = sanitize_objects(objects)
    return {
        "file": file, "scope": "frame", "object_n": "", "object_id": "",
        **_meta(meta),
        "type": frame.get("type") or "", "type_ru": names.get(frame.get("type"), frame.get("type") or ""),
        "status": status, "damage": damage or "",
        "damage_ru": names.get(damage, damage or "") if status == "bad" else "",
        "description": str(frame.get("description") or ""),
        "bbox": "",
        "review": bool(frame.get("review") or status == "review"),
        "review_reasons": _joined_reasons(frame),
        "triage": triage_route(frame),
        "frame_status": status, "frame_source": frame.get("source", ""),
        "n_objects": len(clean), "n_invalid_bbox": len(invalid) + len(pred.get("invalid_bbox_objects") or []),
        "error": pred.get("error") or "", "bad_json": bool(pred.get("bad_json")),
        "seconds": pred.get("seconds", 0.0), "cost": pred.get("cost", 0.0),
    }


def export_rows(preds: Mapping[str, Mapping[str, Any]], meta: Mapping[str, Mapping[str, Any]] | None,
                names: Mapping[str, Any] | None) -> dict[str, list[dict]]:
    objects_rows: list[dict] = []
    frames_rows: list[dict] = []
    repair: list[dict] = []
    review: list[dict] = []
    triage: list[dict] = []
    meta = meta or {}
    for file, pred in preds.items():
        clean, _invalid = sanitize_objects(pred.get("objects"))
        obj_rows = [object_row(file, obj, pred, meta.get(file, {}), names, index)
                    for index, obj in enumerate(clean, 1)]
        f_row = frame_row(file, pred, meta.get(file, {}), names)
        objects_rows.extend(obj_rows)
        frames_rows.append(f_row)
        bad_objs = [r for r in obj_rows if r["status"] == "bad"]
        review_objs = [r for r in obj_rows if r["status"] == "review"]
        if bad_objs:
            repair.extend(bad_objs)
        elif f_row["triage"] == "repair":
            repair.append(f_row)
        if review_objs:
            review.extend(review_objs)
        elif f_row["triage"] == "review":
            review.append(f_row)
    # triage — объединение repair и review без повторов строк
    seen = set()
    for row in repair + review:
        key = (row.get("file"), row.get("scope"), row.get("object_n"), row.get("type"), row.get("bbox"))
        if key not in seen:
            seen.add(key)
            triage.append(row)
    return {"objects": objects_rows, "frames": frames_rows, "repair": repair, "review": review,
            "triage": triage}
