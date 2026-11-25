"""Метрики: предсказания против ручной разметки.

    python src/evaluate.py data/labels.json runs/.../preds.json [--split dev]

Правила (см. docs/labeling.md):
- кадры с флагом bad_frame не оцениваются;
- target_not_visible — валидный отрицательный кадр: объектов нет, предсказание на нём
  считается FP/review, а не поводом выбросить кадр;
- объекты типа "other" — «не важно»: их не обязательно находить, предсказание на их месте
  не ошибка;
- статус "unclear" — объект ищем, но в статусных метриках он не участвует;
- объекты сопоставляются жадно по IoU внутри одного типа (0.5; 0.3 отдельно);
- repair — только автоматически подтверждённые bad; triage — bad + review;
- review rate имеет два знаменателя: объекты и кадры; они публикуются отдельно.

Новые поля (triage/frame/cost) не заменяют старые: run.py и отчёты продолжают читать
repair_*, img_bad_*, obj_bad_*.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Mapping

from results import (SCHEMA_VERSION, derive_frame, get_frame, get_objects, make_frame,
                     normalize_reasons, sanitize_objects, triage_route, valid_bbox)

IGNORE_TYPES = {"other"}


def iou(a, b) -> float:
    va, vb = valid_bbox(a), valid_bbox(b)
    if va is None or vb is None:
        return 0.0
    ix = max(0.0, min(va[2], vb[2]) - max(va[0], vb[0]))
    iy = max(0.0, min(va[3], vb[3]) - max(va[1], vb[1]))
    inter = ix * iy
    union = (va[2] - va[0]) * (va[3] - va[1]) + (vb[2] - vb[0]) * (vb[3] - vb[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt: list[dict], pr: list[dict], thr: float, same_type: bool = True) -> list[tuple[int, int]]:
    """Жадное сопоставление по убыванию IoU -> пары (индекс gt, индекс pred).

    Некорректные рамки не попадают в IoU и не роняют генератор: они уже отфильтрованы
    sanitize_objects, но защита оставлена и здесь.
    """
    pairs = []
    for i, g in enumerate(gt):
        if valid_bbox(g.get("bbox")) is None:
            continue
        for j, p in enumerate(pr):
            if valid_bbox(p.get("bbox")) is None:
                continue
            if same_type and g.get("type") != p.get("type"):
                continue
            pairs.append((iou(g["bbox"], p["bbox"]), i, j))
    pairs.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def prf(tp, fp, fn) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def fbeta(beta: float, tp: int, fp: int, fn: int) -> float | None:
    if tp + fp + fn == 0:
        return None
    beta2 = beta * beta
    denom = (1 + beta2) * tp + beta2 * fn + fp
    return (1 + beta2) * tp / denom if denom else 0.0


def _status(obj: Mapping[str, Any]) -> str:
    status = str(obj.get("status") or "")
    return status if status in ("ok", "bad", "review") else "review"


def _gt_frame_status(lab: Mapping[str, Any], objects: list[dict]) -> tuple[str | None, str | None]:
    """-> (ok|bad|unclear|None, причина исключения)."""
    frame = lab.get("frame")
    if isinstance(frame, Mapping) and frame.get("status") in ("ok", "bad", "unclear"):
        return str(frame["status"]), None
    flags = lab.get("frame_flags") or []
    if "target_not_visible" in flags and objects:
        return None, "target_not_visible_with_objects"
    statuses = [_status(o) for o in objects]
    if "bad" in statuses:
        return "bad", None
    if "unclear" in statuses:
        return "unclear", None
    return "ok", None


def prediction_view(lab: Mapping[str, Any], pred: Mapping[str, Any] | None) -> dict:
    """Нормализовать старый/новый ответ в frame+objects+invalid, не падая на bbox=None."""
    if pred is None:
        return {"frame": make_frame("review", review_reasons=["missing_prediction"], source="missing"),
                "objects": [], "invalid": [], "legacy": False, "native": False,
                "missing": True, "scope": "frame"}
    raw = list(get_objects(pred))
    invalid = list(pred.get("invalid_bbox_objects") or [])
    frame = get_frame(pred)
    legacy = False
    native = frame is not None
    if frame is None and len(raw) == 1 and isinstance(raw[0], Mapping) and raw[0].get("bbox") is None:
        obj = raw[0]
        frame = make_frame(_status(obj), type=obj.get("type"), damage=obj.get("damage"),
                           description=str(obj.get("description") or ""), source="legacy")
        legacy = True
        raw = []
    clean, auto_invalid = sanitize_objects(raw)
    invalid += auto_invalid
    if frame is None:
        frame = derive_frame(clean, error=pred.get("error"), bad_json=bool(pred.get("bad_json")),
                             invalid_bbox_count=len(invalid))
    else:
        frame = make_frame(str(frame.get("status") or "review"), type=frame.get("type"),
                           damage=frame.get("damage"), description=str(frame.get("description") or ""),
                           review=frame.get("review"), review_reasons=frame.get("review_reasons"),
                           source=str(frame.get("source") or "native"))
        if invalid and not frame["review"]:
            frame["review"] = True
            frame["review_reasons"] = normalize_reasons(list(frame["review_reasons"]) + ["invalid_bbox"])
    declared = pred.get("scope")
    if declared in ("frame", "objects", "mixed"):
        scope = declared
    elif native and clean:
        scope = "mixed"
    elif native or legacy:
        scope = "frame"
    else:
        scope = "objects"
    return {"frame": frame, "objects": clean, "invalid": invalid, "legacy": legacy,
            "native": native, "missing": False, "scope": scope}


def _match_counts(gt: list[dict], pr: list[dict], thr: float) -> dict:
    m = match(gt, pr, thr)
    counts = Counter()
    for i, j in m:
        g, p = _status(gt[i]), _status(pr[j])
        counts[f"{g}_to_{p}"] += 1
    matched_gt = {i for i, _ in m}
    matched_pr = {j for _, j in m}
    for i, g in enumerate(gt):
        if i not in matched_gt and _status(g) == "bad":
            counts["bad_to_missed"] += 1
    for j, p in enumerate(pr):
        if j not in matched_pr and _status(p) == "bad":
            counts["spurious_bad"] += 1
        elif j not in matched_pr and _status(p) == "review":
            counts["spurious_review"] += 1
    return {"matches": m, "counts": counts}


def evaluate(labels: list[dict], preds: dict[str, dict], thr: float = 0.5,
             cost: Mapping[str, float] | None = None) -> dict:
    cost_cfg = dict(cost or {})
    has_cost = all(k in cost_cfg for k in ("fn", "fp", "review"))
    det = {thr: Counter(), 0.3: Counter()}
    per_type: dict[str, Counter] = {}
    counts = Counter()
    status_conf = Counter()
    type_conf = Counter()
    damage_conf = Counter()
    frame = Counter()
    review_reasons = Counter()
    invalid_bbox = invalid_gt = missing = legacy = bad_json = errors = skipped = 0
    n_bad_gt = gt_objects = 0
    dmg_ok = dmg_n = 0
    review_objs = total_pr = repair_pred_total = 0
    secs, api_cost = [], 0.0
    per_image = []
    invalid_labels: list[dict] = []
    scopes = set()

    for lab in labels:
        if "bad_frame" in (lab.get("frame_flags") or []):
            skipped += 1
            continue
        pred = preds.get(lab["file"])
        view = prediction_view(lab, pred)
        scopes.add(view["scope"])
        missing += view["missing"]
        legacy += view["legacy"]
        invalid_bbox += len(view["invalid"])
        raw_gt, gt_invalid = sanitize_objects(list(lab.get("objects") or []))
        invalid_gt += len(gt_invalid)
        if gt_invalid:
            invalid_labels.append({"file": lab["file"], "reason": "invalid_gt_bbox", "count": len(gt_invalid)})
        ignore = [o for o in raw_gt if o.get("type") in IGNORE_TYPES]
        gt = [o for o in raw_gt if o.get("type") not in IGNORE_TYPES]
        pr = [q for q in view["objects"]
              if not any(iou(q.get("bbox"), o.get("bbox")) >= 0.3 for o in ignore)
              or any(iou(q.get("bbox"), g.get("bbox")) >= 0.3 for g in gt)]
        gt_status, label_problem = _gt_frame_status(lab, gt)
        if label_problem:
            invalid_labels.append({"file": lab["file"], "reason": label_problem})
        n_bad_gt += sum(_status(o) == "bad" for o in gt)
        gt_objects += len(gt)
        if pred is not None:
            bad_json += bool(pred.get("bad_json"))
            errors += bool(pred.get("error"))
            api_cost += float(pred.get("cost") or 0.0)
            if "seconds" in pred:
                secs.append(float(pred["seconds"]))

        info = _match_counts(gt, pr, thr)
        m = info["matches"]
        counts.update(info["counts"])
        for t in det:
            det[t]["tp"] += len(match(gt, pr, t))
            det[t]["gt"] += len(gt)
            det[t]["pr"] += len(pr)
        for o in gt:
            per_type.setdefault(o.get("type"), Counter())["gt"] += 1
        for q in pr:
            per_type.setdefault(q.get("type"), Counter())["pr"] += 1
        for i, _ in m:
            per_type[gt[i].get("type")]["tp"] += 1
        for i, j in m:
            g, p = gt[i], pr[j]
            gs, ps = _status(g), _status(p)
            if gs in ("ok", "bad", "unclear"):
                status_conf[(gs, ps)] += 1
            damage_conf[(g.get("damage") or "ok", p.get("damage") or "ok")] += 1
            if gs == "bad" and ps == "bad":
                dmg_n += 1
                dmg_ok += g.get("damage") == p.get("damage")
        type_conf.update(Counter((gt[i].get("type"), pr[j].get("type"))
                                 for i, j in match(gt, pr, thr, same_type=False)))
        for q in pr:
            if _status(q) == "review":
                review_reasons.update(q.get("review_reasons") or ["low_confidence"])
        review_objs += sum(_status(q) == "review" for q in pr)
        repair_pred_total += sum(_status(q) == "bad" for q in pr)
        total_pr += len(pr)
        if gt_status in ("ok", "bad"):
            route = triage_route(view["frame"])
            key = {"review": "frame_bad_to_review" if gt_status == "bad" else "frame_ok_to_review",
                   "repair": "frame_bad_to_bad" if gt_status == "bad" else "frame_ok_to_bad",
                   "none": "frame_bad_to_ok" if gt_status == "bad" else "frame_ok_to_ok"}[route]
            frame[key] += 1
            if route == "review":
                review_reasons.update(view["frame"].get("review_reasons") or ["low_confidence"])
        elif gt_status == "unclear":
            frame["gt_unclear"] += 1
        route = triage_route(view["frame"])
        per_image.append({
            "file": lab["file"], "fn": len(gt) - len(m), "fp": len(pr) - len(m),
            "status_err": sum(1 for i, j in m if _status(gt[i]) != _status(pr[j])
                              and _status(gt[i]) in ("ok", "bad")),
            "missed_bad": bool(gt_status == "bad" and route == "none"),
            "needs_review": route == "review",
            "gt_status": gt_status, "pred_status": view["frame"].get("status"), "route": route,
        })

    n = len(labels) - skipped
    bad_to_bad = counts["bad_to_bad"]
    bad_to_review = counts["bad_to_review"]
    bad_to_ok = counts["bad_to_ok"]
    bad_to_missed = counts["bad_to_missed"]
    ok_to_bad = counts["ok_to_bad"]
    ok_to_review = counts["ok_to_review"]
    ok_to_ok = counts["ok_to_ok"]
    triage_tp = bad_to_bad + bad_to_review
    triage_fp = (ok_to_bad + ok_to_review + counts["spurious_bad"] + counts["spurious_review"])
    triage_fn = bad_to_ok + bad_to_missed
    triage_unclear = counts["unclear_to_bad"] + counts["unclear_to_review"]
    triage_pred = triage_tp + triage_fp + triage_unclear
    tprec = triage_tp / (triage_tp + triage_fp) if triage_tp + triage_fp else 0.0
    trec = triage_tp / n_bad_gt if n_bad_gt else 0.0
    repair_pred = repair_pred_total
    repair_tp = bad_to_bad
    repair_fp = repair_pred - repair_tp
    repair_fn = n_bad_gt - repair_tp
    rp, rr, rf = prf(repair_tp, repair_fp, repair_fn)

    gt_bad_frames = frame["frame_bad_to_bad"] + frame["frame_bad_to_review"] + frame["frame_bad_to_ok"]
    gt_ok_frames = frame["frame_ok_to_bad"] + frame["frame_ok_to_review"] + frame["frame_ok_to_ok"]
    review_frames = frame["frame_bad_to_review"] + frame["frame_ok_to_review"]
    n_fn, n_fp, n_review = frame["frame_bad_to_ok"], frame["frame_ok_to_bad"], review_frames
    operational_cost = (cost_cfg["fn"] * n_fn + cost_cfg["fp"] * n_fp + cost_cfg["review"] * n_review) \
        if has_cost else None
    # review-FP уже входит в review_objs: не списываем его ещё раз как FP
    triage_fp_auto = ok_to_bad + counts["spurious_bad"]
    triage_fp_review = ok_to_review + counts["spurious_review"]
    triage_cost = (cost_cfg["fn"] * triage_fn + cost_cfg["fp"] * triage_fp_auto +
                   cost_cfg["review"] * review_objs) if has_cost else None
    review_rate_objects = review_objs / total_pr if total_pr else None
    review_rate_frames = review_frames / n if n else None
    ip, ir, if1 = prf(frame["frame_bad_to_bad"], frame["frame_ok_to_bad"],
                      frame["frame_bad_to_ok"] + frame["frame_bad_to_review"])
    sp, sr, sf1 = prf(status_conf[("bad", "bad")], status_conf[("ok", "bad")],
                      status_conf[("bad", "ok")] + status_conf[("bad", "review")])
    if "frame" in scopes and "objects" in scopes:
        eval_mode = "mixed"
    elif "mixed" in scopes:
        eval_mode = "mixed"
    elif scopes == {"frame"}:
        eval_mode = "frame"
    else:
        eval_mode = "objects"

    out = {
        "schema_version": SCHEMA_VERSION, "evaluation_mode": eval_mode,
        "images": n, "skipped_bad_frames": skipped, "gt_objects": gt_objects,
        "gt_bad_objects": n_bad_gt, "gt_frames": gt_bad_frames + gt_ok_frames,
        "gt_bad_frames": gt_bad_frames, "gt_unclear_frames": frame["gt_unclear"],
        "invalid_bbox_objects": invalid_bbox, "invalid_gt_objects": invalid_gt,
        "invalid_labels": invalid_labels, "missing_predictions": missing,
        "legacy_photo_dino": legacy,
    }
    if eval_mode in ("objects", "mixed"):
        for t, c in det.items():
            sfx = "" if t == thr else f"@{t}"
            out[f"det_precision{sfx}"] = c["tp"] / c["pr"] if c["pr"] else 0.0
            out[f"det_recall{sfx}"] = c["tp"] / c["gt"] if c["gt"] else 0.0
        out["det_recall_anytype"] = _any_type_recall(labels, preds, thr)
    else:
        for t in det:
            sfx = "" if t == thr else f"@{t}"
            out[f"det_precision{sfx}"] = None
            out[f"det_recall{sfx}"] = None
        out["det_recall_anytype"] = None
    out |= {
        "repair_tp": repair_tp if eval_mode != "frame" else None,
        "repair_fp": repair_fp if eval_mode != "frame" else None,
        "repair_fn": repair_fn if eval_mode != "frame" else None,
        "repair_precision": rp if eval_mode != "frame" else None,
        "repair_recall": rr if eval_mode != "frame" else None,
        "repair_f1": rf if eval_mode != "frame" else None,
        "repair_f2": fbeta(2, repair_tp, repair_fp, repair_fn) if eval_mode != "frame" else None,
        "triage_pred": triage_pred, "triage_tp": triage_tp, "triage_fp": triage_fp,
        "triage_fn": triage_fn, "triage_precision": tprec, "triage_recall": trec,
        "triage_f2": fbeta(2, triage_tp, triage_fp, triage_fn),
        "triage_fp_matched": ok_to_bad + ok_to_review,
        "triage_fp_unmatched": counts["spurious_bad"] + counts["spurious_review"],
        "triage_fp_auto": triage_fp_auto, "triage_fp_review": triage_fp_review,
        "spurious_bad": counts["spurious_bad"], "spurious_review": counts["spurious_review"],
        "missed_bad_after_triage": triage_fn,
        "bad_to_bad": bad_to_bad, "bad_to_review": bad_to_review, "bad_to_ok": bad_to_ok,
        "bad_to_missed": bad_to_missed,
        "ok_to_bad": ok_to_bad, "ok_to_review": ok_to_review, "ok_to_ok": ok_to_ok,
        "review_objects": review_objs, "review_rate_objects": review_rate_objects,
        "review_frames": review_frames, "review_rate": review_rate_frames,
        "review_reasons": dict(sorted(review_reasons.items())),
        "frame_bad_to_bad": frame["frame_bad_to_bad"],
        "frame_bad_to_review": frame["frame_bad_to_review"],
        "frame_bad_to_ok": frame["frame_bad_to_ok"],
        "frame_ok_to_bad": frame["frame_ok_to_bad"],
        "frame_ok_to_review": frame["frame_ok_to_review"],
        "frame_ok_to_ok": frame["frame_ok_to_ok"],
        "frame_bad_recall": ir, "frame_bad_precision": ip, "frame_bad_f1": if1,
        "frame_triage_recall": ((frame["frame_bad_to_bad"] + frame["frame_bad_to_review"]) /
                                gt_bad_frames) if gt_bad_frames else None,
        "operational_cost": operational_cost, "triage_cost": triage_cost,
        "baselines": _baselines(gt_bad_frames, gt_ok_frames, cost_cfg if has_cost else None),
        "per_type": {t: {"gt": c["gt"], "pred": c["pr"],
                         "precision": round(c["tp"] / c["pr"], 3) if c["pr"] else 0.0,
                         "recall": round(c["tp"] / c["gt"], 3) if c["gt"] else 0.0}
                     for t, c in sorted(per_type.items())} if eval_mode != "frame" else {},
        "type_confusion": {f"{a} -> {b}": c for (a, b), c in sorted(type_conf.items()) if a != b}
        if eval_mode != "frame" else {},
        "damage_confusion": {f"{a} -> {b}": c for (a, b), c in sorted(damage_conf.items())},
        "damage_acc": (dmg_ok / dmg_n) if dmg_n else (None if eval_mode == "frame" else 0.0),
        "img_bad_recall": ir, "img_bad_precision": ip, "img_bad_f1": if1,
        "img_accuracy": ((frame["frame_bad_to_bad"] + frame["frame_ok_to_ok"]) /
                         (gt_bad_frames + gt_ok_frames)) if gt_bad_frames + gt_ok_frames else 0.0,
        "img_bad_f2": fbeta(2, frame["frame_bad_to_bad"], frame["frame_ok_to_bad"],
                            frame["frame_bad_to_ok"] + frame["frame_bad_to_review"]),
        "obj_bad_recall": sr if eval_mode != "frame" else None,
        "obj_bad_precision": sp if eval_mode != "frame" else None,
        "obj_bad_f1": sf1 if eval_mode != "frame" else None,
        "obj_bad_f2": fbeta(2, status_conf[("bad", "bad")], status_conf[("ok", "bad")],
                            status_conf[("bad", "ok")] + status_conf[("bad", "review")])
        if eval_mode != "frame" else None,
        "bad_json_rate": bad_json / n if n else 0.0, "error_rate": errors / n if n else 0.0,
        "sec_per_image": sum(secs) / len(secs) if secs else 0.0, "cost_usd": api_cost,
        "worst": sorted((x for x in per_image if x["fn"] + x["fp"] + x["status_err"] or x["missed_bad"]
                         or x["needs_review"]),
                        key=lambda x: (x["missed_bad"], x["needs_review"], x["fn"] + x["fp"] + x["status_err"]),
                        reverse=True)[:20],
    }
    return out


def _any_type_recall(labels: list[dict], preds: dict, thr: float) -> float:
    tp = total = 0
    for lab in labels:
        if "bad_frame" in (lab.get("frame_flags") or []):
            continue
        view = prediction_view(lab, preds.get(lab["file"]))
        raw_gt, _ = sanitize_objects(list(lab.get("objects") or []))
        gt = [o for o in raw_gt if o.get("type") not in IGNORE_TYPES]
        total += len(gt)
        tp += len(match(gt, view["objects"], thr, same_type=False))
    return tp / total if total else 0.0


def _baselines(gt_bad: int, gt_ok: int, cost: Mapping[str, float] | None) -> dict:
    total = gt_bad + gt_ok
    def c(fn, fp, review):
        return (cost["fn"] * fn + cost["fp"] * fp + cost["review"] * review) if cost else None
    return {
        "all_review": {"N_FN": 0, "N_FP": 0, "N_review": total, "cost": c(0, 0, total)},
        "all_bad": {"N_FN": 0, "N_FP": gt_ok, "N_review": 0, "cost": c(0, gt_ok, 0)},
        "all_ok": {"N_FN": gt_bad, "N_FP": 0, "N_review": 0, "cost": c(gt_bad, 0, 0)},
    }


def load_labels(path, split: str | None = None) -> list[dict]:
    labels = json.load(open(path))
    return [x for x in labels if split is None or x.get("split") == split]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("preds")
    ap.add_argument("--split")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--cost-json", help="JSON с {fn, fp, review}; по умолчанию [cost] config.toml")
    args = ap.parse_args()
    cfg_cost = None
    if args.cost_json:
        cfg_cost = json.loads(args.cost_json)
    else:
        try:
            import tomllib
            from pathlib import Path
            root = Path(__file__).resolve().parents[1]
            cfg_cost = tomllib.loads((root / "config.toml").read_text(encoding="utf-8")).get("cost")
        except Exception:  # noqa: BLE001
            cfg_cost = None
    m = evaluate(load_labels(args.labels, args.split), json.load(open(args.preds)), args.iou, cfg_cost)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
