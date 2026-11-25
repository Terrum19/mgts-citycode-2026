"""Image-level отбор связки для final.py: operational cost вместо macro-F1.

final.py работает с папками «класс на фото», поэтому triage/cost здесь считаются по
кадрам, а не по объектам. Object-level triage остаётся в evaluate.py для общих кадров.

Baselines считаются на фактическом распределении validation:
  all_review — всё в review, FN=FP=0, review=N;
  all_bad    — всё в repair, FN=0, FP=N_ok, review=0;
  all_ok     — всё ok, FN=N_bad, FP=0, review=0.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

BASE_COMPLEXITY = {"logreg": 0, "mean": 1, "tail": 2, "vlm": 3}
DEFAULT_COST = {"fn": 20.0, "fp": 2.0, "review": 1.0}


def frame_metrics(y_bad: Sequence[bool], decisions: Sequence[Mapping[str, Any]],
                  cost: Mapping[str, float] | None = None) -> dict:
    """Метрики по кадрам. decision: {status, review, review_reasons}."""
    if len(y_bad) != len(decisions):
        raise ValueError("y_bad и decisions должны быть одной длины")
    cost = dict(cost or DEFAULT_COST)
    counts = {"bad_to_bad": 0, "bad_to_review": 0, "bad_to_ok": 0,
              "ok_to_bad": 0, "ok_to_review": 0, "ok_to_ok": 0}
    support_bad = sum(bool(x) for x in y_bad)
    support_ok = len(y_bad) - support_bad
    for gt_bad, decision in zip(y_bad, decisions):
        status = str(decision.get("status") or "review")
        route = "review" if decision.get("review") or status == "review" else (
            "repair" if status == "bad" else "none")
        key = ("bad" if gt_bad else "ok") + "_to_" + {"review": "review", "repair": "bad",
                                                      "none": "ok"}[route]
        counts[key] += 1
    n_fn = counts["bad_to_ok"]
    n_fp = counts["ok_to_bad"]
    n_review = counts["bad_to_review"] + counts["ok_to_review"]
    total = len(y_bad)
    triage_tp = counts["bad_to_bad"] + counts["bad_to_review"]
    triage_fp = counts["ok_to_bad"] + counts["ok_to_review"]
    triage_fn = counts["bad_to_ok"]
    auto_precision = counts["bad_to_bad"] / (counts["bad_to_bad"] + counts["ok_to_bad"]) \
        if counts["bad_to_bad"] + counts["ok_to_bad"] else None
    return {
        **counts,
        "support_bad": support_bad, "support_ok": support_ok,
        "N_FN": n_fn, "N_FP": n_fp, "N_review": n_review,
        "operational_cost": cost["fn"] * n_fn + cost["fp"] * n_fp + cost["review"] * n_review,
        "auto_bad_recall": counts["bad_to_bad"] / support_bad if support_bad else None,
        "auto_bad_precision": auto_precision,
        "triage_recall": triage_tp / support_bad if support_bad else None,
        "triage_precision": triage_tp / (triage_tp + triage_fp) if triage_tp + triage_fp else None,
        "triage_fn": triage_fn,
        "review_rate": n_review / total if total else None,
        "coverage": (counts["bad_to_bad"] + counts["ok_to_ok"]) / total if total else None,
    }


def baselines(y_bad: Sequence[bool], cost: Mapping[str, float] | None = None) -> dict:
    cost = dict(cost or DEFAULT_COST)
    total = len(y_bad)
    bad = sum(bool(x) for x in y_bad)
    ok = total - bad
    return {
        "all_review": frame_metrics(y_bad, [{"status": "review", "review": True}] * total, cost),
        "all_bad": frame_metrics(y_bad, [{"status": "bad", "review": False}] * total, cost),
        "all_ok": frame_metrics(y_bad, [{"status": "ok", "review": False}] * total, cost),
    }


def _support_ok(cfg: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    min_bad = int(cfg.get("min_bad", 0) or 0)
    min_ok = int(cfg.get("min_ok", 0) or 0)
    if int(row.get("support_bad") or 0) < min_bad:
        reasons.append(f"support_bad {row.get('support_bad')} < {min_bad}")
    if int(row.get("support_ok") or 0) < min_ok:
        reasons.append(f"support_ok {row.get('support_ok')} < {min_ok}")
    return not reasons, reasons


def select_candidate(rows: Sequence[Mapping[str, Any]], selection_cfg: Mapping[str, Any] | None = None,
                     baselines_rows: Mapping[str, Mapping[str, Any]] | None = None) -> dict:
    """Выбрать candidate по ограничениям, cost и детерминированному tie-break."""
    cfg = dict(selection_cfg or {})
    min_recall = float(cfg.get("min_triage_recall", 0.0) or 0.0)
    max_review = float(cfg.get("max_review_rate", 1.0) or 1.0)
    considered, eligible, rejected = [], [], []
    for row in rows:
        item = dict(row)
        ok_support, support_reasons = _support_ok(cfg, item)
        reasons = list(support_reasons)
        if item.get("serializable") is False:
            reasons.append("no_serializable_production_path")
        recall = item.get("triage_recall")
        review = item.get("review_rate")
        if recall is None:
            reasons.append("triage_recall=None")
        elif float(recall) < min_recall:
            reasons.append(f"triage_recall {float(recall):.3f} < {min_recall:.3f}")
        if review is None:
            reasons.append("review_rate=None")
        elif float(review) > max_review:
            reasons.append(f"review_rate {float(review):.3f} > {max_review:.3f}")
        item["reject_reasons"] = reasons
        considered.append(item)
        if reasons:
            rejected.append(item)
        else:
            eligible.append(item)

    if not eligible:
        support_bad = max((int(r.get("support_bad") or 0) for r in considered), default=0)
        support_ok = max((int(r.get("support_ok") or 0) for r in considered), default=0)
        reason = "support_failed" if (support_bad < int(cfg.get("min_bad", 0) or 0) or
                                      support_ok < int(cfg.get("min_ok", 0) or 0)) else "constraints_failed"
        return {"status": "selection_failed", "chosen": None, "reason": reason,
                "support_bad": support_bad, "support_ok": support_ok,
                "constraints": {"min_bad": cfg.get("min_bad"), "min_ok": cfg.get("min_ok"),
                                "min_triage_recall": min_recall, "max_review_rate": max_review},
                "considered": considered, "eligible": [], "rejected": rejected,
                "baselines": baselines_rows or {}}

    def key(row):
        return (
            round(float(row.get("operational_cost") or 0.0), 12),
            -float(row.get("triage_recall") or 0.0),
            float(row.get("review_rate") or 0.0),
            -float(row.get("auto_bad_recall") or 0.0),
            bool(row.get("uses_vlm")),
            int(BASE_COMPLEXITY.get(str(row.get("base")), 9)),
            str(row.get("name") or ""),
        )

    chosen = min(eligible, key=key)
    return {"status": "ok", "chosen": chosen,
            "reason": "min operational_cost with deterministic tie-break",
            "support_bad": int(chosen.get("support_bad") or 0),
            "support_ok": int(chosen.get("support_ok") or 0),
            "constraints": {"min_bad": cfg.get("min_bad"), "min_ok": cfg.get("min_ok"),
                            "min_triage_recall": min_recall, "max_review_rate": max_review},
            "considered": considered, "eligible": eligible, "rejected": rejected,
            "baselines": baselines_rows or {}}
