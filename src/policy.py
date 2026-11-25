"""Общая decision policy для validation, test-инференса, CropDino и PhotoDino.

Раньше final.py принимал решение по argmax вероятностей, а PhotoDino — по отдельной
копии каскада. Здесь одна функция ``initial`` строит решение по вероятностям, а
``finalize`` применяет к нему VLM-ответ (resolve спорного или description-only).

Режимы:
  * ``legacy`` — прежнее поведение: argmax по поддержанным классам, VLM добавляется
    к вероятности класса с весом qwen_weight;
  * ``cascade`` — суммарные P(bad)/P(ok), уверенность статуса, margin, ``min_support``;
    VLM разрешает review и пишет описания подтверждённых bad.

Модуль stdlib-only: тяжёлые вероятности приходят снаружи.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from cascade import answer_error, as_tuple, decide, merge_vlm, pick, summarize
from results import normalize_reasons

POLICY_KEYS = (
    "mode", "cascade", "qwen_weight", "class_policy", "min_support", "min_confidence",
    "p_bad_min", "p_bad_max_ok", "status_conf_min", "margin_min",
    "describe_bad", "not_target", "hint_mode",
)
DEFAULT_POLICY = {
    "cascade": False,
    "qwen_weight": 0.0,
    "class_policy": "detector",
    "min_support": 2,
    "min_confidence": 0.0,
    "p_bad_min": 0.60,
    "p_bad_max_ok": 0.40,
    "status_conf_min": 0.65,
    "margin_min": 0.10,
    "describe_bad": False,
    "not_target": "review",
    "hint_mode": "none",
}


def semantic_policy(cfg: Mapping[str, Any] | None) -> dict:
    """Выделить из варианта только семантические ключи, которые должны жить в checkpoint."""
    cfg = cfg or {}
    out = {key: cfg[key] for key in POLICY_KEYS if key in cfg}
    for key, value in DEFAULT_POLICY.items():
        out.setdefault(key, value)
    return out


def effective_policy(saved: Mapping[str, Any] | None, runtime_cfg: Mapping[str, Any]) -> dict:
    """Семантическая policy checkpoint vs текущий конфиг.

    Старый checkpoint (без policy) работает по runtime config. Новый checkpoint — по
    сохранённой policy; конфликтующий semantic-ключ в config — явная ошибка, а не
    молчаливая подмена уже измеренной конфигурации. Deployment-ключи (device, vlm,
    base_url, checkpoint) сюда не входят.
    """
    runtime = dict(runtime_cfg or {})
    if not saved:
        return runtime
    if "policy" in saved and isinstance(saved["policy"], Mapping) and not any(k in saved for k in POLICY_KEYS):
        saved = saved["policy"]
    saved = {key: value for key, value in dict(saved).items() if key in POLICY_KEYS}
    if not saved:
        return runtime
    override = bool(runtime.get("override_checkpoint_policy"))
    conflicts = []
    for key, value in saved.items():
        if key in runtime and runtime[key] != value:
            # mode и cascade — одно и то же; сравниваем нормализованно.
            if key in ("mode", "cascade") and _mode(runtime) == _mode(saved):
                continue
            conflicts.append(f"{key}: checkpoint={value!r}, config={runtime[key]!r}")
    if conflicts and not override:
        raise ValueError("Конфиг конфликтует с сохранённой decision policy: " + "; ".join(conflicts)
                         + ". Переобучите classifier или задайте override_checkpoint_policy=true.")
    out = dict(runtime)
    out.update(saved)
    out["cascade"] = saved.get("cascade", _mode(saved) == "cascade")
    out["mode"] = "cascade" if out["cascade"] else "legacy"
    return out


def _mode(cfg: Mapping[str, Any] | None) -> str:
    cfg = cfg or {}
    if cfg.get("mode") in ("legacy", "cascade"):
        return str(cfg["mode"])
    return "cascade" if cfg.get("cascade") else "legacy"


def decision_reasons(summary: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[str]:
    """Почему каскад ушёл в review — машинные причины для CSV/UI."""
    if not summary.get("ready"):
        if not summary.get("eligible"):
            return ["low_support"]
        return ["missing_status"]
    if float(summary.get("status_confidence", 0.0)) < float(cfg.get("status_conf_min", 0.65)):
        return ["low_status_confidence"]
    if float(summary.get("margin", 0.0)) < float(cfg.get("margin_min", 0.10)):
        return ["low_margin"]
    if float(summary.get("p_bad", 0.0)) > float(cfg.get("p_bad_max_ok", 0.40)) and \
            float(summary.get("p_bad", 0.0)) < float(cfg.get("p_bad_min", 0.60)):
        return ["grey_zone"]
    return []


def _validate(labels: Sequence[str], classes: Sequence[Sequence[Any]], scores: Sequence[float]) -> None:
    if not (len(labels) == len(classes) == len(scores)):
        raise ValueError("labels, classes и scores должны быть одной длины")
    for value in scores:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"вероятности должны быть конечными и неотрицательными, получено {value!r}")


def initial(
    labels: Sequence[str],
    classes: Sequence[Sequence[Any]],
    scores: Sequence[float],
    *,
    mode: str,
    cfg: Mapping[str, Any] | None = None,
    support: Mapping[str, int] | None = None,
    type_hint: str | None = None,
    class_policy: str = "detector",
) -> dict:
    """Решение только по вероятностям: статус, тип, damage, причины, нужен ли VLM."""
    cfg = dict(cfg or {})
    labels = [str(x) for x in labels]
    classes = [as_tuple(c) for c in classes]
    scores = [float(x) for x in scores]
    _validate(labels, classes, scores)
    if mode not in ("legacy", "cascade"):
        raise ValueError("mode должен быть legacy или cascade")
    if class_policy not in ("joint", "detector"):
        raise ValueError("class_policy должен быть joint или detector")
    if mode == "cascade":
        summary = summarize(classes, scores, labels=labels, support=support, type_hint=type_hint,
                            policy=class_policy, min_support=int(cfg.get("min_support", 2)))
        status = decide(summary, cfg)
        typ, damage = pick(summary, classes, status) if status in ("ok", "bad") else (None, None)
        reasons = decision_reasons(summary, cfg) if status == "review" else []
        label = None
        if status in ("ok", "bad"):
            index = summary.get("bad_index" if status == "bad" else "ok_index")
            label = labels[index] if index is not None else None
        return {
            "mode": "cascade", "status": status, "type": typ, "damage": damage, "label": label,
            "description": "", "review_reasons": normalize_reasons(reasons),
            "needs_vlm": bool(status == "review" or (status == "bad" and cfg.get("describe_bad"))),
            "summary": summary, "labels": labels, "classes": classes, "scores": scores,
            "eligible": list(summary.get("eligible") or []), "best_index": summary.get("top_index"),
            "class_policy": class_policy, "type_hint": type_hint,
        }

    # legacy: argmax по классам, с ограничениями support/min_confidence/type_hint
    eligible = [i for i, cls in enumerate(classes)
                if class_policy == "joint" or not type_hint or cls[0] == type_hint]
    min_support = int(cfg.get("min_support", 0) or 0)
    if support is not None and min_support:
        eligible = [i for i in eligible if int(support.get(labels[i], 0)) >= min_support]
    if not eligible:
        return {
            "mode": "legacy", "status": "review", "type": None, "damage": None, "label": None,
            "description": "", "review_reasons": ["low_support"], "needs_vlm": False,
            "summary": {}, "labels": labels, "classes": classes, "scores": scores,
            "eligible": [], "best_index": None, "class_policy": class_policy, "type_hint": type_hint,
        }
    best = max(eligible, key=lambda i: scores[i])
    kind, status, damage = classes[best]
    reasons: list[str] = []
    if float(scores[best]) < float(cfg.get("min_confidence", 0.0) or 0.0):
        status, damage = "review", None
        reasons.append("low_confidence")
    if support is not None:
        supported = {classes[i][1] for i in eligible if classes[i][0] == kind}
        if supported != {"ok", "bad"}:
            status, damage = "review", None
            reasons.append("missing_status")
    return {
        "mode": "legacy", "status": status, "type": kind, "label": labels[best],
        "damage": damage if status == "bad" else None, "description": "",
        "review_reasons": normalize_reasons(reasons),
        "needs_vlm": float(cfg.get("qwen_weight", 0.0) or 0.0) > 0,
        "summary": {}, "labels": labels, "classes": classes, "scores": scores,
        "eligible": eligible, "best_index": best, "class_policy": class_policy, "type_hint": type_hint,
    }


def _vlm_label_status(label: str | None, labels: Sequence[str], classes: Sequence[Sequence[Any]]) -> tuple[str, str, str | None]:
    if label not in labels:
        return "review", "", None
    index = labels.index(label)
    _, status, damage = classes[index]
    return status, classes[index][0], damage if status == "bad" else None


def _resolve_whole_frame(decision: dict, vlm: Mapping[str, Any]) -> tuple[str, str | None, str | None, list[str]]:
    """VLM выбрала класс для целого кадра (final.py validation/test/PhotoDino)."""
    label = vlm.get("label")
    status, typ, damage = _vlm_label_status(label if isinstance(label, str) else None,
                                           decision["labels"], decision["classes"])
    if status in ("ok", "bad"):
        return status, typ, damage, []
    if vlm.get("error"):
        return "review", None, None, ["vlm_error"]
    if vlm.get("bad_json"):
        return "review", None, None, ["bad_json"]
    return "review", None, None, ["taxonomy_mismatch"]


def finalize(
    decision: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any] | None = None,
    tax: dict | None = None,
    vlm: Mapping[str, Any] | None = None,
) -> dict | None:
    """Применить VLM-ответ к решению initial. None — находку нужно выбросить."""
    cfg = dict(cfg or {})
    vlm = dict(vlm or {})
    out = {
        "status": str(decision.get("status", "review")),
        "type": decision.get("type"),
        "damage": decision.get("damage"),
        "label": decision.get("label"),
        "description": str(decision.get("description") or ""),
        "review_reasons": normalize_reasons(decision.get("review_reasons")),
        "mode": decision.get("mode"),
    }
    if out["status"] not in ("ok", "bad", "review"):
        out["status"] = "review"
        out["review_reasons"] = normalize_reasons(out["review_reasons"] + ["low_confidence"])

    if out["mode"] == "legacy":
        weight = float(cfg.get("qwen_weight", 0.0) or 0.0)
        label = vlm.get("label")
        if weight > 0 and isinstance(label, str) and label in decision["labels"]:
            scores = list(decision["scores"])
            scores[decision["labels"].index(label)] += weight
            eligible = list(decision.get("eligible") or range(len(scores)))
            best = max(eligible, key=lambda i: scores[i])
            kind, status, damage = decision["classes"][best]
            if float(scores[best]) >= float(cfg.get("min_confidence", 0.0) or 0.0):
                out.update(status=status, type=kind, label=decision["labels"][best],
                           damage=damage if status == "bad" else None)
                out["review_reasons"] = []
        if vlm.get("error") and out["status"] == "review":
            out["review_reasons"] = normalize_reasons(out["review_reasons"] + ["vlm_error"])
        out["description"] = str(vlm.get("description") or out["description"])
        out["review"] = out["status"] == "review"
        return out

    answer = vlm.get("answer")
    label = vlm.get("label")
    not_target = str(cfg.get("not_target", "review"))
    if not_target not in ("drop", "review"):
        not_target = "review"

    if out["status"] == "review":
        if isinstance(answer, Mapping):
            why = vlm.get("error") or answer_error(answer)
            if why:
                out["review_reasons"] = normalize_reasons(out["review_reasons"] + ["vlm_error"])
            else:
                base = {"type": out["type"] or decision.get("type_hint"), "status": "review",
                        "damage": None, "description": "", "review_reasons": out["review_reasons"]}
                merged = merge_vlm(base, answer, tax or {}, not_target=not_target, authoritative=True)
                if merged is None:
                    return None
                out.update(status=merged.get("status", "review"), type=merged.get("type"),
                           damage=merged.get("damage"), description=merged.get("description", ""),
                           review_reasons=normalize_reasons(merged.get("review_reasons")))
                if answer.get("status") == "review":
                    out["review_reasons"] = normalize_reasons(out["review_reasons"] + ["low_confidence"])
        elif isinstance(label, str) and label in decision["labels"] and not vlm.get("review"):
            status, typ, damage, reasons = _resolve_whole_frame(dict(decision), vlm)
            out.update(status=status, type=typ, damage=damage,
                       label=label if status in ("ok", "bad") else None)
            out["review_reasons"] = normalize_reasons(out["review_reasons"] + reasons)
        else:
            if vlm.get("error"):
                out["review_reasons"] = normalize_reasons(out["review_reasons"] + ["vlm_error"])
            elif vlm.get("bad_json"):
                out["review_reasons"] = normalize_reasons(out["review_reasons"] + ["bad_json"])
            elif vlm.get("review") or (isinstance(label, str) and label in decision["labels"]):
                out["review_reasons"] = normalize_reasons(out["review_reasons"] + ["low_confidence"])
            elif cfg.get("vlm") and not vlm.get("available", True):
                out["review_reasons"] = normalize_reasons(out["review_reasons"] + ["vlm_unavailable"])
    elif out["status"] == "bad" and cfg.get("describe_bad"):
        if isinstance(answer, Mapping) and not (vlm.get("error") or vlm.get("bad_json")):
            # description-only: не меняем статус/тип и не удаляем подтверждённый bad
            out["description"] = str(answer.get("description") or "")
        elif vlm.get("error") or vlm.get("bad_json"):
            pass  # ошибка необязательного описания не понижает bad

    if out["status"] == "review":
        out["damage"] = None
        if not out["review_reasons"]:
            out["review_reasons"] = ["low_confidence"]
    if out["status"] in ("ok", "bad") and out.get("type") is None:
        out["status"] = "review"
        out["review_reasons"] = normalize_reasons(out["review_reasons"] + ["taxonomy_mismatch"])
    out["review"] = out["status"] == "review"
    return out
