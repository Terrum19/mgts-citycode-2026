"""Каскад «быстрый DINOv2 → тяжёлая VLM только на спорное»: решение по объекту.

Модуль без torch/sklearn/requests — только stdlib: логику можно проверить тестами без
тяжёлых зависимостей и без сети.

Что считается (внутри одного типа объекта):
  P(bad) / P(ok)     — СУММА вероятностей всех bad-/ok-классов, а не вероятность одного
                       узкого кортежа (type, status, damage). Объект, где модель колеблется
                       между cover_broken 0.42 и cover_open 0.42, — это P(bad)=0.84, а не 0.42.
  status_confidence  — max(P(ok), P(bad)): насколько уверенно мы знаем СТАТУС.
  margin             — отрыв МЕЖДУ СТАТУСАМИ |P(bad) - P(ok)|. Именно он измеряет спор
                       «исправен или сломан»; внутренний разрыв между двумя похожими
                       повреждениями одного статуса спором не считается.

Тип: при `class_policy = "detector"` и известном типе от детектора/организатора вероятности
суммируются только по классам этого типа. Иначе тип выбирается по лучшему поддержанному
кортежу, и все показатели считаются уже внутри выбранного типа — иначе массы разных типов
складывались бы в искусственно высокие P(bad)/P(ok).

Правило decide():
  нет обоих статусов среди поддержанных классов      -> review
  status_confidence < status_conf_min                -> review
  margin < margin_min                                -> review
  P(bad) >= p_bad_min                                -> bad
  P(bad) <= p_bad_max_ok                             -> ok
  иначе (серая зона)                                 -> review

Вероятности не перенормируются после отсева классов: масса неподдержанного класса должна
снижать уверенность, а не исчезать. Пороги — стартовые (config.toml), подбирать на dev.
"""
import json

from results import normalize_reasons

# Классы обученного классификатора. review сюда НЕ входит: summarize() проверяет
# готовность по наличию обученных ok/bad, а не по статусу ответа VLM.
STATUSES = ("ok", "bad")
ANSWER_STATUSES = ("ok", "bad", "review")


def as_tuple(cls) -> list:
    """Класс из checkpoint -> [type, status, damage]; принимает кортеж, список или JSON-строку."""
    if isinstance(cls, str):
        cls = json.loads(cls)
    return list(cls)


def summarize(classes, scores, *, labels=None, support=None, type_hint=None,
              policy: str = "detector", min_support: int = 2) -> dict:
    """Свернуть вероятности классов в признаки для решения.

    classes  — список [type, status, damage] (или JSON-строк, как в clf.classes_);
    scores   — вероятности классов в том же порядке;
    labels   — сырые ключи support (обычно clf.classes_); по умолчанию JSON из classes;
    support  — {класс: число примеров}; None — ограничение поддержки не применяем
               (так ведут себя старые checkpoint без поля support);
    type_hint — тип объекта от детектора/организатора;
    policy   — "detector": при известном type_hint только классы этого типа,
               "joint": тип выбирается по лучшему поддержанному кортежу;
    min_support — класс с меньшим числом примеров сам подтвердить результат не может.
    """
    classes = [as_tuple(c) for c in classes]
    if policy not in ("joint", "detector"):
        raise ValueError("policy должен быть joint или detector")
    if labels is None:
        labels = [json.dumps(c, ensure_ascii=False) for c in classes]
    if len(labels) != len(classes) or len(scores) != len(classes):
        raise ValueError("classes, labels и scores должны быть одной длины")

    candidates = []
    for i, cls in enumerate(classes):
        if policy == "detector" and type_hint and cls[0] != type_hint:
            continue
        if support is not None and min_support and support.get(labels[i], 0) < min_support:
            continue
        candidates.append(i)

    eligible = list(candidates)
    if eligible and (policy == "joint" or not type_hint):
        # тип не задан: берём тип лучшего поддержанного кортежа и считаем только внутри него
        anchor = max(eligible, key=lambda i: float(scores[i]))
        kind = classes[anchor][0]
        eligible = [i for i in eligible if classes[i][0] == kind]

    p_bad = sum(float(scores[i]) for i in eligible if classes[i][1] == "bad")
    p_ok = sum(float(scores[i]) for i in eligible if classes[i][1] == "ok")
    statuses = {classes[i][1] for i in eligible}
    ordered = sorted(eligible, key=lambda i: float(scores[i]), reverse=True)
    return {
        "p_bad": p_bad, "p_ok": p_ok,
        "status_confidence": max(p_bad, p_ok),
        "margin": abs(p_bad - p_ok),
        "ready": bool(eligible) and set(STATUSES) <= statuses,
        "top_index": ordered[0] if ordered else None,
        "bad_index": next((i for i in ordered if classes[i][1] == "bad"), None),
        "ok_index": next((i for i in ordered if classes[i][1] == "ok"), None),
        "eligible": eligible,
    }


def decide(summary: dict, cfg: dict | None) -> str:
    """-> "ok" | "bad" | "review". cfg — секция варианта (пороги p_bad_min и т.д.)."""
    cfg = cfg or {}
    if not summary.get("ready"):
        return "review"
    if summary["status_confidence"] < float(cfg.get("status_conf_min", 0.65)):
        return "review"
    if summary["margin"] < float(cfg.get("margin_min", 0.10)):
        return "review"
    if summary["p_bad"] >= float(cfg.get("p_bad_min", 0.60)):
        return "bad"
    if summary["p_bad"] <= float(cfg.get("p_bad_max_ok", 0.40)):
        return "ok"
    return "review"


def pick(summary: dict, classes, decision: str) -> tuple[str | None, str | None]:
    """Тип и damage из лучшего кортежа нужного статуса (не обязательно из глобального топ-1)."""
    if decision not in STATUSES:
        return None, None
    classes = [as_tuple(c) for c in classes]
    index = summary.get("bad_index" if decision == "bad" else "ok_index")
    if index is None:
        return None, None
    return classes[index][0], classes[index][2]


def clean_state(o: dict, tax: dict) -> tuple[str, str | None]:
    """status/damage к допустимым значениям.

    Неизвестный/отсутствующий статус -> review, а не ok: молчаливое превращение
    «maybe» в «исправен» — самый опасный способ потерять дефект. explicit review
    всегда имеет damage=None; bad с видом вне таксономии сохраняет "unknown".
    """
    raw = str(o.get("status", "")).strip().lower()
    status = raw if raw in ANSWER_STATUSES else "review"
    damage = o.get("damage") if status == "bad" else None
    if status == "bad" and damage not in (tax or {}).get("damage", {}):
        damage = "unknown"
    return status, damage


def answer_error(answer: dict | None) -> str | None:
    """Почему ответ VLM нельзя использовать, или None.

    Допустимы ok/bad/review. Неизвестный статус — ошибка схемы: clean_state перевёл бы
    его в безопасный review, но каскад должен знать, что ответ битый, а не считать
    это осознанным решением VLM.
    """
    if not isinstance(answer, dict):
        return "ответ VLM не JSON-объект"
    if answer.get("is_object") is False:
        return None
    if str(answer.get("status", "")).strip().lower() not in ANSWER_STATUSES:
        return f"недопустимый status={answer.get('status')!r}"
    return None


def vlm_ready(vlm, cfg: dict | None = None) -> tuple[bool, str | None]:
    """Можно ли звать VLM -> (готовность, причина отказа).

    None — VLM не настроена (это не ошибка API, спорное просто уходит в review).
    Пустой key_env при заданном имени переменной — ключа нет, запрос не делаем.
    """
    if vlm is None:
        return False, None
    vcfg = cfg if cfg is not None else getattr(vlm, "cfg", {}) or {}
    key_env = vcfg.get("key_env", "")
    if key_env and not getattr(vlm, "key", ""):
        return False, f"нет ключа {key_env}"
    return True, None


def classify_crop_safe(vlm, crop, type_hint: str | None) -> dict:
    """VLM.classify_crop без исключений -> {answer, error, bad_json, cost}."""
    if vlm is None:
        return {"answer": None, "error": None, "bad_json": False, "cost": 0.0}
    try:
        r = vlm.classify_crop(crop, type_hint or "")
    except Exception as e:  # noqa: BLE001 — сбой сети/ключа не должен ронять разбор кадра
        return {"answer": None, "error": str(e)[:300], "bad_json": False, "cost": 0.0}
    return {"answer": r.get("answer"), "error": r.get("error"), "bad_json": bool(r.get("bad_json")),
            "cost": float(r.get("cost") or 0.0)}


def merge_vlm(base: dict, answer: dict | None, tax: dict, *, not_target: str = "review",
              authoritative: bool = True) -> dict | None:
    """Слить ответ VLM с результатом DINO.

    authoritative=True  — VLM разрешает спорный объект: её status/damage главные;
    authoritative=False — режим «только описание подтверждённого bad»: type/status/damage
                          остаются от DINO, меняется только description.
    is_object=False     — VLM считает, что в вырезе не наш объект: not_target="drop" выбрасывает
                          находку детектора, "review" оставляет её человеку (рамки организатора
                          молча удалять нельзя).
    Возвращает None, если объект нужно выбросить.
    """
    out = dict(base)
    if answer is None or not isinstance(answer, dict):
        return out
    description = str(answer.get("description") or "")
    if not authoritative:
        # description-only: is_object=false не удаляет подтверждённый DINO bad
        out["description"] = description or out.get("description", "")
        return out
    if answer.get("is_object") is False:
        if not_target == "drop":
            return None
        out.update(status="review", damage=None, description=description,
                   review_reasons=normalize_reasons(list(out.get("review_reasons") or []) + ["not_target"]))
        return out

    status, damage = clean_state(answer, tax)
    reasons = list(out.get("review_reasons") or [])
    if status == "review":
        reasons.append("low_confidence")
    out.update(status=status, damage=damage, description=description,
               review_reasons=normalize_reasons(reasons))
    typ = answer.get("type")
    if not out.get("type") and typ in (tax or {}).get("object_types", {}):
        out["type"] = typ
    if status in ("ok", "bad") and not out.get("type"):
        out.update(status="review", damage=None,
                   review_reasons=normalize_reasons(list(out.get("review_reasons") or []) + ["taxonomy_mismatch"]))
    return out
