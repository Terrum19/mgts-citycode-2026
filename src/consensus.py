"""Сверка мнений моделей, профили таксономии, оценка метрик и формирование отчётов для UI."""
from __future__ import annotations

import copy
from collections import Counter
from math import hypot
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from draw import annotate, font as draw_font
from evaluate import evaluate as evaluate_objects, iou
from vlm import ROOT, env

TYPE_ALIASES = {
    "ткш": "tksh",
    "телекоммуникационный климатический шкаф": "tksh",
    "tksh": "tksh",
    "рщ": "rsh",
    "распределительный щит": "rsh",
    "rsh": "rsh",
    "шкаф": "cabinet",
    "уличный шкаф": "cabinet",
    "cabinet": "cabinet",
    "колодец": "manhole",
    "люк": "manhole",
    "кабельный колодец": "manhole",
    "manhole": "manhole",
    "опора": "pole",
    "столб": "pole",
    "pole": "pole",
    "оборудование на опоре": "aerial_equipment",
    "aerial_equipment": "aerial_equipment",
}

DAMAGE_ALIASES = {
    "мп": "mechanical_damage",
    "механические повреждения": "mechanical_damage",
    "механическое повреждение": "mechanical_damage",
    "mechanical": "mechanical_damage",
    "mechanical_damage": "mechanical_damage",
    "лкп": "coating_corrosion",
    "неудовлетворительное состояние лкп": "coating_corrosion",
    "повреждение лкп / коррозия": "coating_corrosion",
    "paint": "coating_corrosion",
    "coating_corrosion": "coating_corrosion",
}

DSS_DEFECT_EQUIV = {
    "mechanical_damage": "МП",
    "mechanical": "МП",
    "cabinet_damaged": "МП",
    "cabinet_tilted": "МП",
    "equipment_open": "МП",
    "cables_exposed": "МП",
    "coating_corrosion": "ЛКП",
    "paint": "ЛКП",
}

DEFAULT_NAMES = {
    "tksh": "ТКШ (Телекоммуникационный климатический шкаф)",
    "rsh": "РЩ (Распределительный щит)",
    "cabinet": "Шкаф",
    "manhole": "Колодец",
    "pole": "Опора",
    "aerial_equipment": "Оборудование на опоре",
    "mechanical_damage": "Механическое повреждение (МП)",
    "mechanical": "Механическое повреждение (МП)",
    "coating_corrosion": "Повреждение ЛКП / коррозия",
    "paint": "Повреждение ЛКП / коррозия",
    "equipment_open": "Открыта/сорвана дверца",
    "cabinet_damaged": "Повреждён корпус шкафа",
    "cabinet_tilted": "Шкаф наклонён",
    "cables_exposed": "Оголённые кабели",
    "cover_broken": "Крышка сломана",
    "cover_missing": "Нет крышки",
    "cover_open_or_shifted": "Крышка сдвинута/открыта",
    "ring_or_surface_damaged": "Разрушено кольцо/покрытие",
    "pole_tilted": "Наклон опоры",
    "pole_damaged": "Повреждение опоры",
    "cables_damaged": "Провисшие/оборванные кабели",
    "cable_coil_unwound": "Размотан запас кабеля",
    "equipment_fallen": "Оборудование сорвано",
    "unknown": "Повреждение (вид не определён)",
}

TAXONOMY_PROFILES = {
    "cabinets": {
        "title": "Шкафы и щиты (ТКШ / РЩ)",
        "object_types": {
            "tksh": "телекоммуникационный климатический шкаф (ТКШ): обычно глубокий уличный корпус с вентиляционной панелью, жалюзи, радиатором или климатическим отсеком; на земле, цоколе или опоре",
            "rsh": "распределительный щит (РЩ): компактный металлический или пластиковый короб/шкаф с дверцей, часто на стене или опоре; внутри могут быть автоматы и клеммы",
        },
        "damage": {
            "mechanical_damage": "механические повреждения (МП): вмятины, смятая или порванная стенка, пролом, сорванная/сломанная/открытая дверца, нарушенное крепление, наклон корпуса, торчащие провода",
            "coating_corrosion": "неудовлетворительное состояние ЛКП: отслоение, шелушение или вздутие краски, ржавчина, коррозия, оголённый металл на корпусе или дверце",
        },
    },
    "infra": {
        "title": "Вся инфраструктура (config.toml)",
    },
    "combined": {
        "title": "Универсальный (ТКШ/РЩ + колодцы/опоры)",
        "object_types_extra": {
            "tksh": "телекоммуникационный климатический шкаф (ТКШ): уличный корпус с вентиляционной панелью, жалюзи или климатическим отсеком",
            "rsh": "распределительный щит (РЩ): компактный распределительный/электрический шкаф или короб с дверцей",
        },
        "damage_extra": {
            "mechanical_damage": "шкаф/щит: механические повреждения (МП) — вмятины, деформация, пролом, сломанная дверца или крепление",
            "coating_corrosion": "шкаф/щит: повреждение ЛКП / коррозия — отслоение краски, ржавчина, оголённый металл",
        },
    },
}

STATUS_RU = {"ok": "Исправен", "bad": "Повреждён", "review": "На проверке"}
CATEGORY_RU = {
    "status": "Расхождение по статусу (норма / поломка)",
    "type": "Расхождение по типу объекта / шкафа",
    "damage": "Расхождение по виду повреждения",
    "detection_count": "Разное число найденных объектов",
    "uncertain": "Низкая уверенность / статус review",
    "error": "Ошибка модели или некорректный ответ",
}


def build_display_names(cfg: dict) -> dict[str, str]:
    merged = dict(DEFAULT_NAMES)
    merged.update(cfg.get("names", {}))
    return merged


def apply_taxonomy_profile(cfg: dict, profile_key: str) -> dict:
    """Вернуть копию конфигурации с выбранным профилем таксономии."""
    out = copy.deepcopy(cfg)
    if profile_key == "cabinets":
        p = TAXONOMY_PROFILES["cabinets"]
        out["taxonomy"] = {
            "object_types": dict(p["object_types"]),
            "damage": dict(p["damage"]),
        }
    elif profile_key == "combined":
        p = TAXONOMY_PROFILES["combined"]
        base_types = dict(out["taxonomy"].get("object_types", {}))
        base_types.pop("cabinet", None)
        out["taxonomy"]["object_types"] = {**p["object_types_extra"], **base_types}
        out["taxonomy"]["damage"] = {**p["damage_extra"], **out["taxonomy"].get("damage", {})}
    return out


def variant_readiness(name: str, cfg: dict) -> tuple[bool, str]:
    """Проверить, готовы ли веса и ключи API для запуска варианта."""
    v = cfg["variants"][name]
    kind = v.get("kind", "vlm")
    if kind in ("det_vlm", "det_dino"):
        det_model = Path(cfg["detector"]["model"])
        if not det_model.is_absolute():
            det_model = ROOT / det_model
        if not det_model.is_dir():
            return False, f"Нет весов детектора: {cfg['detector']['model']}"
    if kind == "grounding_qwen":
        gd_model = ROOT / v["model"]
        if not gd_model.is_dir():
            return False, f"Нет весов Grounding DINO: {v['model']}"
    if kind in ("det_dino", "boxes_dino", "photo_dino"):
        ckpt = ROOT / v["checkpoint"]
        if not ckpt.is_file():
            return False, f"Нет чекпоинта: {v['checkpoint']}"
        enc = v.get("encoder")
        if enc and not (ROOT / enc).is_dir():
            return False, f"Нет энкодера: {enc}"
    vlm_name = v.get("vlm") or (name if kind == "vlm" else None)
    if vlm_name and vlm_name in cfg["variants"]:
        key_env = cfg["variants"][vlm_name].get("key_env", "")
        if key_env and not env(key_env):
            return False, f"Не задан {key_env} в .env / окружении"
    return True, "Готов к запуску"


def norm_type(value: str | None) -> str:
    raw = (value or "").strip()
    return TYPE_ALIASES.get(raw.casefold(), raw.casefold())


def norm_damage(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    return DAMAGE_ALIASES.get(raw.casefold(), raw.casefold())


def type_label(code: str | None, names: dict[str, str]) -> str:
    if not code:
        return "Не определён"
    return names.get(code) or names.get(norm_type(code)) or str(code)


def damage_label(code: str | None, names: dict[str, str]) -> str:
    if not code:
        return "—"
    return names.get(code) or names.get(norm_damage(code)) or str(code)


def status_label(status: str | None) -> str:
    return STATUS_RU.get(status or "", str(status or "—"))


def _center_dist_ratio(a: list[float], b: list[float]) -> float:
    dx = (a[0] + a[2] - b[0] - b[2]) / 2.0
    dy = (a[1] + a[3] - b[1] - b[3]) / 2.0
    diag = max(hypot(a[2] - a[0], a[3] - a[1]), hypot(b[2] - b[0], b[3] - b[1]), 1.0)
    return hypot(dx, dy) / diag


def _match_objects_across_models(objs_a: list[dict], objs_b: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Сопоставить объекты двух моделей по пространственному перекрытию (независимо от типа)."""
    if not objs_a or not objs_b:
        return [], list(range(len(objs_a))), list(range(len(objs_b)))

    # Если у какой-то модели объект без рамки (целое фото), при 1 объекте сопоставляем напрямую.
    has_no_box_a = any(o.get("bbox") is None for o in objs_a)
    has_no_box_b = any(o.get("bbox") is None for o in objs_b)
    if has_no_box_a or has_no_box_b:
        if len(objs_a) == 1 and len(objs_b) == 1:
            return [(0, 0)], [], []
        # Если одна модель дала один вердикт на кадр, а другая несколько рамок — связываем с первым/самым критичным
        if len(objs_a) == 1 and has_no_box_a:
            j_best = max(range(len(objs_b)), key=lambda j: (objs_b[j].get("status") == "bad", j == 0))
            unmatched_b = [j for j in range(len(objs_b)) if j != j_best]
            return [(0, j_best)], [], unmatched_b
        if len(objs_b) == 1 and has_no_box_b:
            i_best = max(range(len(objs_a)), key=lambda i: (objs_a[i].get("status") == "bad", i == 0))
            unmatched_a = [i for i in range(len(objs_a)) if i != i_best]
            return [(i_best, 0)], unmatched_a, []

    candidates = []
    for i, oa in enumerate(objs_a):
        ba = oa.get("bbox")
        if not ba:
            continue
        for j, ob in enumerate(objs_b):
            bb = ob.get("bbox")
            if not bb:
                continue
            ov = iou(ba, bb)
            dist = _center_dist_ratio(ba, bb)
            if ov >= 0.20 or (ov >= 0.05 and dist <= 0.35):
                same_t = 0.05 if norm_type(oa.get("type")) == norm_type(ob.get("type")) else 0.0
                score = ov - 0.1 * dist + same_t
                candidates.append((score, i, j))

    used_a, used_b, pairs = set(), set(), []
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))

    unmatched_a = [i for i in range(len(objs_a)) if i not in used_a]
    unmatched_b = [j for j in range(len(objs_b)) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


def _pred_summary(pred: dict | None, model_name: str, names: dict[str, str]) -> dict | None:
    if pred is None:
        return None
    objs = pred.get("objects") or []
    statuses = [o.get("status") for o in objs]
    if pred.get("error") or pred.get("bad_json"):
        frame_status = "error"
    elif "review" in statuses:
        frame_status = "review"
    elif "bad" in statuses:
        frame_status = "bad"
    else:
        frame_status = "ok"
    return {
        "name": model_name,
        "status": frame_status,
        "objects": objs,
        "types": [type_label(o.get("type"), names) for o in objs],
        "damages": [damage_label(o.get("damage"), names) for o in objs if o.get("status") == "bad"],
        "error": pred.get("error"),
        "bad_json": bool(pred.get("bad_json")),
        "seconds": round(float(pred.get("seconds") or 0.0), 2),
        "cost": float(pred.get("cost") or 0.0),
        "sub_opinions": pred.get("sub_opinions") or [],
    }


def compare_predictions(
    pred_a: dict,
    pred_b: dict | None,
    model_a_name: str,
    model_b_name: str | None,
    names: dict[str, str],
) -> dict[str, Any]:
    """Сравнить предсказания одной или двух моделей и определить итоговый статус фото."""
    objs_a = pred_a.get("objects") or []
    reasons: list[str] = []
    categories: list[str] = []

    def add_reason(cat: str, text: str) -> None:
        if cat not in categories:
            categories.append(cat)
        if text not in reasons:
            reasons.append(text)

    if pred_a.get("error"):
        add_reason("error", f"Ошибка модели {model_a_name}: {pred_a['error']}")
    elif pred_a.get("bad_json"):
        add_reason("error", f"Некорректный JSON от модели {model_a_name}")

    merged_objects: list[dict] = []

    if pred_b is None or not model_b_name:
        # Одиночный режим: проверяем review, внутренние суб-мнения (photo_dino) и crop_results
        sub_ops = pred_a.get("sub_opinions") or []
        if len(sub_ops) >= 2:
            sub_labels = {op.get("label") for op in sub_ops if op.get("label")}
            if len(sub_labels) > 1:
                parts = [f"{op['model']} → «{op['label']}»" for op in sub_ops if op.get("label")]
                add_reason("status", "Разошлись внутренние модели: " + "; ".join(parts))

        for crop in pred_a.get("crop_results") or []:
            if crop.get("error"):
                add_reason("error", f"Ошибка VLM на вырезе #{crop.get('proposal_id')}: {crop['error']}")
            elif crop.get("bad_json"):
                add_reason("error", f"Некорректный ответ VLM на вырезе #{crop.get('proposal_id')}")

        for idx, oa in enumerate(objs_a, 1):
            st = oa.get("status") or "ok"
            obj_diffs: list[str] = []
            if st == "review":
                msg = oa.get("description") or f"Объект #{idx} ({type_label(oa.get('type'), names)}): требуется ручная проверка"
                add_reason("uncertain", msg)
                obj_diffs.append(msg)
                if oa.get("top_classes"):
                    top_str = ", ".join(f"{c} ({p:.0%})" for c, p in oa["top_classes"][:2])
                    obj_diffs.append(f"Конкурирующие классы: {top_str}")
            elif reasons and st != "bad":
                st = "review"
            merged_objects.append({
                "type": oa.get("type") or "cabinet",
                "bbox": oa.get("bbox"),
                "status": st,
                "damage": oa.get("damage") if st == "bad" else None,
                "candidate_damage": oa.get("damage"),
                "description": oa.get("description") or "",
                "model_a_obj": oa,
                "model_b_obj": None,
                "agreed": not obj_diffs and "status" not in categories,
                "disagreements": obj_diffs,
                "source": model_a_name,
            })
    else:
        objs_b = pred_b.get("objects") or []
        if pred_b.get("error"):
            add_reason("error", f"Ошибка модели {model_b_name}: {pred_b['error']}")
        elif pred_b.get("bad_json"):
            add_reason("error", f"Некорректный JSON от модели {model_b_name}")

        pairs, unmatched_a, unmatched_b = _match_objects_across_models(objs_a, objs_b)

        for i, j in pairs:
            oa, ob = objs_a[i], objs_b[j]
            sa = oa.get("status") or "ok"
            sb = ob.get("status") or "ok"
            ta, tb = norm_type(oa.get("type")), norm_type(ob.get("type"))
            da, db = norm_damage(oa.get("damage")), norm_damage(ob.get("damage"))
            pair_diffs: list[str] = []

            if "review" in (sa, sb):
                msg = f"Неуверенный ответ одной из моделей ({model_a_name}: {status_label(sa)}, {model_b_name}: {status_label(sb)})"
                add_reason("uncertain", msg)
                pair_diffs.append(msg)
            elif sa != sb:
                msg = f"Статус: {model_a_name} → «{status_label(sa)}», {model_b_name} → «{status_label(sb)}»"
                add_reason("status", msg)
                pair_diffs.append(msg)

            if ta != tb:
                msg = f"Тип шкафа/объекта: {model_a_name} → «{type_label(oa.get('type'), names)}», {model_b_name} → «{type_label(ob.get('type'), names)}»"
                add_reason("type", msg)
                pair_diffs.append(msg)

            if sa == "bad" and sb == "bad" and (da or "unknown") != (db or "unknown"):
                msg = f"Вид повреждения: {model_a_name} → «{damage_label(oa.get('damage'), names)}», {model_b_name} → «{damage_label(ob.get('damage'), names)}»"
                add_reason("damage", msg)
                pair_diffs.append(msg)

            agreed_pair = not pair_diffs
            merged_status = sa if agreed_pair else "review"
            merged_damage = oa.get("damage") if agreed_pair and sa == "bad" else None
            cand_damage = oa.get("damage") or ob.get("damage")
            desc_a = (oa.get("description") or "").strip()
            desc_b = (ob.get("description") or "").strip()
            combined_desc = desc_a or desc_b
            if not agreed_pair and desc_a and desc_b and desc_a != desc_b:
                combined_desc = f"{model_a_name}: {desc_a} | {model_b_name}: {desc_b}"

            merged_objects.append({
                "type": oa.get("type") or ob.get("type") or "cabinet",
                "alt_type": ob.get("type") if ta != tb else None,
                "bbox": oa.get("bbox") or ob.get("bbox"),
                "bbox_a": oa.get("bbox"),
                "bbox_b": ob.get("bbox"),
                "status": merged_status,
                "damage": merged_damage,
                "candidate_damage": cand_damage,
                "description": combined_desc,
                "model_a_obj": oa,
                "model_b_obj": ob,
                "agreed": agreed_pair,
                "disagreements": pair_diffs,
                "source": f"{model_a_name} + {model_b_name}",
            })

        for i in unmatched_a:
            oa = objs_a[i]
            msg = (
                f"Только {model_a_name} обнаружила объект: "
                f"«{type_label(oa.get('type'), names)}» ({status_label(oa.get('status'))})"
            )
            add_reason("detection_count", msg)
            merged_objects.append({
                "type": oa.get("type") or "cabinet",
                "bbox": oa.get("bbox"),
                "bbox_a": oa.get("bbox"),
                "bbox_b": None,
                "status": "review",
                "damage": None,
                "candidate_damage": oa.get("damage"),
                "description": oa.get("description") or "",
                "model_a_obj": oa,
                "model_b_obj": None,
                "agreed": False,
                "disagreements": [msg],
                "source": model_a_name,
            })

        for j in unmatched_b:
            ob = objs_b[j]
            msg = (
                f"Только {model_b_name} обнаружила объект: "
                f"«{type_label(ob.get('type'), names)}» ({status_label(ob.get('status'))})"
            )
            add_reason("detection_count", msg)
            merged_objects.append({
                "type": ob.get("type") or "cabinet",
                "bbox": ob.get("bbox"),
                "bbox_a": None,
                "bbox_b": ob.get("bbox"),
                "status": "review",
                "damage": None,
                "candidate_damage": ob.get("damage"),
                "description": ob.get("description") or "",
                "model_a_obj": None,
                "model_b_obj": ob,
                "agreed": False,
                "disagreements": [msg],
                "source": model_b_name,
            })

    if reasons or any(o["status"] == "review" for o in merged_objects):
        outcome = "review"
    elif any(o["status"] == "bad" for o in merged_objects):
        outcome = "bad"
    else:
        outcome = "ok"

    agreed = outcome != "review"
    seconds = float(pred_a.get("seconds") or 0.0) + float((pred_b or {}).get("seconds") or 0.0)
    cost = float(pred_a.get("cost") or 0.0) + float((pred_b or {}).get("cost") or 0.0)

    return {
        "outcome": outcome,
        "agreed": agreed,
        "disagreement_reasons": reasons,
        "disagreement_categories": categories,
        "objects": merged_objects,
        "model_a": _pred_summary(pred_a, model_a_name, names),
        "model_b": _pred_summary(pred_b, model_b_name, names) if (pred_b is not None and model_b_name) else None,
        "seconds": round(seconds, 2),
        "cost": cost,
    }


def annotate_verdict(
    img: Image.Image,
    verdict: dict,
    names: dict[str, str],
    model_a_name: str,
    model_b_name: str | None = None,
) -> Image.Image:
    """Нарисовать рамки для результата: при расхождении показываем позиции обеих моделей."""
    if verdict["outcome"] != "review" or not verdict.get("model_b"):
        return annotate(img, verdict["objects"], names=names)

    base = img.convert("RGB")
    d = ImageDraw.Draw(base)
    lw = max(2, round(max(base.size) / 400))
    f = draw_font(max(12, round(max(base.size) / 70)))

    color_a = (255, 140, 0)   # оранжевый — Модель 1
    color_b = (34, 150, 243)  # синий — Модель 2

    def draw_dashed_rect(box, color):
        x1, y1, x2, y2 = box
        dash = max(6, lw * 3)
        for a, b, horiz, c in ((x1, x2, True, y1), (x1, x2, True, y2), (y1, y2, False, x1), (y1, y2, False, x2)):
            for s in range(int(a), int(b), dash * 2):
                e = min(s + dash, int(b))
                d.line((s, c, e, c) if horiz else (c, s, c, e), fill=color, width=lw)

    objs_a = (verdict.get("model_a") or {}).get("objects") or []
    objs_b = (verdict.get("model_b") or {}).get("objects") or []
    for k, oa in enumerate(objs_a, 1):
        box = oa.get("bbox")
        tag = f"M1.{k} {type_label(oa.get('type'), names)} ({status_label(oa.get('status'))})"
        if box is not None:
            d.rectangle(box, outline=color_a, width=lw)
            x, y = box[0], max(0, box[1] - f.size - 4)
        else:
            x, y = 6, 6 + (k - 1) * (f.size + 6)
        d.text((x, y), tag, fill=color_a, font=f, stroke_width=2, stroke_fill="black")

    for k, ob in enumerate(objs_b, 1):
        box = ob.get("bbox")
        tag = f"M2.{k} {type_label(ob.get('type'), names)} ({status_label(ob.get('status'))})"
        if box is not None:
            draw_dashed_rect(box, color_b)
            x, y = box[0], min(base.height - f.size - 4, box[3] + 2)
        else:
            x, y = 6, 6 + (len(objs_a) + k - 1) * (f.size + 6)
        d.text((x, y), tag, fill=color_b, font=f, stroke_width=2, stroke_fill="black")

    return base


def _frame_pred_summary(verdict: dict, use_model: str = "consensus") -> tuple[str, str | None, str | None]:
    """Свести вердикт по кадру к (status, dominant_type, dominant_damage)."""
    if use_model == "model_a" and verdict.get("model_a"):
        objs = verdict["model_a"]["objects"]
        st = verdict["model_a"]["status"]
    elif use_model == "model_b" and verdict.get("model_b"):
        objs = verdict["model_b"]["objects"]
        st = verdict["model_b"]["status"]
    else:
        objs = verdict["objects"]
        st = verdict["outcome"]

    bad_objs = [o for o in objs if o.get("status") == "bad"]
    chosen = bad_objs[0] if bad_objs else (objs[0] if objs else None)
    typ = norm_type(chosen.get("type")) if chosen else None
    dmg = norm_damage(chosen.get("damage") or chosen.get("candidate_damage")) if chosen else None
    return st, typ, dmg


def _compute_classification_metrics(
    items: list[dict],
    verdicts_by_key: dict[str, dict],
    mode: str = "consensus",
) -> dict[str, Any]:
    """Метрики против классовой разметки датасета (папки или таблица)."""
    labeled_items = [it for it in items if it.get("gt_status") in ("ok", "bad")]
    if not labeled_items:
        return {}

    tp = fp = fn = tn = review_on_bad = review_on_ok = 0
    type_total = type_correct = 0
    defect_total = defect_correct = 0

    for it in labeled_items:
        v = verdicts_by_key[it["key"]]
        gt_st = it["gt_status"]
        pr_st, pr_type, pr_dmg = _frame_pred_summary(v, mode)
        if pr_st == "error":
            pr_st = "review"

        if gt_st == "bad":
            if pr_st == "bad":
                tp += 1
            elif pr_st == "review":
                review_on_bad += 1
            else:
                fn += 1
        else:
            if pr_st == "ok":
                tn += 1
            elif pr_st == "review":
                review_on_ok += 1
            else:
                fp += 1

        gt_t = norm_type(it.get("gt_type"))
        if gt_t and pr_type:
            type_total += 1
            if gt_t == pr_type or (gt_t in ("tksh", "rsh") and pr_type == "cabinet"):
                # Точное совпадение ТКШ==ТКШ / РЩ==РЩ; общее "cabinet" не засчитываем как точный ТКШ/РЩ
                if gt_t == pr_type:
                    type_correct += 1

        gt_d_raw = it.get("gt_defect") or ""
        gt_d_dss = DSS_DEFECT_EQUIV.get(norm_damage(gt_d_raw))
        if gt_st == "bad" and gt_d_dss and pr_st == "bad" and pr_dmg:
            defect_total += 1
            if DSS_DEFECT_EQUIV.get(pr_dmg) == gt_d_dss:
                defect_correct += 1

    total = len(labeled_items)
    # Строгий зачёт (только подтверждённые bad) и полный зачёт (bad + отправленные на review не пропущены)
    flagged_bad = tp + review_on_bad
    flagged_fp = fp + review_on_ok
    strict_recall = tp / (tp + fn + review_on_bad) if (tp + fn + review_on_bad) else 0.0
    flagged_recall = flagged_bad / (tp + fn + review_on_bad) if (tp + fn + review_on_bad) else 0.0
    strict_prec = tp / (tp + fp) if (tp + fp) else 0.0
    acc_auto = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1_bad = (2 * strict_prec * strict_recall / (strict_prec + strict_recall)) if (strict_prec + strict_recall) else 0.0

    return {
        "labeled_photos": total,
        "auto_decided": tp + tn + fp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "review_on_bad": review_on_bad,
        "review_on_ok": review_on_ok,
        "accuracy_auto": acc_auto,
        "defect_recall_strict": strict_recall,
        "defect_recall_with_review": flagged_recall,
        "defect_precision": strict_prec,
        "f1_bad": f1_bad,
        "type_accuracy": (type_correct / type_total) if type_total else None,
        "defect_type_accuracy": (defect_correct / defect_total) if defect_total else None,
        "flagged_fp": flagged_fp,
    }


def summarize_run(
    verdicts_by_key: dict[str, dict],
    items: list[dict],
    gt_kind: str | None,
    names: dict[str, str],
    model_a_name: str,
    model_b_name: str | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Посчитать сводные метрики по датасету и метрики качества (если есть разметка)."""
    overrides = overrides or {}
    outcome_counts = Counter()
    type_counts = Counter()
    damage_counts = Counter()
    reason_counts = Counter()
    agreed_count = 0
    error_photos = 0
    total_seconds = 0.0
    total_cost = 0.0
    total_objects = 0
    bad_objects = 0
    ok_objects = 0
    review_objects = 0

    for it in items:
        key = it["key"]
        v = verdicts_by_key[key]
        eff_outcome = overrides.get(key, v["outcome"])
        outcome_counts[eff_outcome] += 1
        if v["agreed"]:
            agreed_count += 1
        if "error" in v["disagreement_categories"]:
            error_photos += 1
        total_seconds += v["seconds"]
        total_cost += v["cost"]
        for cat in v["disagreement_categories"]:
            reason_counts[CATEGORY_RU.get(cat, cat)] += 1
        for obj in v["objects"]:
            total_objects += 1
            type_counts[type_label(obj.get("type"), names)] += 1
            st = obj.get("status")
            if st == "bad":
                bad_objects += 1
                damage_counts[damage_label(obj.get("damage"), names)] += 1
            elif st == "review":
                review_objects += 1
            else:
                ok_objects += 1

    n = len(items)
    summary = {
        "total_photos": n,
        "confirmed_bad_photos": outcome_counts["bad"],
        "confirmed_ok_photos": outcome_counts["ok"],
        "review_photos": outcome_counts["review"],
        "error_photos": error_photos,
        "total_objects": total_objects,
        "bad_objects": bad_objects,
        "ok_objects": ok_objects,
        "review_objects": review_objects,
        "agreed_photos": agreed_count,
        "agreement_rate": (agreed_count / n) if (n and model_b_name) else None,
        "type_counts": dict(type_counts.most_common()),
        "damage_counts": dict(damage_counts.most_common()),
        "disagreement_reason_counts": dict(reason_counts.most_common()),
        "total_seconds": round(total_seconds, 2),
        "avg_seconds": round(total_seconds / n, 2) if n else 0.0,
        "total_cost": round(total_cost, 5),
        "gt_metrics": None,
        "gt_object_metrics": None,
    }

    if gt_kind == "objects":
        labels = [{"file": it["key"], "objects": it["gt_objects"] or []} for it in items if it.get("gt_objects") is not None]
        if labels:
            preds_eval = {
                k: {"objects": [o for o in v["objects"] if o.get("bbox") is not None],
                    "error": (v.get("model_a") or {}).get("error"),
                    "bad_json": (v.get("model_a") or {}).get("bad_json", False),
                    "seconds": v["seconds"], "cost": v["cost"]}
                for k, v in verdicts_by_key.items()
            }
            summary["gt_object_metrics"] = evaluate_objects(labels, preds_eval)
    elif gt_kind in ("folders", "table"):
        gt_m = {"consensus": _compute_classification_metrics(items, verdicts_by_key, "consensus")}
        gt_m["model_a"] = _compute_classification_metrics(items, verdicts_by_key, "model_a")
        if model_b_name:
            gt_m["model_b"] = _compute_classification_metrics(items, verdicts_by_key, "model_b")
        summary["gt_metrics"] = gt_m

    return summary


def build_export_tables(
    items: list[dict],
    verdicts_by_key: dict[str, dict],
    names: dict[str, str],
    overrides: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Сформировать таблицы objects, repair, review, frames для CSV/Excel."""
    overrides = overrides or {}
    objects_rows: list[dict] = []
    frames_rows: list[dict] = []
    repair_rows: list[dict] = []
    review_rows: list[dict] = []

    for it in items:
        key = it["key"]
        v = verdicts_by_key[key]
        eff_outcome = overrides.get(key, v["outcome"])
        meta = it.get("meta") or {}
        ma = v.get("model_a") or {}
        mb = v.get("model_b")

        frame_row = {
            "file": key,
            "outcome": eff_outcome,
            "outcome_ru": {"bad": "Поломка подтверждена", "ok": "Исправен", "review": "На доп. подтверждение"}.get(eff_outcome, eff_outcome),
            "operator_override": overrides.get(key, ""),
            "agreed": v["agreed"],
            "objects_count": len(v["objects"]),
            "disagreement_reasons": "; ".join(v["disagreement_reasons"]),
            "model_a": ma.get("name", ""),
            "model_a_status": ma.get("status", ""),
            "model_b": mb.get("name", "") if mb else "",
            "model_b_status": mb.get("status", "") if mb else "",
            "gt_label": it.get("gt_label") or "",
            "seconds": v["seconds"],
            "cost": v["cost"],
            **meta,
        }
        frames_rows.append(frame_row)

        if not v["objects"]:
            if eff_outcome == "review":
                review_rows.append({
                    "file": key, "object_n": 0, "type_ru": "—", "status": "review",
                    "candidate_damage_ru": "", "description": "",
                    "disagreement_reasons": "; ".join(v["disagreement_reasons"]),
                    "gt_label": it.get("gt_label") or "", **meta,
                })
            continue

        for idx, obj in enumerate(v["objects"], 1):
            eff_obj_status = eff_outcome if overrides.get(key) in ("ok", "bad") else obj.get("status")
            row = {
                "file": key,
                "object_n": idx,
                "type": obj.get("type", ""),
                "type_ru": type_label(obj.get("type"), names),
                "status": eff_obj_status,
                "status_ru": status_label(eff_obj_status),
                "damage": obj.get("damage") or "",
                "damage_ru": damage_label(obj.get("damage"), names) if eff_obj_status == "bad" else "",
                "candidate_damage_ru": damage_label(obj.get("candidate_damage"), names) if obj.get("candidate_damage") else "",
                "description": obj.get("description") or "",
                "bbox": " ".join(map(str, obj["bbox"])) if obj.get("bbox") else "",
                "disagreement_reasons": "; ".join(obj.get("disagreements") or v["disagreement_reasons"]),
                "gt_label": it.get("gt_label") or "",
                **meta,
            }
            objects_rows.append(row)
            if eff_obj_status == "bad":
                repair_rows.append(row)
            elif eff_obj_status == "review":
                review_rows.append(row)

    return {
        "objects": objects_rows,
        "frames": frames_rows,
        "repair": repair_rows,
        "review": review_rows,
    }
