"""Функциональный веб-интерфейс диагностики инфраструктуры и шкафов МГТС.

    streamlit run src/app.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from tempfile import mkdtemp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants  # noqa: E402
from consensus import (  # noqa: E402
    CATEGORY_RU,
    PRIORITY_RU,
    REASONS_RU,
    TAXONOMY_PROFILES,
    annotate_verdict,
    apply_taxonomy_profile,
    build_display_names,
    build_export_tables,
    compare_predictions,
    damage_label,
    priority_rank,
    status_label,
    summarize_run,
    type_label,
    variant_readiness,
)
from dataset_io import (  # noqa: E402
    DatasetBundle,
    ingest_local_directory,
    ingest_uploaded_directory,
    ingest_uploaded_photos,
    ingest_zip_archive,
    inspect_dataset,
)
from asset import enrich_result, load_registry  # noqa: E402
from predict import IMG_EXT, read_meta, write_csv  # noqa: E402
from run import load_config  # noqa: E402
from vlm import ROOT  # noqa: E402

LAT_COLS = ("lat", "latitude", "широта", "pano_lat")
LON_COLS = ("lon", "lng", "longitude", "долгота", "pano_lon")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
        .badge-bad {
            display: inline-block; padding: 2px 8px; border-radius: 6px;
            background: rgba(230, 40, 40, 0.14); color: #d32f2f; font-weight: 600; font-size: 0.85rem;
        }
        .badge-ok {
            display: inline-block; padding: 2px 8px; border-radius: 6px;
            background: rgba(40, 200, 70, 0.14); color: #2e7d32; font-weight: 600; font-size: 0.85rem;
        }
        .badge-review {
            display: inline-block; padding: 2px 8px; border-radius: 6px;
            background: rgba(255, 160, 0, 0.18); color: #b26a00; font-weight: 600; font-size: 0.85rem;
        }
        .model-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 8px; padding: 0.75rem 0.9rem;
            background: rgba(128, 128, 128, 0.04);
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _model_for_session(slot: str, name: str, config: dict):
    """Кэшировать тяжёлые локальные модели в session_state с учётом профиля таксономии."""
    cache_key = f"cached_model_{slot}"
    tax_sig = json.dumps(config.get("taxonomy", {}), sort_keys=True, ensure_ascii=False)
    v = config["variants"][name]
    identity = (name, tax_sig, v.get("kind"), v.get("model"), v.get("vlm"), v.get("checkpoint"))
    cached = st.session_state.get(cache_key)
    if cached is not None and cached[0] == identity:
        return cached[1]
    model = variants.make(name, config)
    st.session_state[cache_key] = (identity, model)
    return model


def _pick_default_variants(cfg: dict) -> tuple[str, str]:
    ready = [k for k in cfg["variants"] if k != FRAME_CROP and variant_readiness(k, cfg)[0]]
    all_names = [name for name in cfg["variants"] if name != FRAME_CROP]
    m1 = "qwen235" if "qwen235" in ready else (ready[0] if ready else all_names[0])
    m2_candidates = [k for k in ("gemma31", "det_qwen235", "gemma31_free", "nemotron_free", "det_dino", "photo_dino")
                     if k in ready and k != m1]
    if not m2_candidates:
        m2_candidates = [k for k in ready if k != m1]
    if not m2_candidates:
        m2_candidates = [k for k in all_names if k != m1]
    m2 = m2_candidates[0] if m2_candidates else m1
    return m1, m2


def _predict_item(item: dict, model_a, model_b, boxes_map: dict | None, kind_a: str, kind_b: str | None):
    path = item["path"]
    key = item["key"]
    with Image.open(path) as im:
        rgb = im if im.mode == "RGB" else im.convert("RGB")
        rgb.load()

    def call_model(model, kind):
        if kind == "boxes_dino":
            if boxes_map is None or (key not in boxes_map and path.name not in boxes_map):
                raise ValueError(f"Нет рамок для {key}")
            supplied = boxes_map.get(key) or boxes_map[path.name]
            return model.predict(rgb, boxes=supplied)
        return model.predict(rgb)

    if model_b is None:
        pred_a = call_model(model_a, kind_a)
        pred_b = None
    else:
        with ThreadPoolExecutor(max_workers=2) as pair_ex:
            fa = pair_ex.submit(call_model, model_a, kind_a)
            fb = pair_ex.submit(call_model, model_b, kind_b)
            pred_a = fa.result()
            pred_b = fb.result()
    return key, rgb, pred_a, pred_b


FRAME_CROP = "frame_crop_qwen235"
FRAME_NAME = "Qwen235 · целое фото"
CROP_NAME = "Qwen235 · основной вырез"


def _frame_crop_verdict(prediction, names):
    decision = compare_predictions(
        {**prediction, "sub_opinions": [], "crop_results": []}, None, FRAME_CROP, None, names
    )
    outcome = (prediction.get("objects") or [{}])[0].get("status") or "review"
    decision["outcome"] = outcome
    decision["agreed"] = outcome != "review"
    reason = prediction.get("review_reason")
    if outcome == "review" and reason:
        decision["disagreement_reasons"] = [reason]
    decision["checks"] = {}
    for prefix, slot, name in (("frame", "model_a", FRAME_NAME), ("crop", "model_b", CROP_NAME)):
        label = prediction.get(prefix + "_class") or "review"
        description = prediction.get(prefix + "_description") or ""
        kind, _, state = label.partition("_")
        kind = {"rs": "rsh", "tk": "tksh"}.get(kind)
        damage = {"mech": "mechanical_damage", "paint": "coating_corrosion"}.get(state)
        status = "ok" if state == "ok" else "bad" if damage else "review"
        bbox = decision["objects"][0].get("bbox") if prefix == "crop" and decision["objects"] else None
        objects = [{"type": kind, "status": status, "damage": damage, "description": description, "bbox": bbox}]
        decision[slot] = {"name": name, "status": status, "objects": objects}
        decision["checks"][prefix + "_class"] = label
        decision["checks"][prefix + "_description"] = description
    frame_class = prediction.get("frame_class")
    crop_class = prediction.get("crop_class")
    if outcome == "review" and frame_class and crop_class and "review" not in (frame_class, crop_class):
        frame_type, frame_state = frame_class.split("_")
        crop_type, crop_state = crop_class.split("_")
        categories = []
        if frame_type != crop_type:
            categories.append("type")
        if frame_state != crop_state:
            categories.append("status" if "ok" in (frame_state, crop_state) else "damage")
        if categories:
            decision["disagreement_categories"] = categories
    return decision


def _export_results(items, verdicts, names, overrides=None):
    tables = build_export_tables(items, verdicts, names, overrides=overrides)
    for sheet in ("frames", "objects", "repair", "review"):
        for row in tables[sheet]:
            row.update(verdicts[row["file"]].get("checks", {}))
    tables["passport"] = []
    for filename, verdict in verdicts.items():
        asset = verdict.get("asset") or {}
        assessment = verdict.get("assessment") or {}
        tables["passport"].append({
            "file": filename, "asset_id": asset.get("asset_id") or "", "identity": asset.get("identity_status", ""),
            "registry": asset.get("registry_status", ""), "address": asset.get("address") or "",
            "lat": asset.get("lat"), "lon": asset.get("lon"), "effect": assessment.get("effect", ""),
            "recommendation": assessment.get("recommendation", ""), "urgency": assessment.get("urgency", ""),
            "scope": verdict.get("scope", ""),
        })
    return tables


def _to_excel_bytes(tables: dict[str, list[dict]]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, rows in tables.items():
            df = pd.DataFrame(rows) if rows else pd.DataFrame([{"info": "Пусто"}])
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return buf.getvalue()


def _render_object_cards(verdict: dict, names: dict[str, str], show_agreement_note: bool = False) -> None:
    objs = verdict.get("objects") or []
    if not objs:
        st.info("Целевые объекты (шкафы / инфраструктура) на фотографии не обнаружены.")
        return

    for idx, obj in enumerate(objs, 1):
        t_ru = type_label(obj.get("type"), names)
        alt_t = obj.get("alt_type")
        if alt_t:
            t_display = f"{t_ru} / {type_label(alt_t, names)}"
        else:
            t_display = t_ru

        st_code = obj.get("status")
        if st_code == "bad":
            dmg_ru = damage_label(obj.get("damage"), names)
            pr = PRIORITY_RU.get(obj.get("priority", "none"), "")
            pr_note = f" · {pr}" if pr and pr != "—" else ""
            st.markdown(
                f"**{idx}. {t_display}** &nbsp; <span class='badge-bad'>Повреждён · {dmg_ru}{pr_note}</span>",
                unsafe_allow_html=True,
            )
        elif st_code == "review":
            cand = obj.get("candidate_damage")
            cand_note = f" (подозрение: {damage_label(cand, names)})" if cand else ""
            st.markdown(
                f"**{idx}. {t_display}** &nbsp; <span class='badge-review'>Требует подтверждения{cand_note}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"**{idx}. {t_display}** &nbsp; <span class='badge-ok'>Исправен</span>",
                unsafe_allow_html=True,
            )

        desc = (obj.get("description") or "").strip()
        if desc:
            st.write(f"Описание: {desc}")
        elif st_code == "bad":
            st.caption("Описание дефекта не предоставлено моделью.")

        if show_agreement_note and obj.get("agreed") and verdict.get("model_b"):
            st.caption("✓ Подтверждено обеими моделями")


def _render_model_opinion_block(summary: dict | None, title: str, names: dict[str, str], accent: str) -> None:
    if not summary:
        return
    st.markdown(f"**{accent} {title}: `{summary['name']}`**")
    if summary.get("error"):
        st.error(f"Ошибка модели: {summary['error']}")
        return
    if summary.get("bad_json"):
        st.warning("Модель вернула некорректный JSON.")
        return

    objs = summary.get("objects") or []
    elapsed = f" · {summary['seconds']:.1f} с" if 'seconds' in summary else ""
    st.caption(f"Итог: **{status_label(summary['status'])}**{elapsed}")
    if not objs:
        st.write("Объекты не найдены.")
    else:
        for k, o in enumerate(objs, 1):
            t_ru = type_label(o.get("type"), names)
            st_ru = status_label(o.get("status"))
            dmg_txt = f" — {damage_label(o.get('damage'), names)}" if o.get("status") == "bad" else ""
            desc = (o.get("description") or "").strip()
            line = f"{k}. **{t_ru}**: {st_ru}{dmg_txt}"
            if o.get("det_score") is not None:
                line += f" (score={o['det_score']:.2f})"
            st.markdown(line)
            if desc:
                st.caption(desc)
            if o.get("top_classes"):
                probs = ", ".join(f"{c}: {p:.0%}" for c, p in o["top_classes"][:3])
                st.caption(f"Вероятности: {probs}")

    sub_ops = summary.get("sub_opinions") or []
    if len(sub_ops) >= 2:
        st.caption("Внутренние модели:")
        for op in sub_ops:
            conf = f" ({op['confidence']:.0%})" if op.get("confidence") is not None else ""
            st.caption(f"• {op['model']}: {op.get('label')}{conf}")


def _render_disagreement_section(verdict: dict, names: dict[str, str]) -> None:
    st.markdown("#### Почему фото передано на проверку" if verdict.get("checks") else "#### Как разошелось мнение моделей")
    reasons = verdict.get("disagreement_reasons") or []
    if reasons:
        for r in reasons:
            st.warning(r, icon="⚠️")
    else:
        st.warning("Модель отметила низкую уверенность (review).", icon="⚠️")

    ma = verdict.get("model_a")
    mb = verdict.get("model_b")
    if ma and mb:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                _render_model_opinion_block(ma, "Модель 1", names, "🟠")
        with c2:
            with st.container(border=True):
                _render_model_opinion_block(mb, "Модель 2", names, "🔵")
    elif ma:
        with st.container(border=True):
            _render_model_opinion_block(ma, "Детализация модели", names, "🟠")


def _render_item_meta(item: dict, verdict: dict) -> None:
    meta = item.get("meta") or {}
    parts = []
    if item.get("gt_label"):
        parts.append(f"Эталон: `{item['gt_label']}`")
    for k in ("ID объекта", "asset_id", "object_id", "Адрес", "address"):
        if meta.get(k):
            parts.append(f"{k}: `{meta[k]}`")
    if meta.get("lat") not in (None, "") and meta.get("lon") not in (None, ""):
        parts.append(f"Координаты снимка: `{meta['lat']}, {meta['lon']}`")
    asset = verdict.get("asset") or {}
    if asset.get("registry_status") == "matched":
        parts.append(f"Реестр: {asset['asset_id']}")
        if asset.get("address"):
            parts.append(str(asset["address"]))
        if asset.get("lat") is not None and asset.get("lon") is not None:
            parts.append(f"Координаты объекта: `{asset['lat']}, {asset['lon']}`")
    if parts:
        st.caption(" · ".join(parts))
    recommendation = (verdict.get("assessment") or {}).get("recommendation")
    if recommendation:
        st.write(f"Рекомендация по автоматической оценке: {recommendation}")
    if item.get("meta_description"):
        st.caption(f"Описание из таблицы дата-сета: {item['meta_description']}")


def _effective_outcome(key: str, verdict: dict, overrides: dict[str, str]) -> str:
    return overrides.get(key, verdict["outcome"])


def main() -> None:
    st.set_page_config(
        page_title="МГТС: Диагностика инфраструктуры и шкафов",
        page_icon="🛠️",
        layout="wide",
    )
    _inject_styles()

    cfg_base = load_config()
    variant_names = list(cfg_base["variants"])

    # ==================== ЛЕВЫЙ САЙДБАР ====================
    with st.sidebar:
        st.header("📁 Загрузка дата-сета")
        upload_mode = st.radio(
            "Источник данных",
            ["Папка", "Фотографии", "ZIP-архив"],
            horizontal=True,
        )

        uploaded_photos = None
        uploaded_dir_files = None
        local_dir_path = ""
        uploaded_zip = None

        if upload_mode == "Фотографии":
            uploaded_photos = st.file_uploader(
                "Выберите фотографии",
                type=[e[1:] for e in IMG_EXT],
                accept_multiple_files=True,
                help="Можно выбрать несколько файлов .jpg, .png, .webp и др.",
            )
        elif upload_mode == "Папка":
            uploaded_dir_files = st.file_uploader(
                "Выберите папку с дата-сетом",
                type=None,
                accept_multiple_files="directory",
                help="Поддерживаются вложенные папки классов (например ТКШ/МП, МП.РЩ) и таблицы описаний (.xlsx/.csv).",
            )
            with st.expander("Или указать путь к локальной папке", expanded=False):
                local_dir_path = st.text_input(
                    "Путь на диске",
                    placeholder="Например: data/dss_test",
                )
        else:
            uploaded_zip = st.file_uploader(
                "Загрузите ZIP-архив",
                type=["zip"],
                help="ZIP с фотографиями, подпапками классов, таблицами метаданных или labels.json.",
            )

        with st.expander("Доп. таблицы и рамки (необязательно)", expanded=False):
            extra_meta_file = st.file_uploader(
                "Таблица ID / координат (CSV, XLSX, JSONL)",
                type=["csv", "xlsx", "xls", "jsonl"],
            )
            extra_boxes_file = st.file_uploader(
                "Готовые рамки (boxes.json)",
                type=["json"],
            )
            registry_file = st.file_uploader(
                "Реестр объектов: asset_id, адрес, координаты (CSV/XLSX/JSONL)",
                type=["csv", "xlsx", "xls", "jsonl"],
            )

        has_dataset_input = bool(
            (upload_mode == "Фотографии" and uploaded_photos)
            or (upload_mode == "Папка" and (uploaded_dir_files or local_dir_path.strip()))
            or (upload_mode == "ZIP-архив" and uploaded_zip)
        )

        if upload_mode == "Фотографии" and uploaded_photos:
            st.caption(f"Выбрано фото: **{len(uploaded_photos)}**")
        elif upload_mode == "Папка" and uploaded_dir_files:
            n_imgs = sum(Path(f.name).suffix.lower() in IMG_EXT for f in uploaded_dir_files)
            st.caption(f"Файлов в папке: **{len(uploaded_dir_files)}** (изображений: **{n_imgs}**)")
        elif upload_mode == "Папка" and local_dir_path.strip():
            p_check = Path(local_dir_path.strip()).expanduser()
            if p_check.is_dir():
                st.caption(f"Локальная папка найдена: `{p_check}`")
            else:
                st.warning("Указанная локальная папка не найдена")
        elif upload_mode == "ZIP-архив" and uploaded_zip:
            st.caption(f"Архив: **{uploaded_zip.name}** ({uploaded_zip.size / 1024:.0f} КБ)")

        photo_limit = st.number_input(
            "Максимум фото за прогон",
            min_value=1,
            max_value=500,
            value=24,
            step=1,
            help="Ограничение стоимости и времени: в обработку попадут первые N фото в алфавитном порядке.",
        )

        st.divider()
        st.header("🤖 Модели и режим")
        profile_key = st.selectbox(
            "Профиль классификации",
            list(TAXONOMY_PROFILES.keys()),
            format_func=lambda k: TAXONOMY_PROFILES[k]["title"],
            help="Для шкафов МГТС (ТКШ / РЩ) выберите первый профиль.",
        )

        # Консенсус всегда считает qwen235: второе мнение даёт второй промт, а не другая модель.
        model_a_name = FRAME_CROP
        model_b_name = None
        ok_a, msg_a = variant_readiness(model_a_name, cfg_base)
        st.caption(f"Модель: **qwen235** · кадр и вырез · {msg_a}")
        if not ok_a:
            st.error(f"Маршрут не готов: {msg_a}")
        can_run = has_dataset_input and ok_a
        run_clicked = st.button(
            "🚀 Запустить анализ дата-сета",
            type="primary",
            width="stretch",
            disabled=not can_run,
        )

        example_clicked = st.button("Открыть готовый пример", width="stretch")

    # ==================== ЦЕНТРАЛЬНАЯ ОБЛАСТЬ ====================
    st.title("Диагностика сетевой инфраструктуры и шкафов по фото")

    if run_clicked or example_clicked:
        if example_clicked:
            model_a_name, model_b_name, profile_key = FRAME_CROP, None, "cabinets"
        cfg_run = apply_taxonomy_profile(cfg_base, profile_key)
        names = build_display_names(cfg_run)
        results_dir = ROOT / "out" / "app"
        results_dir.mkdir(parents=True, exist_ok=True)
        out = Path(mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=results_dir))
        (out / "annotated").mkdir(parents=True, exist_ok=True)

        # Центральный блок прогресса (лоад-бар)
        st.subheader("Ход обработки дата-сета")
        stage_box = st.empty()
        progress_bar = st.progress(0.0, text="Подготовка дата-сета…")
        live_stats_box = st.empty()

        try:
            registry = None
            example_predictions = {}
            if example_clicked:
                from defense_examples import load_examples
                examples = load_examples()
                if not examples:
                    raise ValueError("Сохранённые примеры недоступны")
                bundle = DatasetBundle(root=out, items=[])
                for path, prediction in examples:
                    bundle.items.append({"key": path.name, "path": path, "meta": {}})
                    example_predictions[path.name] = prediction
                pool_workers = 1
            else:
                stage_box.info("Этап 1/3: Распаковка и индексация дата-сета…")
                input_dir = out / "input"
                if upload_mode == "Фотографии":
                    ds_root = ingest_uploaded_photos(uploaded_photos, input_dir)
                elif upload_mode == "Папка":
                    if uploaded_dir_files:
                        ds_root = ingest_uploaded_directory(uploaded_dir_files, input_dir)
                    else:
                        ds_root = ingest_local_directory(Path(local_dir_path.strip()))
                else:
                    ds_root = ingest_zip_archive(uploaded_zip, input_dir)

                meta_path = None
                if extra_meta_file is not None:
                    meta_path = out / ("extra_meta" + Path(extra_meta_file.name).suffix)
                    meta_path.write_bytes(extra_meta_file.getbuffer())

                boxes_path = None
                if extra_boxes_file is not None:
                    boxes_path = out / "extra_boxes.json"
                    boxes_path.write_bytes(extra_boxes_file.getbuffer())

                if registry_file is not None:
                    reg_path = out / ("registry" + Path(registry_file.name).suffix)
                    reg_path.write_bytes(registry_file.getbuffer())
                    try:
                        registry = load_registry(read_meta(reg_path))
                        st.caption(f"Реестр: {len(registry)} объектов")
                    except ValueError as exc:
                        st.error(f"Реестр отклонён: {exc}")
                bundle = inspect_dataset(ds_root, extra_meta_file=meta_path, extra_boxes_file=boxes_path)
                if len(bundle.items) > photo_limit:
                    bundle.warnings.append(
                        f"Ограничение прогона: обработано {photo_limit} из {len(bundle.items)} фото."
                    )
                    bundle.items = bundle.items[:photo_limit]
                for w in bundle.warnings:
                    st.warning(w)

                progress_bar.progress(0.05, text=f"Найдено фотографий: {len(bundle.items)}. Загрузка моделей…")
                stage_box.info(
                    f"Этап 2/3: Загрузка моделей (`{model_a_name}`"
                    + (f" и `{model_b_name}`" if model_b_name else "")
                    + ")…"
                )

                model_a = _model_for_session("a", model_a_name, cfg_run)
                model_b = _model_for_session("b", model_b_name, cfg_run) if model_b_name else None
                kind_a = cfg_run["variants"][model_a_name].get("kind", "vlm")
                kind_b = cfg_run["variants"][model_b_name].get("kind", "vlm") if model_b_name else None

                if "boxes_dino" in (kind_a, kind_b) and bundle.boxes is None:
                    raise ValueError("Выбран вариант boxes_dino, но файл рамок boxes.json не загружен.")

                workers_a = getattr(model_a, "workers", 2)
                workers_b = getattr(model_b, "workers", 2) if model_b else workers_a
                pool_workers = max(1, min(6, max(workers_a, workers_b)))
                if kind_a in ("det_vlm", "det_dino", "grounding_qwen", "frame_crop_qwen", "photo_dino") or kind_b in (
                    "det_vlm", "det_dino", "grounding_qwen", "frame_crop_qwen", "photo_dino"
                ):
                    # Локальные нейросети имеют lock; 2 воркера позволяют перекрывать CPU и сетевой VLM без лишней памяти
                    pool_workers = min(pool_workers, 2)

            total = len(bundle.items)

            verdicts_by_key: dict[str, dict] = {}
            annotated_paths: dict[str, Path] = {}
            raw_preds: dict[str, dict] = {}
            t_start = time.monotonic()
            done_bad = done_review = done_ok = 0

            stage_box.info("Открываем сохранённые ответы…" if example_clicked else f"Этап 3/3: Анализ {total} фото и сверка результатов…")

            with ThreadPoolExecutor(max_workers=pool_workers) as executor:
                if example_clicked:
                    def replay(item):
                        with Image.open(item["path"]) as image:
                            return item["key"], image.convert("RGB"), example_predictions[item["key"]], None
                    futures = {executor.submit(replay, item): item for item in bundle.items}
                else:
                    futures = {
                        executor.submit(_predict_item, item, model_a, model_b, bundle.boxes, kind_a, kind_b): item
                        for item in bundle.items
                    }
                for done_idx, fut in enumerate(as_completed(futures), 1):
                    item = futures[fut]
                    key, rgb, pred_a, pred_b = fut.result()
                    if model_a_name == FRAME_CROP:
                        verdict = _frame_crop_verdict(pred_a, names)
                    else:
                        verdict = compare_predictions(pred_a, pred_b, model_a_name, model_b_name, names)
                    verdict = enrich_result(verdict, meta=bundle.meta_by_key.get(key, {}), registry=registry)
                    verdicts_by_key[key] = verdict
                    raw_preds[key] = {"model_a": pred_a, "model_b": pred_b, "verdict": verdict}

                    ann_img = annotate_verdict(rgb, verdict, names, model_a_name, model_b_name)
                    ann_target = (out / "annotated" / key).with_suffix(".jpg")
                    ann_target.parent.mkdir(parents=True, exist_ok=True)
                    ann_img.save(ann_target, quality=88)
                    annotated_paths[key] = ann_target

                    has_bad = any(o.get("status") == "bad" for o in verdict["objects"])
                    if has_bad:
                        done_bad += 1
                    if verdict["outcome"] == "review":
                        done_review += 1
                    elif not has_bad:
                        done_ok += 1

                    frac = 0.05 + 0.95 * (done_idx / total)
                    elapsed = time.monotonic() - t_start
                    eta = (elapsed / done_idx) * (total - done_idx)
                    progress_bar.progress(
                        min(1.0, frac),
                        text=f"Обработано {done_idx} / {total} фото ({done_idx * 100 // total}%) · {key} · осталось ~{eta:.0f} с",
                    )
                    with live_stats_box.container():
                        lc = st.columns(4)
                        lc[0].metric("Обработано", f"{done_idx} / {total}")
                        lc[1].metric("Поломок найдено", done_bad)
                        lc[2].metric("На доп. подтверждение", done_review)
                        lc[3].metric("Исправно", done_ok)

            # Сохранение результатов на диск
            tables = _export_results(bundle.items, verdicts_by_key, names)
            for t_name, t_rows in tables.items():
                write_csv(out / f"{t_name}.csv", t_rows)
            (out / "preds.json").write_text(json.dumps(raw_preds, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

            st.session_state["ui_result"] = {
                "out": out,
                "bundle": bundle,
                "verdicts": verdicts_by_key,
                "annotated": annotated_paths,
                "names": names,
                "model_a": FRAME_NAME if model_a_name == FRAME_CROP else model_a_name,
                "model_b": CROP_NAME if model_a_name == FRAME_CROP else model_b_name,
                "frame_crop": model_a_name == FRAME_CROP,
                "example": example_clicked,
                "profile_title": TAXONOMY_PROFILES[profile_key]["title"],
                "elapsed": round(time.monotonic() - t_start, 1),
            }
            st.session_state["ui_overrides"] = {}
            if example_clicked:
                stage_box.empty()
            else:
                stage_box.success(f"Обработано {total} фото за {time.monotonic() - t_start:.1f} с.")
            progress_bar.progress(1.0, text=f"Завершено: {total} / {total} фото (100%)")
            live_stats_box.empty()
        except Exception as exc:
            stage_box.error(f"Ошибка при обработке дата-сета: {exc}")
            progress_bar.empty()
            st.stop()

    # ==================== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ====================
    res = st.session_state.get("ui_result")
    if not res:
        st.info("Загрузите дата-сет (папку, фотографии или ZIP-архив) в левом сайдбаре, выберите модели и нажмите **«Запустить анализ дата-сета»**.")
        return

    bundle = res["bundle"]
    verdicts = res["verdicts"]
    annotated = res["annotated"]
    names = res["names"]
    model_a_name = res["model_a"]
    model_b_name = res["model_b"]
    overrides: dict[str, str] = st.session_state.setdefault("ui_overrides", {})

    summary = summarize_run(
        verdicts, bundle.items, bundle.gt_kind, names, model_a_name, model_b_name, overrides=overrides
    )

    st.divider()
    st.subheader("📊 Общие результаты по дата-сету")
    if res.get("example"):
        st.info("Сохранённый результат прогона на официальных фото МГТС. Открытие примера не вызывает API.")
    st.caption(
        f"Профиль: **{res['profile_title']}** · Модель 1: `{model_a_name}`"
        + (f" · Модель 2: `{model_b_name}`" if model_b_name else " · Одиночный режим")
        + ("" if res.get("example") else f" · Время прогона: {res['elapsed']} с")
        + (f" · API в исходном прогоне: ${summary['total_cost']:.5f}" if res.get("example")
           else f" · Стоимость по API: ${summary['total_cost']:.5f}")
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Всего фото", summary["total_photos"], help=f"Ошибок API/JSON: {summary['error_photos']}")
    k2.metric("Найдено объектов", summary["total_objects"], help="Общее количество обнаруженных шкафов/объектов")
    k3.metric("Фото с поломкой", summary["defect_photos"], delta=f"{summary['bad_objects']} объект(ов)", delta_color="inverse")
    pct_review = (summary["review_photos"] * 100 / summary["total_photos"]) if summary["total_photos"] else 0
    k4.metric("На доп. подтверждение", summary["review_photos"], delta=f"{pct_review:.0f}% фото", delta_color="off")
    k5.metric("Исправно (фото)", summary["confirmed_ok_photos"])
    if summary["agreement_rate"] is not None:
        timing_label = "с/фото в исходном прогоне" if res.get("example") else "с/фото"
        k6.metric("Согласие проверок", f"{summary['agreement_rate']:.0%}",
                  delta=f"{summary['avg_seconds']:.1f} {timing_label}", delta_color="off")
    else:
        k6.metric("Среднее время", f"{summary['avg_seconds']:.1f} с/фото", delta=f"${summary['total_cost']:.3f}" if summary["total_cost"] else None)
    st.caption(
        "Фото с подтверждённой поломкой и фото на проверку считаются отдельно: кадр попадает в оба списка, "
        "если на нём есть и подтверждённый дефект, и спорный объект."
    )

    # Аналитика: типы объектов, виды поломок и причины расхождения
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Обнаруженные объекты и виды повреждений**")
        if summary["type_counts"]:
            st.dataframe(
                pd.DataFrame(
                    [{"Тип объекта / шкафа": k, "Количество": v} for k, v in summary["type_counts"].items()]
                ),
                hide_index=True,
                width="stretch",
            )
        if summary["damage_counts"]:
            dmg_df = pd.DataFrame(
                [{"Вид повреждения": k, "Количество": v} for k, v in summary["damage_counts"].items()]
            )
            st.dataframe(dmg_df, hide_index=True, width="stretch")
        else:
            st.info("Подтверждённых повреждений не найдено.")

    with col_right:
        st.markdown("**Статус сверки и причины доп. проверки**")
        if summary["disagreement_reason_counts"]:
            reason_df = pd.DataFrame(
                [{"Причина": k, "Фото": v} for k, v in summary["disagreement_reason_counts"].items()]
            )
            st.dataframe(reason_df, hide_index=True, width="stretch")
        else:
            st.success("Все фотографии классифицированы без расхождений и ошибок.")
        if summary.get("priority_counts"):
            st.caption("Приоритет починки: " + " · ".join(
                f"{PRIORITY_RU.get(k, k)} — {v}" for k, v in summary["priority_counts"].items()))

    # Метрики качества по эталонной разметке (если она была в загруженном датасете)
    if summary["gt_metrics"]:
        st.markdown("### 🎯 Метрики качества (по эталонной разметке дата-сета)")
        cons_m = summary["gt_metrics"]["consensus"]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy (авто-решения)", f"{cons_m['accuracy_auto']:.1%}", help=f"Автоматически решено {cons_m['auto_decided']} из {cons_m['labeled_photos']}")
        mc2.metric("Полнота поломок (Recall)", f"{cons_m['defect_recall_strict']:.1%}", help=f"С учётом отправленных на проверку: {cons_m['defect_recall_with_review']:.1%}")
        mc3.metric("Точность поломок (Precision)", f"{cons_m['defect_precision']:.1%}")
        mc4.metric("F1 по поломкам", f"{cons_m['f1_bad']:.3f}")
        mc5.metric(
            "Точность типа шкафа",
            f"{cons_m['type_accuracy']:.1%}" if cons_m["type_accuracy"] is not None else "—",
        )

        comp_rows = []
        for label_title, key_m in [("Консенсус", "consensus"), (f"Модель 1 ({model_a_name})", "model_a"), (f"Модель 2 ({model_b_name})", "model_b")]:
            m_data = summary["gt_metrics"].get(key_m)
            if not m_data:
                continue
            comp_rows.append({
                "Модель": label_title,
                "Авто-решений": f"{m_data['auto_decided']} / {m_data['labeled_photos']}",
                "Accuracy": f"{m_data['accuracy_auto']:.1%}",
                "Defect Recall (строгий)": f"{m_data['defect_recall_strict']:.1%}",
                "Defect Recall (с проверкой)": f"{m_data['defect_recall_with_review']:.1%}",
                "Defect Precision": f"{m_data['defect_precision']:.1%}",
                "Точность типа": f"{m_data['type_accuracy']:.1%}" if m_data["type_accuracy"] is not None else "—",
                "Точность вида дефекта": f"{m_data['defect_type_accuracy']:.1%}" if m_data["defect_type_accuracy"] is not None else "—",
            })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")

    elif summary["gt_object_metrics"]:
        st.markdown("### 🎯 Метрики детекции и ремонта (по `labels.json`)")
        gm = summary["gt_object_metrics"]
        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
        gc1.metric("Det Precision", f"{gm['det_precision']:.1%}")
        gc2.metric("Det Recall", f"{gm['det_recall']:.1%}")
        gc3.metric("Repair Recall", f"{gm['repair_recall']:.1%}")
        gc4.metric("Repair Precision", f"{gm['repair_precision']:.1%}")
        gc5.metric("Damage Acc", f"{gm['damage_acc']:.1%}")

    # Экспорт таблиц (с учётом ручных подтверждений оператора)
    export_tables = _export_results(bundle.items, verdicts, names, overrides=overrides)
    d1, d2, d3, d4, d5 = st.columns(5)
    for col, fname, tkey in (
        (d1, "repair.csv", "repair"),
        (d2, "review.csv", "review"),
        (d3, "objects.csv", "objects"),
        (d4, "frames.csv", "frames"),
    ):
        df_exp = pd.DataFrame(export_tables[tkey]) if export_tables[tkey] else pd.DataFrame([{"info": "Пусто"}])
        col.download_button(
            f"⬇ {fname}",
            df_exp.to_csv(index=False).encode("utf-8-sig"),
            fname,
            "text/csv",
            width="stretch",
        )
    d5.download_button(
        "⬇ report.xlsx",
        _to_excel_bytes(export_tables),
        "report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

    # Карта координат (если в дата-сете или EXIF есть lat/lon)
    if bundle.has_gps:
        pts = []
        color_map = {"bad": "#e62828", "ok": "#28c846", "review": "#ff9800"}
        for it in bundle.items:
            m = it.get("meta") or {}
            lat_val = next((m[c] for c in m if str(c).lower() in LAT_COLS), None)
            lon_val = next((m[c] for c in m if str(c).lower() in LON_COLS), None)
            if lat_val not in (None, "") and lon_val not in (None, ""):
                eff = _effective_outcome(it["key"], verdicts[it["key"]], overrides)
                pts.append({"lat": lat_val, "lon": lon_val, "color": color_map.get(eff, "#ff9800")})
        if pts:
            pts_df = pd.DataFrame(pts)
            pts_df["lat"] = pd.to_numeric(pts_df["lat"], errors="coerce")
            pts_df["lon"] = pd.to_numeric(pts_df["lon"], errors="coerce")
            pts_df = pts_df.dropna()
            if len(pts_df):
                with st.expander("🗺 Карта объектов", expanded=False):
                    st.map(pts_df, color="color", size=8)

    st.divider()

    # ==================== СПИСОК 1: УСПЕШНО КЛАССИФИЦИРОВАННЫЕ ПОЛОМКИ ====================

    def _has_confirmed_bad(item: dict) -> bool:
        key = item["key"]
        if _effective_outcome(key, verdicts[key], overrides) == "bad":
            return True
        return any(o.get("status") == "bad" for o in verdicts[key]["objects"])

    bad_items = [it for it in bundle.items if _has_confirmed_bad(it)]
    bad_items.sort(key=lambda it: (priority_rank(verdicts[it["key"]].get("priority")),
                                   -sum(o.get("status") == "bad" for o in verdicts[it["key"]]["objects"])))
    review_items = [it for it in bundle.items if _effective_outcome(it["key"], verdicts[it["key"]], overrides) == "review"]
    ok_items = [it for it in bundle.items if _effective_outcome(it["key"], verdicts[it["key"]], overrides) == "ok"]

    st.subheader(f"🔴 Успешно классифицированные поломки ({len(bad_items)})")
    if not bad_items:
        st.info("Подтверждённых поломок нет.")
    else:
        f_col1, f_col2 = st.columns([2, 1])
        search_bad = f_col1.text_input("Поиск по имени файла или описанию", key="search_bad")
        all_bad_types = sorted({
            type_label(o.get("type"), names)
            for it in bad_items
            for o in verdicts[it["key"]]["objects"]
        })
        filter_type = f_col2.multiselect("Фильтр по типу шкафа/объекта", all_bad_types, key="filter_bad_type")

        for it in bad_items:
            key = it["key"]
            v = verdicts[key]
            objs = v["objects"]
            obj_types = [type_label(o.get("type"), names) for o in objs]
            bad_dmgs = [damage_label(o.get("damage") or o.get("candidate_damage"), names) for o in objs if o.get("status") == "bad" or overrides.get(key) == "bad"]
            if search_bad:
                q = search_bad.casefold()
                hay = " ".join([key] + [o.get("description", "") for o in objs] + obj_types + bad_dmgs).casefold()
                if q not in hay:
                    continue
            if filter_type and not any(t in filter_type for t in obj_types):
                continue

            types_hdr = ", ".join(dict.fromkeys(obj_types)) or "Объект"
            dmgs_hdr = ", ".join(dict.fromkeys(bad_dmgs)) or "Повреждение"
            override_tag = " · подтверждено оператором" if overrides.get(key) == "bad" else ""
            pr = PRIORITY_RU.get(v.get("priority", "none"), "—")
            header = f"🔴 [{pr}] {key} — {types_hdr} · {dmgs_hdr}{override_tag}"
            with st.expander(header, expanded=len(bad_items) <= 3):
                img_col, info_col = st.columns([3, 2])
                img_col.image(str(annotated[key]), width="stretch")
                with info_col:
                    _render_item_meta(it, v)
                    _render_object_cards(v, names, show_agreement_note=True)
                    if overrides.get(key) == "bad":
                        _render_disagreement_section(v, names)
                    elif v.get("checks"):
                        _render_model_opinion_block(v["model_a"], "Целое фото", names, "")
                        _render_model_opinion_block(v["model_b"], "Основная рамка", names, "")

    # ==================== СПИСОК 2: ФОТО ДЛЯ ДОП. ПОДТВЕРЖДЕНИЯ ====================
    st.subheader(f"🟠 Фото для дополнительного подтверждения ({len(review_items)})")
    if not review_items:
        st.success("Спорных фотографий, требующих дополнительного подтверждения, нет.")
    else:
        if res.get("frame_crop"):
            st.caption("Синяя пунктирная рамка показывает корпус второй проверки. Первая проверка оценивает весь кадр.")
        else:
            st.caption("Оранжевая сплошная рамка — Модель 1; синяя пунктирная рамка — Модель 2 (при наличии обеих моделей).")
        for it in review_items:
            key = it["key"]
            v = verdicts[key]
            objs = v["objects"]
            obj_types = sorted({type_label(o.get("type"), names) for o in objs})
            types_hdr = ", ".join(obj_types) if obj_types else "Объект не подтверждён"
            short_reason = v["disagreement_reasons"][0] if v["disagreement_reasons"] else "Требуется ручная проверка"
            header = f"🟠 {key} — {types_hdr} · {short_reason}"
            with st.expander(header, expanded=len(review_items) <= 5):
                img_col, info_col = st.columns([3, 2])
                img_col.image(str(annotated[key]), width="stretch")
                with info_col:
                    _render_item_meta(it, v)
                    st.markdown("#### Классификация и описания объектов")
                    _render_object_cards(v, names, show_agreement_note=False)
                st.divider()
                _render_disagreement_section(v, names)
                st.divider()
                prev = overrides.get(key, "review")
                choice = st.radio(
                    "Решение оператора",
                    ["Оставить на проверке", "Подтвердить поломку", "Признать исправным"],
                    index={"review": 0, "bad": 1, "ok": 2}[prev],
                    key=f"op_decision_{res['out'].name}_{key}",
                    horizontal=True,
                )
                mapped = {"Оставить на проверке": "review", "Подтвердить поломку": "bad", "Признать исправным": "ok"}[choice]
                if mapped != prev:
                    if mapped == "review":
                        overrides.pop(key, None)
                    else:
                        overrides[key] = mapped
                    for table_name, rows in _export_results(bundle.items, verdicts, names, overrides).items():
                        write_csv(res["out"] / f"{table_name}.csv", rows)
                    st.rerun()

    # ==================== СПИСОК 3: ИСПРАВНЫЕ ОБЪЕКТЫ ====================
    st.subheader(f"🟢 Успешно классифицированные: исправные объекты ({len(ok_items)})")
    if not ok_items:
        st.info("Исправных фотографий в результатах нет.")
    elif st.checkbox("Показать список исправных фотографий", value=False, key="show_ok_list"):
        for it in ok_items:
            key = it["key"]
            v = verdicts[key]
            obj_types = ", ".join(dict.fromkeys(type_label(o.get("type"), names) for o in v["objects"])) or "Без целевых объектов"
            with st.expander(f"🟢 {key} — {obj_types}", expanded=False):
                img_col, info_col = st.columns([3, 2])
                img_col.image(str(annotated[key]), width="stretch")
                with info_col:
                    _render_item_meta(it, v)
                    _render_object_cards(v, names, show_agreement_note=True)
                    if v.get("checks"):
                        _render_model_opinion_block(v["model_a"], "Целое фото", names, "")
                        _render_model_opinion_block(v["model_b"], "Основная рамка", names, "")


if __name__ == "__main__":
    main()
