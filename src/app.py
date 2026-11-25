"""Интерфейс для защиты: фото -> рамки и описания -> очередь triage, карта/схема, CSV.

    streamlit run src/app.py

Использует ту же функцию, что src/predict.py; результаты сохраняются в out/app/<время>/.
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants  # noqa: E402
from config_utils import is_offline, resolve_variant  # noqa: E402
from export_rows import export_rows  # noqa: E402
from predict import IMG_EXT, index_meta, read_boxes, read_meta, run, write_csv  # noqa: E402
from run import load_config  # noqa: E402
from vlm import ROOT  # noqa: E402

LAT = ("lat", "latitude", "широта", "pano_lat")
LON = ("lon", "lng", "longitude", "долгота", "pano_lon")

st.set_page_config(page_title="МГТС: осмотр инфраструктуры", layout="wide")
cfg = load_config()
names = cfg.get("names", {})

with st.sidebar:
    st.header("Настройки")
    variant = st.selectbox("Вариант", list(cfg["variants"]), help="Описаны в config.toml, раздел [variants]")
    v = resolve_variant(cfg, variant)
    st.caption(f"{v.get('kind', 'vlm')} · {v.get('model') or v.get('vlm')}")
    only_bad = st.checkbox("Показывать только фото с повреждениями", value=False)

st.title("Осмотр сетевой инфраструктуры по фото")
c1, c2 = st.columns([3, 2])
files = c1.file_uploader("Фото", type=[e[1:] for e in IMG_EXT], accept_multiple_files=True)
meta_file = c2.file_uploader("Таблица с ID / координатами фото (необязательно)", type=["csv", "xlsx", "jsonl"])

boxes_kind = v.get("kind") in ("boxes_dino", "boxes_vlm")
boxes_file = c2.file_uploader("Рамки объектов", type=["json"]) if boxes_kind else None
if st.button("Запустить", type="primary", disabled=not files or (boxes_kind and not boxes_file)):
    out = ROOT / "out" / "app" / time.strftime("%Y%m%d-%H%M%S")
    (out / "input").mkdir(parents=True)
    paths = []
    for f in files:
        p = out / "input" / Path(f.name).name
        p.write_bytes(f.getbuffer())
        paths.append(p)
    meta = {}
    if meta_file:
        mp = out / ("meta" + Path(meta_file.name).suffix)
        mp.write_bytes(meta_file.getbuffer())
        meta = index_meta(read_meta(mp), [p.name for p in paths])
    boxes = None
    if boxes_file:
        box_path = out / "boxes.json"
        box_path.write_bytes(boxes_file.getbuffer())
        boxes = read_boxes(box_path)
    if meta_file and len(meta) != len(paths):
        st.warning(f"У {len(paths) - len(meta)} фото нет однозначной привязки к таблице")
    bar = st.progress(0.0, text="Загружаю модель…")
    model = variants.make(variant, cfg)

    def progress(k, n, p, pred):
        bar.progress(k / n, text=f"{k}/{n}: {p.name}")

    rows, preds = run(model, paths, meta, names, out, model.workers, progress, boxes, cfg.get("quality_gate"))
    tables = export_rows(preds, meta, names)
    for name in ("objects", "frames", "repair", "review", "triage"):
        write_csv(out / f"{name}.csv", tables[name])
    st.session_state["result"] = {"out": out, "tables": tables, "preds": preds, "paths": paths}
    bar.empty()

res = st.session_state.get("result")
if res:
    tables, preds, out = res["tables"], res["preds"], res["out"]
    frames = pd.DataFrame(tables["frames"])
    repair = pd.DataFrame(tables["repair"])
    review = pd.DataFrame(tables["review"])
    triage = pd.DataFrame(tables["triage"])
    errors = sum(bool(p.get("error")) for p in preds.values())
    m = st.columns(5)
    m[0].metric("Фото", len(preds))
    m[1].metric("Объектов", len(tables["objects"]))
    m[2].metric("В очереди реагирования", len(triage))
    m[3].metric("Подтверждено bad", len(repair))
    m[4].metric("Требует review", len(review))
    st.warning("**review — обязательная проверка специалистом или выезд по регламенту, "
               "а не отброшенный результат.**")

    st.subheader("Очередь реагирования (bad + review)")
    if len(triage):
        st.dataframe(triage, hide_index=True, width="stretch")
    else:
        st.info("Очередь пуста: подтверждённых дефектов и review нет.")
    st.subheader("Кадровые результаты")
    if len(frames):
        cols = [c for c in ("file", "status", "review", "review_reasons", "triage", "n_objects",
                            "n_invalid_bbox", "error") if c in frames]
        st.dataframe(frames[cols] if cols else frames, hide_index=True, width="stretch")
    st.subheader("Подтверждённый ремонт")
    if len(repair):
        st.dataframe(repair, hide_index=True, width="stretch")
    else:
        st.info("Подтверждённых bad нет.")

    d1, d2, d3, d4 = st.columns(4)
    for col, name, data in ((d1, "triage.csv", triage), (d2, "frames.csv", frames),
                            (d3, "repair.csv", repair), (d4, "objects.csv", pd.DataFrame(tables["objects"]))):
        buf = io.StringIO()
        data.to_csv(buf, index=False)
        col.download_button(f"Скачать {name}", buf.getvalue().encode("utf-8-sig"), name, "text/csv")

    lat = next((c for c in frames.columns if str(c).lower() in LAT), None)
    lon = next((c for c in frames.columns if str(c).lower() in LON), None)
    if lat and lon and len(frames):
        st.subheader("Схема координат")
        pts = frames[[lat, lon, "status"]].copy()
        pts[lat] = pd.to_numeric(pts[lat], errors="coerce")
        pts[lon] = pd.to_numeric(pts[lon], errors="coerce")
        pts = pts.dropna().drop_duplicates()
        pts["color"] = pts["status"].map({"bad": "#e62828", "ok": "#28c846", "review": "#ffd200"})
        if is_offline(v, cfg) or cfg.get("app", {}).get("offline_map", True):
            st.caption("Координатная схема, а не географическая карта: подложка не скачивается.")
            st.scatter_chart(pts.rename(columns={lat: "lat", lon: "lon"}), x="lon", y="lat", color="status")
        else:
            st.map(pts.rename(columns={lat: "lat", lon: "lon"}), color="color", size=6)

    st.subheader("Фото")
    for p in res["paths"]:
        pred = preds[p.name]
        frame = pred.get("frame") or {}
        n_bad = sum(o.get("status") == "bad" for o in (pred.get("objects") or []))
        n_review = sum(o.get("status") == "review" for o in (pred.get("objects") or []))
        if only_bad and not n_bad and frame.get("status") != "bad":
            continue
        title = (f"{p.name} — frame {frame.get('status', '?')}, объектов {len(pred.get('objects') or [])}, "
                 f"bad {n_bad}, review {n_review}")
        with st.expander(title, expanded=n_bad > 0 or frame.get("status") == "review"):
            a, b = st.columns([3, 2])
            a.image(str(out / "annotated" / (p.stem + ".jpg")), width="stretch")
            if pred.get("error"):
                b.error(pred["error"])
            if frame.get("review") or frame.get("status") == "review":
                reasons = ", ".join(frame.get("review_reasons") or [])
                b.warning(f"Кадр: review{f' ({reasons})' if reasons else ''}")
            for k, o in enumerate(pred.get("objects") or [], 1):
                tag = names.get(o.get("type"), o.get("type"))
                if o["status"] == "bad":
                    b.markdown(f"**{k}. {tag}** — :red[{names.get(o['damage'], o['damage'])}]  \n"
                               f"{o.get('description', '')}")
                elif o["status"] == "review":
                    reasons = ", ".join(o.get("review_reasons") or [])
                    b.markdown(f"**{k}. {tag}** — на проверку{f' ({reasons})' if reasons else ''}")
                else:
                    b.markdown(f"**{k}. {tag}** — :green[исправен]  \n{o.get('description', '')}")
