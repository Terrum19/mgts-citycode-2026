"""Интерфейс для защиты: фото -> рамки и описания -> перечень на ремонт, карта, CSV.

    streamlit run src/app.py

Использует ту же функцию, что и src/predict.py, результаты прогона сохраняются в out/app/<время>/.
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants  # noqa: E402
from predict import IMG_EXT, index_meta, read_boxes, read_meta, run, write_csv  # noqa: E402
from run import load_config  # noqa: E402
from vlm import ROOT  # noqa: E402

LAT = ("lat", "latitude", "широта", "pano_lat")
LON = ("lon", "lng", "longitude", "долгота", "pano_lon")


def weight_files_version(directories):
    versions = []
    for directory in directories:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                info = path.stat()
                versions.append((path, info.st_ino, info.st_size, info.st_ctime_ns))
    return versions


def model_for_session(name, config):
    settings = config["variants"][name]
    if settings.get("kind") not in ("det_dino", "boxes_dino", "photo_dino") or settings.get("qwen"):
        st.session_state.pop("local_model", None)
        return variants.make(name, config)
    checkpoint = (ROOT / settings["checkpoint"]).stat()
    identity = (name, config, checkpoint.st_ino, checkpoint.st_size, checkpoint.st_ctime_ns)
    cached = st.session_state.get("local_model")
    if cached is not None and cached[0] == identity:
        directories = cached[1]
        if all(path.is_dir() for path in directories) and weight_files_version(directories) == cached[2]:
            return cached[3]
    st.session_state.pop("local_model", None)
    del cached
    model = variants.make(name, config)
    directories = [getattr(model.encoder, "path", ROOT / "weights/dinov2-base")]
    if settings["kind"] == "det_dino":
        directories.append(Path(config["detector"]["model"]))
    if all(path.is_dir() for path in directories):
        st.session_state.local_model = (identity, directories, weight_files_version(directories), model)
    return model


st.set_page_config(page_title="МГТС: осмотр инфраструктуры", layout="wide")
cfg = load_config()
names = cfg.get("names", {})

with st.sidebar:
    st.header("Настройки")
    variant = st.selectbox("Вариант", list(cfg["variants"]), help="Описаны в config.toml, раздел [variants]")
    v = cfg["variants"][variant]
    st.caption(f"{v.get('kind', 'vlm')} · {v.get('model') or v.get('vlm')}")
    only_bad = st.checkbox("Показывать только фото с повреждениями", value=False)

st.title("Осмотр сетевой инфраструктуры по фото")
c1, c2 = st.columns([3, 2])
files = c1.file_uploader("Фото", type=[e[1:] for e in IMG_EXT], accept_multiple_files=True)
meta_file = c2.file_uploader("Таблица с ID / координатами фото (необязательно)", type=["csv", "xlsx", "jsonl"])

boxes_file = c2.file_uploader("Рамки объектов", type=["json"]) if v.get("kind") == "boxes_dino" else None
if st.button("Запустить", type="primary", disabled=not files or (v.get("kind") == "boxes_dino" and not boxes_file)):
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
    model = model_for_session(variant, cfg)

    def progress(k, n, p, pred):
        bar.progress(k / n, text=f"{k}/{n}: {p.name}")

    rows, preds = run(model, paths, meta, names, out, model.workers, progress, boxes)
    write_csv(out / "objects.csv", rows)
    write_csv(out / "repair.csv", [r for r in rows if r["status"] == "bad"])
    write_csv(out / "review.csv", [r for r in rows if r["status"] == "review"])
    st.session_state["result"] = {"out": out, "rows": rows, "preds": preds, "paths": paths}
    bar.empty()

res = st.session_state.get("result")
if res:
    rows, preds, out = res["rows"], res["preds"], res["out"]
    df = pd.DataFrame(rows)
    bad = df[df["status"] == "bad"] if len(df) else df
    review = df[df["status"] == "review"] if len(df) else df
    errors = sum(bool(p["error"]) for p in preds.values())
    m = st.columns(5)
    m[0].metric("Фото", len(preds))
    m[1].metric("Объектов", len(df))
    m[2].metric("На ремонт", len(bad))
    m[3].metric("Сек/фото", f"{sum(p['seconds'] for p in preds.values()) / max(1, len(preds)):.1f}")
    m[4].metric("Ошибок модели", errors)

    st.subheader("Перечень на проверку / ремонт")
    if len(bad):
        hide = ["status", "type", "damage", "bbox"]
        st.dataframe(bad.drop(columns=[c for c in hide if c in bad]), width="stretch", hide_index=True)
    else:
        st.info("Повреждённых объектов не найдено.")
    if len(review):
        st.subheader("На ручную проверку")
        st.dataframe(review, hide_index=True)
    d1, d2, d3 = st.columns(3)
    for col, name, data in ((d1, "repair.csv", bad), (d2, "objects.csv", df), (d3, "review.csv", review)):
        buf = io.StringIO()
        data.to_csv(buf, index=False)
        col.download_button(f"Скачать {name}", buf.getvalue().encode("utf-8-sig"), name, "text/csv")

    lat = next((c for c in df.columns if c.lower() in LAT), None)
    lon = next((c for c in df.columns if c.lower() in LON), None)
    if lat and lon and len(df):
        st.subheader("Карта")
        pts = df[[lat, lon, "status"]].copy()
        pts[lat] = pd.to_numeric(pts[lat], errors="coerce")
        pts[lon] = pd.to_numeric(pts[lon], errors="coerce")
        pts = pts.dropna().drop_duplicates()
        pts["color"] = pts["status"].map({"bad": "#e62828", "ok": "#28c846", "review": "#ffd200"})
        st.map(pts.rename(columns={lat: "lat", lon: "lon"}), color="color", size=6)

    st.subheader("Фото")
    for p in res["paths"]:
        pred = preds[p.name]
        n_bad = sum(o["status"] == "bad" for o in pred["objects"])
        if only_bad and not n_bad:
            continue
        with st.expander(f"{p.name} — объектов {len(pred['objects'])}, повреждённых {n_bad}", expanded=n_bad > 0):
            a, b = st.columns([3, 2])
            a.image(str(out / "annotated" / (p.stem + ".jpg")), width="stretch")
            if pred["error"]:
                b.error(pred["error"])
            for k, o in enumerate(pred["objects"], 1):
                tag = names.get(o["type"], o["type"])
                if o["status"] == "bad":
                    b.markdown(f"**{k}. {tag}** — :red[{names.get(o['damage'], o['damage'])}]  \n{o['description']}")
                elif o["status"] == "review":
                    b.markdown(f"**{k}. {tag}** — на проверку")
                else:
                    b.markdown(f"**{k}. {tag}** — :green[исправен]  \n{o['description']}")
