"""Система целиком: папка фото (+ таблица с ID/координатами) -> рамки, описания, перечень на ремонт.

    python src/predict.py --variant qwen235 --images photos/ --out out/run1
    python src/predict.py --variant qwen235 --images photos/ --meta photos.csv --out out/run1
    python src/predict.py --variant qwen235 --images data/raw/panoramas --meta data/raw/panoramas/meta.jsonl --limit 20 --out out/demo

--meta — CSV/XLSX/JSONL от организаторов: строка на фото; колонка с именем файла находится сама
(значения совпадают с именами фото), остальные колонки (ID, широта, долгота, адрес...) переносятся в отчёт.

Выход:
  annotated/<фото>.jpg   рамки: зелёная — исправен, красная — повреждён
  objects.csv            все найденные объекты
  repair.csv             перечень на проверку/ремонт (только повреждённые)
  preds.json             сырые ответы (для отладки)
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants  # noqa: E402
from draw import annotate  # noqa: E402
from run import load_config  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def names_ru(cfg: dict) -> dict:
    return cfg.get("names", {})


def read_meta(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix in (".xlsx", ".xls"):
        import pandas as pd  # только для Excel
        return pd.read_excel(path).astype(str).to_dict("records")
    text = path.read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    return list(csv.DictReader(text.splitlines(), dialect=dialect))


def index_meta(rows: list[dict], files: list[str]) -> dict[str, dict]:
    """{имя файла -> строка}: колонку с именами файлов ищем по совпадениям (с расширением или без)."""
    if not rows:
        return {}
    names = {f: f for f in files} | {Path(f).stem: f for f in files}
    best = max(rows[0], key=lambda c: sum(Path(str(r.get(c, ""))).name in names
                                          or Path(str(r.get(c, ""))).stem in names for r in rows))
    out = {}
    for r in rows:
        v = Path(str(r.get(best, "")))
        f = names.get(v.name) or names.get(v.stem)
        if f:
            out[f] = {k: v for k, v in r.items() if k != best}
    print(f"meta: колонка с файлом «{best}», совпало {len(out)} из {len(files)} фото", file=sys.stderr)
    return out


def object_rows(file: str, pred: dict, meta: dict, names: dict) -> list[dict]:
    ours = {"file", "object_n", "type", "type_ru", "status", "damage", "damage_ru", "description", "bbox"}
    meta = {("meta_" + k if k in ours else k): v for k, v in meta.items()}  # не затирать колонки организаторов
    rows = []
    for k, o in enumerate(pred["objects"], 1):
        rows.append({"file": file, "object_n": k, **meta,
                     "type": o["type"], "type_ru": names.get(o["type"], o["type"]),
                     "status": o["status"], "damage": o["damage"] or "",
                     "damage_ru": names.get(o["damage"], o["damage"] or "") if o["status"] == "bad" else "",
                     "description": o.get("description", ""), "bbox": " ".join(map(str, o["bbox"]))})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["file"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel откроет кириллицу
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def run(model, paths: list[Path], meta: dict, names: dict, out: Path | None = None, workers: int = 2,
        on_done=None) -> tuple[list[dict], dict]:
    """Общая функция для CLI и интерфейса -> (строки объектов, сырые предсказания)."""
    if out:
        (out / "annotated").mkdir(parents=True, exist_ok=True)

    def one(p: Path):
        img = Image.open(p)
        pred = model.predict(img)
        if out:
            annotate(img, pred["objects"], names=names).save(out / "annotated" / (p.stem + ".jpg"), quality=88)
        return p, pred

    rows, preds = [], {}
    with ThreadPoolExecutor(workers) as ex:
        for k, (p, pred) in enumerate(ex.map(one, paths), 1):
            preds[p.name] = pred
            rows += object_rows(p.name, pred, meta.get(p.name, {}), names)
            if on_done:
                on_done(k, len(paths), p, pred)
    return rows, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--images", required=True, help="папка с фото или одно фото")
    ap.add_argument("--meta", help="CSV/XLSX/JSONL с ID и координатами фото")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = load_config()
    names = names_ru(cfg)

    src = Path(args.images)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    paths = paths[:args.limit] if args.limit else paths
    meta = index_meta(read_meta(Path(args.meta)), [p.name for p in paths]) if args.meta else {}
    model = variants.make(args.variant, cfg)
    out = Path(args.out)

    def progress(k, n, p, pred):
        bad = sum(o["status"] == "bad" for o in pred["objects"])
        flag = f"ERR {pred['error'][:80]}" if pred["error"] else f"{len(pred['objects'])} obj, bad {bad}"
        print(f"  {k}/{n} {p.name}: {flag} {pred['seconds']}s", file=sys.stderr)

    rows, preds = run(model, paths, meta, names, out, model.workers, progress)
    write_csv(out / "objects.csv", rows)
    repair = [r for r in rows if r["status"] == "bad"]
    write_csv(out / "repair.csv", repair)
    (out / "preds.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1))
    errors = sum(bool(p["error"]) for p in preds.values())
    cost = sum(p.get("cost", 0) for p in preds.values())
    print(f"\n{len(paths)} фото, объектов {len(rows)}, на ремонт {len(repair)}, ошибок API {errors}, ${cost:.3f}")
    print(f"-> {out}/repair.csv, objects.csv, annotated/")


if __name__ == "__main__":
    main()
