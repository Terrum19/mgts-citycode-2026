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
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants  # noqa: E402
from asset import enrich_result, load_registry  # noqa: E402
from draw import annotate  # noqa: E402
from run import load_config  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def names_ru(cfg: dict) -> dict:
    return cfg.get("names", {})


def exif_gps(path: Path) -> dict:
    """Широта/долгота из EXIF. Пустой dict, если GPS нет или координаты некорректны:
    точку (0, 0) и мусор не подставляем."""
    try:
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)
    except Exception:
        return {}
    if not gps:
        return {}

    def to_deg(value) -> float | None:
        try:
            deg, minutes, seconds = (float(v) for v in value)
        except Exception:
            return None
        return deg + minutes / 60 + seconds / 3600

    lat, lon = to_deg(gps.get(2) or ()), to_deg(gps.get(4) or ())
    if lat is None or lon is None:
        return {}
    if str(gps.get(1, "N")).strip().upper().startswith("S"):
        lat = -lat
    if str(gps.get(3, "E")).strip().upper().startswith("W"):
        lon = -lon
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {}
    return {"lat": round(lat, 7), "lon": round(lon, 7)}


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
    import pandas as pd
    from folders import match_table

    return match_table(pd.DataFrame(rows), files)


def photo_keys(paths: list[Path]) -> dict[Path, str]:
    """Стабильный ключ фото: путь относительно общего корня. Имена файлов в выдаче повторяются
    между папками, поэтому ключ по одному имени склеил бы разные объекты."""
    if not paths:
        return {}
    root = Path(os.path.commonpath([str(p) for p in paths]))
    root = root if root.is_dir() else root.parent
    keys = {}
    for p in paths:
        try:
            keys[p] = p.relative_to(root).as_posix()
        except ValueError:
            keys[p] = p.name
    return keys


def object_rows(file: str, pred: dict, meta: dict, names: dict) -> list[dict]:
    ours = {"file", "object_n", "type", "type_ru", "status", "damage", "damage_ru", "description", "bbox", "object_id",
            "scope", "asset_id", "identity_status", "identity_source", "registry_status",
            "registry_lat", "registry_lon", "registry_address", "coords_source", "photo_lat", "photo_lon",
            "functional_effect", "recommendation", "urgency", "assessment_basis"}
    meta = {("meta_" + k if k in ours else k): v for k, v in meta.items()}  # не затирать колонки организаторов
    asset = pred.get("asset") or {}
    assessment = pred.get("assessment") or {}
    rows = []
    for k, o in enumerate(pred["objects"], 1):
        rows.append({"file": file, "object_n": k, **meta,
                     "object_id": o.get("object_id", ""), "type": o["type"], "type_ru": names.get(o["type"], o["type"]),
                     "status": o["status"], "damage": o["damage"] or "",
                     "damage_ru": names.get(o["damage"], o["damage"] or "") if o["status"] == "bad" else "",
                     "description": o.get("description", ""), "bbox": " ".join(map(str, o["bbox"] or [])),
                     "scope": pred.get("scope", ""),
                     "asset_id": asset.get("asset_id") or "",
                     "identity_status": asset.get("identity_status", ""),
                     "identity_source": asset.get("identity_source", ""),
                     "registry_status": asset.get("registry_status", ""),
                     "registry_lat": asset.get("lat") if asset.get("lat") is not None else "",
                     "registry_lon": asset.get("lon") if asset.get("lon") is not None else "",
                     "registry_address": asset.get("address") or "",
                     "coords_source": asset.get("coords_source", ""),
                     "photo_lat": asset.get("photo_lat") if asset.get("photo_lat") is not None else "",
                     "photo_lon": asset.get("photo_lon") if asset.get("photo_lon") is not None else "",
                     "functional_effect": assessment.get("effect", ""),
                     "recommendation": assessment.get("recommendation", ""),
                     "urgency": assessment.get("urgency", ""),
                     "assessment_basis": assessment.get("basis", "")})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["file"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel откроет кириллицу
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def read_boxes(path):
    """JSON с записями {file, objects: [{bbox: [x1,y1,x2,y2], type?, object_id?}]}."""
    from math import isfinite
    boxes = {}
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(records, dict):
        for filename, prediction in records.items():
            if not isinstance(prediction, dict) or not isinstance(prediction.get("detections"), list):
                raise ValueError(f"Нет сохранённых рамок детектора для: {filename}")
        records = [{"file": filename, "objects": prediction["detections"]}
                   for filename, prediction in records.items()]
    for row in records:
        name = row["file"]
        if name in boxes:
            raise ValueError(f"Повтор фото в файле рамок: {name}")
        for obj in row["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            if not (all(isfinite(v) for v in (x1, y1, x2, y2)) and 0 <= x1 < x2 and 0 <= y1 < y2):
                raise ValueError(f"Некорректная рамка: {name}")
        boxes[name] = row["objects"]
    return boxes


def run(model, paths: list[Path], meta: dict, names: dict, out: Path | None = None, workers: int = 2,
        on_done=None, boxes=None, registry: dict | None = None) -> tuple[list[dict], dict]:
    """Общая функция для CLI и интерфейса -> (строки объектов, сырые предсказания)."""
    keys = photo_keys(paths)
    if out:
        (out / "annotated").mkdir(parents=True, exist_ok=True)

    if boxes is not None:
        missing = [p.name for p in paths if p.name not in boxes]
        if missing:
            raise ValueError(f"Нет записи о рамках для: {', '.join(missing[:5])}")

    def one(p: Path):
        with Image.open(p) as image:
            img = image if image.mode == "RGB" else image.convert("RGB")
            img.load()
            if boxes is None:
                pred = model.predict(img)
            else:
                supplied = boxes[p.name]
                if any(o["bbox"][2] > img.width or o["bbox"][3] > img.height for o in supplied):
                    raise ValueError(f"Рамка за пределами фото: {p.name}")
                pred = model.predict(img, boxes=supplied)
            pred = enrich_result(pred, meta=meta.get(keys[p]) or meta.get(p.name) or {}, registry=registry)
            if out:
                target = (out / "annotated" / keys[p]).with_suffix(".jpg")
                target.parent.mkdir(parents=True, exist_ok=True)
                annotate(img, pred["objects"], names=names, frame=pred).save(target, quality=88)
        return p, pred

    rows, preds = [], {}
    with ThreadPoolExecutor(workers) as ex:
        for k, (p, pred) in enumerate(ex.map(one, paths), 1):
            key = keys[p]
            preds[key] = pred
            rows += object_rows(key, pred, meta.get(key) or meta.get(p.name) or {}, names)
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
    ap.add_argument("--boxes", type=Path, help="рамки организатора или preds.json предыдущего запуска det_dino")
    ap.add_argument("--registry", type=Path, help="CSV/XLSX/JSONL реестр: asset_id, lat, lon, address")
    args = ap.parse_args()
    cfg = load_config()
    names = names_ru(cfg)
    if bool(args.boxes) != (cfg["variants"][args.variant].get("kind") == "boxes_dino"):
        ap.error("--boxes используется вместе с boxes_dino")
    boxes = read_boxes(args.boxes) if args.boxes else None

    src = Path(args.images)
    if src.is_dir():
        paths = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    else:
        paths = [src]
    paths = paths[:args.limit] if args.limit else paths
    keys = photo_keys(paths)
    meta = index_meta(read_meta(Path(args.meta)), list(keys.values())) if args.meta else {}
    registry = load_registry(read_meta(args.registry)) if args.registry else None
    model = variants.make(args.variant, cfg)
    out = Path(args.out)

    def progress(k, n, p, pred):
        bad = sum(o["status"] == "bad" for o in pred["objects"])
        flag = f"ERR {pred['error'][:80]}" if pred["error"] else f"{len(pred['objects'])} obj, bad {bad}"
        print(f"  {k}/{n} {p.name}: {flag} {pred['seconds']}s", file=sys.stderr)

    rows, preds = run(model, paths, meta, names, out, model.workers, progress, boxes, registry)
    write_csv(out / "objects.csv", rows)
    repair = [r for r in rows if r["status"] == "bad"]
    write_csv(out / "repair.csv", repair)
    write_csv(out / "review.csv", [r for r in rows if r["status"] == "review"])
    (out / "preds.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1))
    errors = sum(bool(p["error"]) for p in preds.values())
    cost = sum(p.get("cost", 0) for p in preds.values())
    print(f"\n{len(paths)} фото, объектов {len(rows)}, на ремонт {len(repair)}, ошибок API {errors}, ${cost:.3f}")
    print(f"-> {out}/repair.csv, objects.csv, annotated/")


if __name__ == "__main__":
    main()
