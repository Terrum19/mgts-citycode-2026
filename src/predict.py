"""Система целиком: папка фото (+ таблица с ID/координатами) -> рамки, описания, очередь triage.

    python src/predict.py --variant qwen235 --images photos/ --out out/run1
    python src/predict.py --variant qwen235 --images photos/ --meta photos.csv --out out/run1
    python src/predict.py --variant boxes_vlm --images photos/ --boxes boxes.json --out out/run1

Выход:
  annotated/<фото>.jpg   рамки: зелёная — исправен, красная — повреждён, жёлтая — review
  objects.csv            только object-level результаты
  frames.csv             кадровый результат (включая выведенный из объектов)
  repair.csv             только подтверждённые bad
  review.csv             только review
  triage.csv             все bad + review (обязательная очередь реагирования)
  preds.json             сырые ответы
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
from atomic import write_json_atomic  # noqa: E402
from config_utils import resolve_variant  # noqa: E402
from draw import annotate  # noqa: E402
from export_rows import export_rows  # noqa: E402
from results import make_frame, make_result, valid_bbox  # noqa: E402
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
    import pandas as pd
    from folders import match_table

    return match_table(pd.DataFrame(rows), files)


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["file"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel откроет кириллицу
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def read_boxes(path):
    """JSON с записями {file, objects: [{bbox: [x1,y1,x2,y2], type?, object_id?}]}."""
    boxes = {}
    for row in json.loads(path.read_text(encoding="utf-8-sig")):
        name = row["file"]
        if name in boxes:
            raise ValueError(f"Повтор фото в файле рамок: {name}")
        for obj in row["objects"]:
            box = valid_bbox(obj.get("bbox"))
            if box is None:
                raise ValueError(f"Некорректная рамка: {name} {obj.get('bbox')!r}")
            if box[0] < 0 or box[1] < 0:
                raise ValueError(f"Отрицательная координата рамки: {name}")
        boxes[name] = row["objects"]
    return boxes


def run(model, paths: list[Path], meta: dict, names: dict, out: Path | None = None, workers: int = 2,
        on_done=None, boxes=None, quality_cfg: dict | None = None) -> tuple[list[dict], dict]:
    """Общая функция для CLI и интерфейса -> (строки объектов, сырые предсказания)."""
    if out:
        (out / "annotated").mkdir(parents=True, exist_ok=True)

    if boxes is not None:
        missing = [p.name for p in paths if p.name not in boxes]
        if missing:
            raise ValueError(f"Нет записи о рамках для: {', '.join(missing[:5])}")

    def one(p: Path):
        with Image.open(p) as image:
            img = image.convert("RGB")
        if boxes is None and quality_cfg and quality_cfg.get("enabled"):
            from quality import frame_quality
            q = frame_quality(img, quality_cfg)
            if not q["passed"]:
                pred = make_result(frame=make_frame("review", review_reasons=q["review_reasons"],
                                                     source="quality_gate"), objects=[], scope="frame",
                                   quality=q["metrics"])
                if out:
                    annotate(img, [], names=names).save(out / "annotated" / (p.stem + ".jpg"), quality=88)
                return p, pred
        if boxes is None:
            pred = model.predict(img)
        else:
            supplied = boxes[p.name]
            for o in supplied:
                box = valid_bbox(o["bbox"])
                if box is None:
                    raise ValueError(f"Некорректная рамка: {p.name} {o.get('bbox')!r}")
                if box[2] > img.width or box[3] > img.height:
                    raise ValueError(f"Рамка за пределами фото: {p.name}")
            pred = model.predict(img, boxes=supplied)
        if out:
            annotate(img, pred.get("objects") or [], names=names).save(out / "annotated" / (p.stem + ".jpg"),
                                                                      quality=88)
        return p, pred

    preds = {}
    with ThreadPoolExecutor(workers) as ex:
        for k, (p, pred) in enumerate(ex.map(one, paths), 1):
            preds[p.name] = pred
            if on_done:
                on_done(k, len(paths), p, pred)
    rows = export_rows(preds, meta, names)["objects"]
    return rows, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--images", required=True, help="папка с фото или одно фото")
    ap.add_argument("--meta", help="CSV/XLSX/JSONL с ID и координатами фото")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--boxes", type=Path, help="JSON с рамками для boxes_dino / boxes_vlm")
    args = ap.parse_args()
    cfg = load_config()
    names = names_ru(cfg)
    vcfg = resolve_variant(cfg, args.variant)
    boxes_kind = vcfg.get("kind") in ("boxes_dino", "boxes_vlm")
    if bool(args.boxes) != boxes_kind:
        ap.error("--boxes используется только с boxes_dino / boxes_vlm (и обязателен для них)")
    boxes = read_boxes(args.boxes) if args.boxes else None

    src = Path(args.images)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    paths = paths[:args.limit] if args.limit else paths
    meta = index_meta(read_meta(Path(args.meta)), [p.name for p in paths]) if args.meta else {}
    model = variants.make(args.variant, cfg)
    out = Path(args.out)

    def progress(k, n, p, pred):
        bad = sum(o.get("status") == "bad" for o in (pred.get("objects") or []))
        review = sum(o.get("status") == "review" for o in (pred.get("objects") or []))
        frame = (pred.get("frame") or {}).get("status", "")
        flag = f"ERR {pred['error'][:80]}" if pred.get("error") else (
            f"{len(pred.get('objects') or [])} obj, bad {bad}, review {review}, frame {frame}")
        print(f"  {k}/{n} {p.name}: {flag} {pred.get('seconds', 0)}s", file=sys.stderr)

    rows, preds = run(model, paths, meta, names, out, model.workers, progress, boxes,
                      cfg.get("quality_gate"))
    tables = export_rows(preds, meta, names)
    write_csv(out / "objects.csv", tables["objects"])
    write_csv(out / "frames.csv", tables["frames"])
    write_csv(out / "repair.csv", tables["repair"])
    write_csv(out / "review.csv", tables["review"])
    write_csv(out / "triage.csv", tables["triage"])
    write_json_atomic(out / "preds.json", preds)
    errors = sum(bool(p.get("error")) for p in preds.values())
    cost = sum(p.get("cost", 0) for p in preds.values())
    print(f"\n{len(paths)} фото, объектов {len(tables['objects'])}, на ремонт {len(tables['repair'])}, "
          f"review {len(tables['review'])}, triage {len(tables['triage'])}, ошибок API {errors}, ${cost:.3f}")
    print(f"-> {out}/triage.csv, repair.csv, frames.csv, objects.csv, annotated/")


if __name__ == "__main__":
    main()
