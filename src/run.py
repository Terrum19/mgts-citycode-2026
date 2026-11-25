"""Сравнение вариантов: прогон на размеченных наборах + метрики + картинки худших ошибок.

    python src/run.py --variant qwen235 --data pano_dev
    python src/run.py --variant qwen235 nemotron_free det_qwen235 --data pano_dev damage_manholes
    python src/run.py --variant qwen235 --data pano_test         # test — только для зафиксированных версий!

Наборы и варианты описаны в config.toml ([datasets.*], [variants.*]).
Результат: runs/<набор>__<вариант>/ — preds.json, metrics.json, review/*.jpg (худшие кадры: разметка
жёлтым пунктиром, предсказание цветом). Ответы кэшируются: повторный запуск досчитывает только новые
кадры и кадры с ошибкой API; при смене промпта/модели/таксономии всё считается заново.
"""
import argparse
import json
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants  # noqa: E402
from draw import annotate  # noqa: E402
from evaluate import evaluate, load_labels  # noqa: E402
from vlm import ROOT  # noqa: E402

ROWS = ["repair_tp", "repair_fp", "repair_fn", "repair_precision", "repair_recall", "review_objects", "images", "gt_objects", "gt_bad_objects", "det_precision", "det_recall",
        "det_precision@0.5", "det_recall@0.5", "det_precision@0.3", "det_recall@0.3",
        "det_recall_anytype", "img_bad_recall", "img_bad_precision", "obj_bad_recall",
        "obj_bad_precision", "damage_acc", "bad_json_rate", "error_rate", "sec_per_image", "cost_usd"]


def load_config() -> dict:
    return tomllib.loads((ROOT / "config.toml").read_text())


def run_one(vname: str, dname: str, cfg: dict, review: int) -> dict:
    ds = cfg["datasets"][dname]
    labels = load_labels(ROOT / ds["labels"], ds.get("split"))
    images = ROOT / ds["images"]
    model = variants.make(vname, cfg)
    fp = model.fingerprint()
    out = ROOT / "runs" / f"{dname}__{vname}"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "preds.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    preds = {f: p for f, p in cache.items() if p.get("fingerprint") == fp and not p.get("error")}
    todo = [lab for lab in labels if lab["file"] not in preds]
    print(f"[{dname} / {vname}] {len(labels)} кадров, из кэша {len(labels) - len(todo)}, считаем {len(todo)}",
          file=sys.stderr)

    def one(lab):
        p = model.predict(Image.open(images / lab["file"]))
        p["fingerprint"] = fp
        return lab["file"], p

    with ThreadPoolExecutor(model.workers) as ex:
        for k, (f, p) in enumerate(ex.map(one, todo), 1):
            preds[f] = p
            flag = "ERR " + p["error"][:80] if p["error"] else ("BADJSON" if p["bad_json"] else f"{len(p['objects'])} obj")
            print(f"  {k}/{len(todo)} {f}: {flag} {p['seconds']}s", file=sys.stderr)
            if k % 10 == 0:  # промежуточное сохранение: прерванный прогон не теряется
                cache_path.write_text(json.dumps(preds, ensure_ascii=False, indent=1))
    cache_path.write_text(json.dumps(preds, ensure_ascii=False, indent=1))

    if ds.get("types"):  # набор размечен не по всем типам: остальные не оцениваем
        keep = set(ds["types"])
        labels = [{**lab, "objects": [o for o in lab["objects"] if o["type"] in keep]} for lab in labels]
        preds = {f: {**p, "objects": [o for o in p["objects"] if o["type"] in keep]} for f, p in preds.items()}
    m = evaluate(labels, preds, ds.get("iou"))
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    if review:
        rdir = out / "review"
        rdir.mkdir(exist_ok=True)
        for old in rdir.glob("*.jpg"):
            old.unlink()
        by_file = {lab["file"]: lab for lab in labels}
        for k, w in enumerate(m["worst"][:review], 1):
            lab = by_file[w["file"]]
            im = annotate(Image.open(images / w["file"]), preds[w["file"]]["objects"], gt=lab["objects"])
            im.save(rdir / f"{k:02d}_{Path(w['file']).stem}.jpg", quality=85)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--review", type=int, default=12, help="сколько худших кадров нарисовать")
    args = ap.parse_args()
    cfg = load_config()

    for dname in args.data:
        res = {v: run_one(v, dname, cfg, args.review) for v in args.variant}
        print(f"\n=== {dname}")
        print("Правило сопоставления:", next(iter(res.values()))["matching"])
        print("metric".ljust(20) + "".join(v[:15].rjust(16) for v in res))
        for k in ROWS:
            print(k.ljust(20) + "".join((f"{r[k]:16.3f}" if isinstance(r[k], float) else f"{r[k]:16d}") for r in res.values()))
        for v, r in res.items():
            print(f"\n{v}: по типам " + json.dumps(r["per_type"], ensure_ascii=False))
            if r["type_confusion"]:
                print(f"{v}: путает типы " + json.dumps(r["type_confusion"], ensure_ascii=False))
            print(f"{v}: повреждения (разметка -> модель) " + json.dumps(r["damage_confusion"], ensure_ascii=False))
        print(f"\nкартинки худших ошибок: runs/{dname}__<вариант>/review/")


if __name__ == "__main__":
    main()
