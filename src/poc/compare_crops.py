"""Compare crop policies and cached VLMs against identical annotations."""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crop_dino import CropDino
from evaluate import evaluate, load_labels
from run import load_config
from vlm import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("detections", type=Path, help="preds.json от det_dino с полем detections")
    parser.add_argument("--baseline", type=Path, action="append", default=[])
    parser.add_argument("--paired-only", action="store_true", help="только кадры, покрытые всеми кэшами")
    args = parser.parse_args()

    cfg = load_config()
    ds = cfg["datasets"][args.dataset]
    labels = load_labels(ROOT / ds["labels"], ds.get("split"))
    cached = json.loads(args.detections.read_text())
    baselines = {p.parent.name: json.loads(p.read_text()) for p in args.baseline}
    available = cached.keys() & {r["file"] for r in labels}
    for baseline in baselines.values():
        available &= baseline.keys()
    missing = {r["file"] for r in labels if "bad_frame" not in r.get("frame_flags", [])} - available
    if missing and not args.paired_only:
        parser.error(f"В кэшах отсутствуют {len(missing)} кадров. Для сравнения на пересечении используйте --paired-only.")
    if missing:
        print(f"Сравнение на пересечении. Исключено кадров: {len(missing)}", file=sys.stderr)
        labels = [r for r in labels if r["file"] in available]
    if not any("bad_frame" not in r.get("frame_flags", []) for r in labels):
        parser.error("Нет кадров для оценки")
    keep = set(ds.get("types", ()))
    if keep:
        labels = [{**r, "objects": [o for o in r["objects"] if o["type"] in keep]} for r in labels]
    preds = {"manual": {}, "joint": {}, "detector": {}}
    model = CropDino(cfg["variants"]["boxes_dino"], None)
    images = ROOT / ds["images"]
    for row in labels:
        name = row["file"]
        if "bad_frame" in row.get("frame_flags", []):
            continue
        with Image.open(images / name) as img:
            boxes = cached[name]["detections"]
            for variant in preds:
                model.policy = variant if variant != "manual" else "detector"
                proposals = [o for o in row["objects"] if o["type"] != "other"] if variant == "manual" else boxes
                result = model.predict(img, boxes=proposals)
                if variant != "manual":
                    result["seconds"] += cached[name].get("det_seconds", 0)
                preds[variant][name] = result
    preds.update(baselines)
    metrics = {}
    for variant, result in preds.items():
        if keep:
            result = {name: {**p, "objects": [o for o in p["objects"] if o["type"] in keep]}
                      for name, p in result.items()}
        metrics[variant] = evaluate(labels, result, ds.get("iou"))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
