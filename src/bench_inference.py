"""Time actual unlabelled photos: DINO features and detector + crop features.

No trained classifier, accuracy, automation or end-to-end product claims.
python src/bench_inference.py --device cpu --limit 8 --repeats 3
"""
import argparse
import json
import platform
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image

from detector import Detector, crop_box
from embed import Embedder, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT / "out/effect/benchmark.json")
    a = p.parse_args()
    paths = sorted((ROOT / "data/raw/panoramas").glob("*.jpg"))[:a.limit]
    if not paths or a.repeats < 1:
        p.error("need photos and positive repeats")
    result = {"platform": platform.platform(), "device": a.device,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "cpu_threads": torch.get_num_threads(), "photos": [x.name for x in paths],
              "repeats": a.repeats,
              "scope": "real photos; no trained classifier, VLM, UI, upload or export",
              "stages": {}}

    def sync():
        if a.device == "mps":
            torch.mps.synchronize()
        elif a.device == "cuda":
            torch.cuda.synchronize()

    def save():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    def features(encoder, images):
        with torch.inference_mode():
            return torch.nn.functional.normalize(encoder._forward(images), dim=1).cpu().numpy()

    start = time.perf_counter()
    enc = Embedder(str(ROOT / "weights/dinov2-base"), device=a.device, size=224)
    sync()
    result["dino_load_seconds"] = time.perf_counter() - start
    with Image.open(paths[0]) as im:
        features(enc, [im.convert("RGB")])

    def measure(name, fn):
        rows = []
        sync()
        wall = time.perf_counter()
        for repeat in range(a.repeats):
            for path in paths:
                sync()
                start = time.perf_counter()
                with Image.open(path) as im:
                    objects = fn(im.convert("RGB"))
                sync()
                seconds = time.perf_counter() - start
                rows.append({"file": path.name, "repeat": repeat,
                             "seconds": seconds, "objects": objects})
                print(name, repeat, path.name, round(seconds, 3), objects, flush=True)
        wall = time.perf_counter() - wall
        ts = [r["seconds"] for r in rows]
        result["stages"][name] = {"rows": rows, "wall_seconds": wall,
                                  "photos_per_second": len(rows) / wall,
                                  "mean_seconds": float(np.mean(ts)),
                                  "p50_seconds": float(np.percentile(ts, 50)),
                                  "p95_seconds": float(np.percentile(ts, 95))}
        save()

    def full_photo(im):
        features(enc, [im])
        return 1

    measure("photo_features_224_batch1", full_photo)
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["detector"]
    cfg = {**cfg, "device": a.device, "model": str(ROOT / "weights/mask2former-mapillary")}
    start = time.perf_counter()
    det = Detector(cfg)
    sync()
    result["detector_load_seconds"] = time.perf_counter() - start
    result["detector_config"] = cfg

    def det_features(im):
        boxes = det.detect(im)
        crops = [im.crop(crop_box(b["bbox"], *im.size, 0.2, 96)) for b in boxes]
        for offset in range(0, len(crops), 16):
            features(enc, crops[offset:offset + 16])
        return len(boxes)

    with Image.open(paths[0]) as im:
        det_features(im.convert("RGB"))
    measure("detector_plus_crop_features", det_features)
    print("Saved", a.out, flush=True)


if __name__ == "__main__":
    main()
