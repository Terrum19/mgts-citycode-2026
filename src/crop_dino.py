"""Train a DINOv2 crop classifier on object annotations; infer with Mask2Former proposals."""
import argparse
import hashlib
import json
import time
import tomllib
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from classify import make_clf
from detector import crop_box
from embed import Embedder, ROOT


def features(encoder, crops):
    with encoder.torch.inference_mode():
        values = encoder._forward(crops)
        return encoder.torch.nn.functional.normalize(values, dim=1).cpu().numpy()


def crop(img, obj, cfg):
    box = crop_box(obj["bbox"], *img.size, cfg["pad"], cfg["crop_min_side"])
    return img.crop(box)


def train(cfg, annotations, images, split, output):
    encoder = Embedder(str(ROOT / cfg["encoder"]), device=cfg.get("device", "cpu"), size=cfg["size"])
    vectors, targets, batch = [], [], []
    for row in json.loads(annotations.read_text()):
        if split and row.get("split") != split or "bad_frame" in row.get("frame_flags", []):
            continue
        with Image.open(images / row["file"]) as img:
            for obj in row["objects"]:
                if obj["status"] not in ("bad", "ok") or obj["type"] == "other":
                    continue
                batch.append(crop(img, obj, cfg).convert("RGB"))
                targets.append(json.dumps([obj["type"], obj["status"], obj.get("damage")], ensure_ascii=False))
                if len(batch) == 16:
                    vectors.extend(features(encoder, batch))
                    batch.clear()
    if batch:
        vectors.extend(features(encoder, batch))
    if len(set(targets)) < 2:
        raise ValueError("Для обучения нужны хотя бы два класса объектов")
    clf = make_clf(32).fit(np.asarray(vectors), targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "support": dict(Counter(targets)), "config": {k: cfg[k] for k in ("encoder","size","pad","crop_min_side")}}, output)
    print(dict(Counter(targets)))


class CropDino:
    workers = 1

    def __init__(self, cfg, detector_cfg):
        from detector import Detector
        path = ROOT / cfg["checkpoint"]
        state = joblib.load(path)
        self.cfg, self.clf = state["config"], state["clf"]
        self.classes = [json.loads(label) for label in self.clf.classes_]
        self.support = state.get("support")
        self.policy = cfg.get("class_policy", "detector")
        if self.policy not in ("joint", "detector"):
            raise ValueError("class_policy должен быть joint или detector")
        self.threshold = cfg.get("min_confidence", 0.55)
        self.encoder = Embedder(str(ROOT / self.cfg["encoder"]),
                                device=(detector_cfg or cfg).get("device", "cpu"), size=self.cfg["size"])
        self.det = Detector(detector_cfg) if detector_cfg else None
        self.signature = hashlib.sha256(path.read_bytes() + json.dumps(
            [cfg, detector_cfg], sort_keys=True).encode()).hexdigest()

    def fingerprint(self):
        return "crop-dino-v2-" + self.signature

    def classify(self, img, found):
        objects = []
        for offset in range(0, len(found), 16):
            group = found[offset:offset + 16]
            x = features(self.encoder, [crop(img, obj, self.cfg) for obj in group])
            for obj, scores in zip(group, self.clf.predict_proba(x)):
                typ = obj.get("type")
                eligible = [i for i, c in enumerate(self.classes)
                            if self.policy == "joint" or not typ or c[0] == typ]
                result = {**obj, "type": typ or "other", "status": "review", "damage": None, "description": ""}
                if eligible:
                    index = max(eligible, key=lambda i: scores[i])
                    kind, status, damage = self.classes[index]
                    supported = {c[1] for i, c in enumerate(self.classes)
                                 if c[0] == kind and (self.support is None or
                                                      self.support[self.clf.classes_[i]] >= 2)}
                    if scores[index] >= self.threshold and supported == {"ok", "bad"} and (
                            self.support is None or self.support[self.clf.classes_[index]] >= 2):
                        result.update(type=kind, status=status, damage=damage)
                objects.append(result)
        return objects

    def predict(self, img, boxes=None):
        start = time.monotonic()
        det_seconds = 0.0
        if boxes is None:
            if self.det is None:
                raise ValueError("Передайте файл рамок через --boxes")
            boxes = self.det.detect(img)
            det_seconds = time.monotonic() - start
        return {"objects": self.classify(img, boxes), "detections": boxes, "det_seconds": det_seconds,
                "seconds": time.monotonic() - start, "cost": 0., "bad_json": False, "error": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--variant", default="det_dino")
    args = parser.parse_args()
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["variants"][args.variant]
    train(cfg, args.annotations, args.images, args.split, ROOT / cfg["checkpoint"])
