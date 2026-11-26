"""Inference on new photos from the checkpoint saved by final.py."""
import hashlib
import time

import joblib
import numpy as np

from embed import ROOT
from final import Encoder, Tail, pick_device, lr_probs, tail_probs
from folders import split_label


class PhotoDino:
    workers = 1

    def __init__(self, cfg):
        path = ROOT / cfg["checkpoint"]
        with path.open("rb") as checkpoint:
            self.signature = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        self.state = state = joblib.load(path)
        device = pick_device(cfg.get("device", "auto"))
        self.encoder = Encoder(device, state["size"], state["k"])
        self.tail = None
        if state["tail"] is not None:
            self.tail = Tail(self.encoder.model, self.encoder.cut, len(state["classes"])).to(device).eval()
            self.tail.load_state_dict(state["tail"])
        self.vlm = None
        if state["qwen_weight"] and cfg.get("qwen"):
            from run import load_config
            from vlm import VLM
            config = load_config()
            self.vlm = VLM(config["variants"][cfg["qwen"]], config["taxonomy"])

    def fingerprint(self):
        return f"photo-dino-{self.signature}"

    def predict(self, img):
        start = time.monotonic()
        state = self.state
        classes = state["classes"]
        h = self.encoder.frozen([img.convert("RGB")])
        scores = []
        sub_opinions = []
        if state["logreg"] is not None:
            p_lr = lr_probs(state["logreg"], self.encoder.feats(h), len(classes))[0]
            scores.append(p_lr)
            idx_lr = int(p_lr.argmax())
            sub_opinions.append({"model": "DINOv2 LogReg", "label": classes[idx_lr], "confidence": round(float(p_lr[idx_lr]), 3)})
        if self.tail is not None:
            p_tail = tail_probs(self.tail, h)[0]
            scores.append(p_tail)
            idx_tail = int(p_tail.argmax())
            sub_opinions.append({"model": "DINOv2 Fine-tuned", "label": classes[idx_tail], "confidence": round(float(p_tail[idx_tail]), 3)})
        p = np.mean(scores, axis=0)
        answer = {}
        if self.vlm:
            from vlm_classify import classify
            answer = classify(self.vlm, img, classes)
            if answer.get("label") in classes:
                p[classes.index(answer["label"])] += state["qwen_weight"]
                sub_opinions.append({"model": "Qwen VLM", "label": answer["label"], "description": answer.get("description", "")})
        label = classes[int(p.argmax())]
        typ, defect = split_label(label, None)
        status = "ok" if state["ok"][label] else "bad"
        return {"objects": [{"type": typ, "bbox": None, "status": status,
                             "damage": defect if status == "bad" else None, "description": answer.get("description", "")}],
                "sub_opinions": sub_opinions,
                "seconds": time.monotonic() - start, "cost": answer.get("cost") or 0,
                "bad_json": answer.get("bad_json", False), "error": answer.get("error")}
