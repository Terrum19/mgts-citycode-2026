"""Inference on new photos from the checkpoint saved by final.py.

Два режима (ключ `cascade` в варианте):
  * cascade = false — прежний: среднее DINO-голов, плюс ответ Qwen как добавка с весом
    `qwen_weight` (VLM вызывается на КАЖДЫЙ кадр);
  * cascade = true — быстрый DINO решает по суммарной P(bad), уверенности статуса и margin;
    VLM вызывается только на спорные кадры (и на описание подтверждённых bad при
    `describe_bad = true`). Нет ключа/сети — спорный кадр уходит в status="review", а
    подтверждённый bad не теряется из-за необязательного описания.
"""
import hashlib
import json
import time

import joblib
import numpy as np

from cascade import decide, vlm_ready
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
        self.cfg = cfg
        self.cascade = bool(cfg.get("cascade", False))
        device = pick_device(cfg.get("device", "auto"))
        self.encoder = Encoder(device, state["size"], state["k"])
        self.tail = None
        if state["tail"] is not None:
            self.tail = Tail(self.encoder.model, self.encoder.cut, len(state["classes"])).to(device).eval()
            self.tail.load_state_dict(state["tail"])
        self.vlm = None
        # в каскаде VLM нужна и при qwen_weight = 0: она разрешает спорные кадры
        if cfg.get("qwen") and (self.cascade or state["qwen_weight"]):
            from run import load_config
            from vlm import VLM
            config = load_config()
            self.vlm = VLM(config["variants"][cfg["qwen"]], config["taxonomy"])
        self.vlm_ok, self.vlm_reason = vlm_ready(self.vlm)

    def fingerprint(self):
        """Меняется при смене режима/порогов/VLM: старые предсказания в кэше не переиспользуются."""
        relevant = {key: self.cfg.get(key) for key in
                    ("cascade", "qwen", "p_bad_min", "p_bad_max_ok", "status_conf_min",
                     "margin_min", "describe_bad", "device")}
        tag = json.dumps(relevant, sort_keys=True)
        if self.vlm is not None:
            tag += self.vlm.fingerprint()
        return f"photo-dino-{self.signature}-{hashlib.sha1(tag.encode()).hexdigest()[:8]}"

    def predict(self, img):
        start = time.monotonic()
        state = self.state
        h = self.encoder.frozen([img.convert("RGB")])
        scores = []
        if state["logreg"] is not None:
            scores.append(lr_probs(state["logreg"], self.encoder.feats(h), len(state["classes"]))[0])
        if self.tail is not None:
            scores.append(tail_probs(self.tail, h)[0])
        p = np.mean(scores, axis=0)
        return self._cascade(img, p, start) if self.cascade else self._legacy(img, p, start)

    def _legacy(self, img, p, start):
        state = self.state
        classes = state["classes"]
        answer = {}
        if self.vlm:
            from vlm_classify import classify
            answer = classify(self.vlm, img, classes)
            if answer.get("label") in classes:
                p[classes.index(answer["label"])] += state["qwen_weight"]
        label = classes[int(p.argmax())]
        typ, defect = split_label(label, None)
        status = "ok" if state["ok"][label] else "bad"
        return {"objects": [{"type": typ, "bbox": None, "status": status,
                             "damage": defect if status == "bad" else None,
                             "description": answer.get("description", "")}],
                "seconds": time.monotonic() - start, "cost": answer.get("cost") or 0,
                "bad_json": answer.get("bad_json", False), "error": answer.get("error")}

    def _cascade(self, img, p, start):
        state = self.state
        classes = state["classes"]
        order = np.argsort(-p)
        # как в Cascade.summarize: считаем внутри типа лучшего класса, иначе массы люков и опор
        # складывались бы в искусственно высокую P(bad)
        kinds = [split_label(cls, None)[0] for cls in classes]
        selected = [i for i, kind in enumerate(kinds) if kind == kinds[int(order[0])]]
        good = [i for i in selected if state["ok"][classes[i]]]
        broken = [i for i in selected if not state["ok"][classes[i]]]
        p_bad = float(p[broken].sum()) if broken else 0.0
        p_ok = float(p[good].sum()) if good else 0.0
        summary = {"p_bad": p_bad, "p_ok": p_ok, "status_confidence": max(p_bad, p_ok),
                   "margin": abs(p_bad - p_ok), "ready": bool(broken and good)}
        decision = decide(summary, self.cfg)
        hint = [[classes[int(j)], float(p[j])] for j in order[:3]]  # топ-3 как подсказка VLM
        answer = {"label": None, "description": "", "cost": 0.0, "bad_json": False, "error": None}
        describe = decision == "bad" and bool(self.cfg.get("describe_bad"))
        if self.vlm_ok and (decision == "review" or describe):
            try:
                from vlm_classify import classify
                answer = classify(self.vlm, img, classes, hint=hint)
            except Exception as e:  # noqa: BLE001 — сбой сети/ключа не должен ронять кадр
                answer = {**answer, "error": str(e)[:300]}
        if decision == "review":
            label = answer.get("label") if answer.get("label") in classes else classes[int(order[0])]
            status = ("ok" if state["ok"][label] else "bad") if answer.get("label") in classes else "review"
        elif decision == "bad":
            label, status = classes[max(broken, key=lambda i: float(p[i]))], "bad"
        else:
            label, status = classes[max(good, key=lambda i: float(p[i]))], "ok"
        typ, defect = split_label(label, None)
        obj = {"type": typ, "bbox": None, "status": status,
               "damage": defect if status == "bad" else None,
               "description": answer.get("description") or ""}
        if status == "review" and self.vlm_reason:
            obj["vlm_error"] = self.vlm_reason
        return {"objects": [obj], "seconds": time.monotonic() - start, "cost": answer.get("cost") or 0,
                "bad_json": answer.get("bad_json", False), "error": answer.get("error")}
