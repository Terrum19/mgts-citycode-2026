"""Inference on new photos from the checkpoint saved by final.py.

Результат — кадровый (frame), а не псевдообъект с bbox=None: у PhotoDino нет локализации,
и выдавать рамку на весь кадр было бы фабрикацией. Решение принимает общая policy
(`src/policy.py`): legacy-argmax или каскад; VLM только на спорные кадры и description для
подтверждённых bad. Ошибка VLM не понижает подтверждённый bad.
"""
import hashlib
import json
import time
import warnings

import joblib
import numpy as np

from cascade import vlm_ready
from config_utils import check_offline_vlm, is_offline, resolve_variant
from embed import ROOT
from final import Encoder, Tail, pick_device, lr_probs, tail_probs
from folders import split_label
from policy import effective_policy, finalize, initial
from results import make_result
from taxonomy import validate_checkpoint


class PhotoDino:
    workers = 1

    def __init__(self, cfg, root_cfg=None):
        path = ROOT / cfg["checkpoint"]
        with path.open("rb") as checkpoint:
            self.signature = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        self.state = state = joblib.load(path)
        self.cfg = dict(cfg)
        self.tax = (root_cfg or {}).get("taxonomy") or {}
        self.offline = is_offline(cfg, root_cfg)
        legacy = "error" if self.offline or cfg.get("strict_checkpoint") else cfg.get("legacy_checkpoint", "warn")
        validate_checkpoint(state, self.tax, legacy=legacy)
        self.classes = list(state["classes"])
        self.triples = []
        for label in self.classes:
            typ, defect = split_label(label, None)
            ok = bool(state["ok"][label])
            self.triples.append([typ, "ok" if ok else "bad", defect or None])
        self.policy = effective_policy(state.get("policy"), cfg)
        self.cascade = bool(self.policy.get("cascade", cfg.get("cascade", False)))
        self.policy["cascade"] = self.cascade
        self.policy["mode"] = "cascade" if self.cascade else "legacy"
        self.hint_mode = str(self.policy.get("hint_mode", "none"))
        if self.hint_mode != "none":
            raise ValueError("PhotoDino поддерживает только hint_mode='none'")

        saved_qwen = state.get("qwen_variant", None)
        wanted_qwen = str(cfg.get("qwen") or "")
        if saved_qwen is not None and saved_qwen != wanted_qwen:
            if saved_qwen == "":
                raise ValueError("Checkpoint обучен offline (без VLM), а вариант требует qwen; "
                                 "выберите photo_dino_offline или переобучите checkpoint.")
            raise ValueError(f"Checkpoint привязан к qwen={saved_qwen!r}, а вариант просит {wanted_qwen!r}. "
                             "Переобучите checkpoint или задайте тот же qwen.")
        if saved_qwen is None and wanted_qwen:
            warnings.warn("Legacy checkpoint без qwen_variant: связка с VLM не проверена", RuntimeWarning,
                          stacklevel=2)
        self.qwen_variant = wanted_qwen
        self.vlm = None
        # в каскаде VLM нужна и при qwen_weight = 0: она разрешает спорные кадры
        if wanted_qwen and (self.cascade or float(self.policy.get("qwen_weight", 0.0) or 0.0) > 0):
            check_offline_vlm(root_cfg or {}, self.policy, self.offline)
            from vlm import VLM
            self.vlm = VLM(resolve_variant(root_cfg or self._root_cfg(), wanted_qwen), self.tax)
        self.vlm_ok, self.vlm_reason = vlm_ready(self.vlm)

        device = pick_device(cfg.get("device", "auto"))
        self.encoder = Encoder(device, state["size"], state["k"], offline=self.offline)
        self.tail = None
        if state["tail"] is not None:
            self.tail = Tail(self.encoder.model, self.encoder.cut, len(state["classes"])).to(device).eval()
            self.tail.load_state_dict(state["tail"])
        self.device = device

    def _root_cfg(self):
        from run import load_config
        return load_config()

    def fingerprint(self):
        """Меняется при смене checkpoint/политики/порогов/VLM: кэш не переиспользуется."""
        relevant = {
            "checkpoint": self.signature, "policy": self.policy, "qwen": self.qwen_variant,
            "hint_mode": self.hint_mode, "device": self.cfg.get("device"),
        }
        tag = json.dumps(relevant, sort_keys=True, default=str)
        if self.vlm is not None:
            tag += self.vlm.fingerprint()
        return f"photo-dino-v3-{hashlib.sha256(tag.encode()).hexdigest()[:12]}"

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
        mode = "cascade" if self.cascade else "legacy"
        decision = initial(self.classes, self.triples, [float(x) for x in p], mode=mode,
                           cfg=self.policy, support=None, type_hint=None,
                           class_policy=str(self.policy.get("class_policy", "detector")))
        answer = {"label": None, "description": "", "cost": 0.0, "bad_json": False, "error": None,
                  "review": False}
        if decision.get("needs_vlm") and self.vlm_ok:
            try:
                from vlm_classify import classify
                answer = classify(self.vlm, img, self.classes, hint=None)
            except Exception as exc:  # noqa: BLE001 — сбой сети/ключа не должен ронять кадр
                answer = {**answer, "error": str(exc)[:300]}
        elif decision.get("needs_vlm") and self.vlm_reason:
            answer = {**answer, "error": self.vlm_reason}
        vlm_norm = {
            "available": self.vlm_ok, "label": answer.get("label"), "answer": None,
            "description": str(answer.get("description") or ""), "error": answer.get("error"),
            "bad_json": bool(answer.get("bad_json")), "review": bool(answer.get("review")),
        }
        out = finalize(decision, cfg=self.policy, tax=self.tax, vlm=vlm_norm)
        if out is None:  # PhotoDino не выбрасывает кадр: это не детекция с чужой рамкой
            out = {"status": "review", "type": None, "damage": None, "description": "",
                   "review_reasons": ["not_target"], "review": True}
        frame = {
            "status": out["status"], "type": out.get("type"), "damage": out.get("damage"),
            "description": out.get("description", ""), "review": out["status"] == "review",
            "review_reasons": out.get("review_reasons", []), "source": "native",
        }
        return make_result(frame=frame, objects=[], scope="frame",
                           seconds=time.monotonic() - start,
                           cost=float(answer.get("cost") or 0.0),
                           bad_json=bool(answer.get("bad_json")), error=answer.get("error"))
