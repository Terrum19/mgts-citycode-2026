"""Train a DINOv2 crop classifier on object annotations; infer with Mask2Former proposals.

Обучение (train): к каждому объекту добавляются погодные копии (`augment.py`) — только train,
после фильтра по split. `support` в checkpoint считается по ОРИГИНАЛАМ: копии не должны
«подтверждать» редкий класс самому себе.

Инференс: по умолчанию прежняя логика (максимум одного кортежа >= min_confidence). При
`cascade = true` в варианте решение принимается по суммарной P(bad)/P(ok), уверенности статуса
и отрыву margin, а спорные объекты уходят в VLM (`is_object`/status/damage/description) или,
если VLM недоступна, в status="review" — без падения на отсутствии ключа или сети.
"""
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

from augment import augment, n_copies, sample_seed
from cascade import answer_error, classify_crop_safe, decide, merge_vlm, pick, summarize, vlm_ready
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


def augment_params() -> dict:
    """Секция [augment] из config.toml: границы преобразований общие для всех вариантов."""
    path = ROOT / "config.toml"
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text()).get("augment", {})


def annotations_rows(annotations: Path, split: str | None) -> list[tuple[str, list[dict]]]:
    """Файлы и объекты, годные для обучения: split, не bad_frame, статус ok/bad, не other."""
    rows = []
    for row in json.loads(annotations.read_text()):
        if split and row.get("split") != split or "bad_frame" in row.get("frame_flags", []):
            continue
        objects = [o for o in row["objects"] if o["status"] in ("bad", "ok") and o["type"] != "other"]
        if objects:
            rows.append((row["file"], objects))
    return rows


def target_of(obj: dict) -> str:
    return json.dumps([obj["type"], obj["status"], obj.get("damage")], ensure_ascii=False)


def train(cfg, annotations, images, split, output):
    rows = annotations_rows(annotations, split)
    origins = [target_of(obj) for _, objects in rows for obj in objects]
    if len(set(origins)) < 2:
        raise ValueError("Для обучения нужны хотя бы два класса объектов")
    support = Counter(origins)  # до копий: иначе редкий класс «подтверждает» себя сам
    encoder = Embedder(str(ROOT / cfg["encoder"]), size=cfg["size"])
    params = augment_params() if cfg.get("augment") else {}
    seed = int(cfg.get("augment_seed", 42))
    vectors, targets, batch = [], [], []

    def flush():
        if batch:
            vectors.extend(features(encoder, [item for item, _ in batch]))
            targets.extend(target for _, target in batch)
            batch.clear()

    for file, objects in rows:
        with Image.open(images / file) as img:
            for index, obj in enumerate(objects):
                target = target_of(obj)
                crop_img = crop(img, obj, cfg).convert("RGB")
                batch.append((crop_img, target))
                for copy in range(n_copies(obj["status"], support[target], cfg)):
                    batch.append((augment(crop_img, params, sample_seed(seed, f"{file}:{index}", copy)), target))
                if len(batch) >= 16:
                    flush()
    flush()
    clf = make_clf(32).fit(np.asarray(vectors), targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {"clf": clf, "support": dict(support),
             "config": {k: cfg[k] for k in ("encoder", "size", "pad", "crop_min_side")}}
    if cfg.get("augment"):
        state["augment"] = {"enabled": True,
                            "copies": int(cfg.get("augment_copies", 1)),
                            "rare_bad_copies": int(cfg.get("augment_rare_bad_copies", cfg.get("augment_copies", 1))),
                            "rare_threshold": int(cfg.get("augment_rare_threshold", 20))}
    joblib.dump(state, output)
    print(dict(support))
    if cfg.get("augment"):
        print(f"аугментированных копий: {len(targets) - len(origins)}, всего примеров {len(targets)}")


class CropDino:
    workers = 1

    def __init__(self, cfg: dict, detector_cfg: dict | None, root_cfg: dict | None = None):
        from detector import Detector
        path = ROOT / cfg["checkpoint"]
        state = joblib.load(path)
        self.cfg, self.clf = state["config"], state["clf"]
        self.raw_labels = list(self.clf.classes_)
        self.classes = [json.loads(label) for label in self.raw_labels]
        self.support = state.get("support")
        self.policy = cfg.get("class_policy", "detector")
        if self.policy not in ("joint", "detector"):
            raise ValueError("class_policy должен быть joint или detector")
        self.threshold = cfg.get("min_confidence", 0.55)
        self.cascade = bool(cfg.get("cascade", False))
        self.variant = cfg
        self.tax = (root_cfg or {}).get("taxonomy") or {}
        self.vlm = self._load_vlm(cfg, root_cfg)
        self.vlm_ok, self.vlm_reason = vlm_ready(self.vlm)
        self.encoder = Embedder(str(ROOT / self.cfg["encoder"]),
                                device=(detector_cfg or cfg).get("device", "cpu"), size=self.cfg["size"])
        self.det = Detector(detector_cfg) if detector_cfg else None
        tag = json.dumps([cfg, detector_cfg], sort_keys=True)
        if self.vlm is not None:
            tag += self.vlm.fingerprint()  # поведение зависит от промпта/модели: кэш обязан обновиться
        self.signature = hashlib.sha256(path.read_bytes() + tag.encode()).hexdigest()

    def _load_vlm(self, cfg: dict, root_cfg: dict | None):
        name = cfg.get("vlm")
        if not self.cascade or not name:
            return None
        variants_cfg = (root_cfg or {}).get("variants") or {}
        if name not in variants_cfg:
            raise ValueError(f"в варианте указан vlm={name!r}, но такой секции нет в [variants]")
        if not self.tax:
            raise ValueError("каскад требует секцию [taxonomy] в config.toml")
        from vlm import VLM
        return VLM(variants_cfg[name], self.tax)

    def fingerprint(self):
        return "crop-dino-v2-" + self.signature

    def classify(self, img, found, telemetry: dict | None = None):
        """-> [{type, bbox, status, damage, description}]; VLM-вызовы копят cost/error в telemetry."""
        objects = []
        for offset in range(0, len(found), 16):
            group = found[offset:offset + 16]
            crops = [crop(img, obj, self.cfg) for obj in group]
            x = features(self.encoder, crops)
            for obj, scores, crop_img in zip(group, self.clf.predict_proba(x), crops):
                result = self._legacy(obj, scores) if not self.cascade \
                    else self._cascade(obj, scores, crop_img, telemetry)
                if result is not None:  # None — VLM сказала «это не наш объект»
                    objects.append(result)
        return objects

    def _legacy(self, obj: dict, scores) -> dict:
        typ = obj.get("type")
        eligible = [i for i, c in enumerate(self.classes)
                    if self.policy == "joint" or not typ or c[0] == typ]
        result = {**obj, "type": typ or "other", "status": "review", "damage": None, "description": ""}
        if eligible:
            index = max(eligible, key=lambda i: scores[i])
            kind, status, damage = self.classes[index]
            supported = {c[1] for i, c in enumerate(self.classes)
                         if c[0] == kind and (self.support is None or
                                              self.support[self.raw_labels[i]] >= 2)}
            if scores[index] >= self.threshold and supported == {"ok", "bad"} and (
                    self.support is None or self.support[self.raw_labels[index]] >= 2):
                result.update(type=kind, status=status, damage=damage)
        return result

    def _cascade(self, obj: dict, scores, crop_img, telemetry: dict | None) -> dict | None:
        typ = obj.get("type")
        summary = summarize(self.classes, scores, labels=self.raw_labels, support=self.support,
                            type_hint=typ, policy=self.policy,
                            min_support=int(self.variant.get("min_support", 2)))
        decision = decide(summary, self.variant)
        result = {**obj, "type": typ, "status": decision, "damage": None, "description": ""}
        if decision in ("ok", "bad"):
            kind, damage = pick(summary, self.classes, decision)
            result.update(type=kind or typ, damage=damage if decision == "bad" else None)
        describe = decision == "bad" and bool(self.variant.get("describe_bad"))
        if self.vlm_ok and (decision == "review" or describe):
            call = classify_crop_safe(self.vlm, crop_img, typ or result.get("type"))
            self._account(telemetry, call)
            answer = call["answer"]
            reason = call["error"] or answer_error(answer)
            if answer is not None and reason is None:
                merged = merge_vlm(result, answer, self.tax,
                                   not_target=self.variant.get("not_target", "review"),
                                   authoritative=decision == "review")
                if merged is None:
                    return None
                result = merged
            elif reason:
                result["vlm_error"] = reason
                if telemetry is not None and not telemetry.get("error"):
                    telemetry["error"] = reason
        elif decision == "review" and self.vlm_reason:
            result["vlm_error"] = self.vlm_reason  # нет ключа — это не ошибка API, просто review
        if result["status"] == "review":
            result["damage"] = None
        if result.get("type") is None:
            result["type"] = "other"
        return result

    @staticmethod
    def _account(telemetry: dict | None, call: dict) -> None:
        if telemetry is None:
            return
        telemetry["cost"] = float(telemetry.get("cost") or 0.0) + float(call.get("cost") or 0.0)
        telemetry["bad_json"] = bool(telemetry.get("bad_json")) or bool(call.get("bad_json"))
        if call.get("error") and not telemetry.get("error"):
            telemetry["error"] = call["error"]

    def predict(self, img, boxes=None):
        start = time.monotonic()
        det_seconds = 0.0
        if boxes is None:
            if self.det is None:
                raise ValueError("Передайте файл рамок через --boxes")
            boxes = self.det.detect(img)
            det_seconds = time.monotonic() - start
        telemetry = {"cost": 0.0, "bad_json": False, "error": None}
        objects = self.classify(img, boxes, telemetry)
        return {"objects": objects, "detections": boxes, "det_seconds": det_seconds,
                "seconds": time.monotonic() - start, "cost": telemetry["cost"],
                "bad_json": telemetry["bad_json"], "error": telemetry["error"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--variant", default="det_dino")
    args = parser.parse_args()
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["variants"][args.variant]
    train(cfg, args.annotations, args.images, args.split, ROOT / cfg["checkpoint"])
