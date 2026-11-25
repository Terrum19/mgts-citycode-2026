"""Train a DINOv2 crop classifier on object annotations; infer with Mask2Former proposals.

Обучение (train): к каждому объекту добавляются погодные копии (`augment.py`) — только train,
после фильтра по split. `support` в checkpoint считается по ОРИГИНАЛАМ: копии не должны
«подтверждать» редкий класс самому себе.

Инференс идёт через общую policy (`src/policy.py`): legacy-порог или каскад с
P(bad)/P(ok), support, VLM. Спорный объект уходит в status="review"; если VLM недоступна,
это не ошибка API, а обязательная ручная проверка.
"""
import argparse
import json
import time
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from atomic import write_json_atomic  # noqa: F401 — публичный хелпер рядом
from augment import augment, n_copies, sample_seed
from cascade import vlm_ready
from classify import make_clf
from config_utils import check_offline_vlm, is_offline
from detector import crop_box
from embed import Embedder, ROOT
from policy import effective_policy, finalize, initial, semantic_policy
from results import make_result
from taxonomy import (FORMAT_VERSION, hash_files, package_versions, taxonomy_fingerprint,
                      validate_checkpoint)


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
    return tomllib.loads(path.read_text(encoding="utf-8")).get("augment", {})


def annotations_rows(annotations: Path, split: str | None) -> list[tuple[str, list[dict]]]:
    """Файлы и объекты, годные для обучения: split, не bad_frame, статус ok/bad, не other."""
    rows = []
    for row in json.loads(annotations.read_text(encoding="utf-8")):
        if split and row.get("split") != split or "bad_frame" in row.get("frame_flags", []):
            continue
        objects = [o for o in row["objects"] if o["status"] in ("bad", "ok") and o["type"] != "other"]
        if objects:
            rows.append((row["file"], objects))
    return rows


def target_of(obj: dict) -> str:
    return json.dumps([obj["type"], obj["status"], obj.get("damage")], ensure_ascii=False)


def train(cfg, annotations, images, split, output, root_cfg=None):
    rows = annotations_rows(annotations, split)
    origins = [target_of(obj) for _, objects in rows for obj in objects]
    if len(set(origins)) < 2:
        raise ValueError("Для обучения нужны хотя бы два класса объектов")
    support = Counter(origins)  # до копий: иначе редкий класс «подтверждает» себя сам
    augmented = Counter(origins)
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
                    augmented[target] += 1
                if len(batch) >= 16:
                    flush()
    flush()
    clf = make_clf(float(cfg.get("C", 32.0))).fit(np.asarray(vectors), targets)
    root_cfg = root_cfg or {}
    tax = (root_cfg.get("taxonomy") or {})
    used_images = [images / file for file, _ in rows]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "taxonomy": tax,
        "taxonomy_sha256": taxonomy_fingerprint(tax),
        "classes": [json.loads(label) for label in clf.classes_],
        "clf": clf,
        "support": dict(support),
        "support_original": dict(support),
        "support_augmented": dict(augmented),
        "config": {k: cfg[k] for k in ("encoder", "size", "pad", "crop_min_side") if k in cfg}
        | {"C": float(cfg.get("C", 32.0))},
        "policy": semantic_policy(cfg),
        "training_data": {"annotations": str(annotations), "images": str(images), "split": split,
                          "files": len(rows), "originals": len(origins), "examples": len(targets)},
        "data_hashes": {
            "annotations": hash_files([annotations], [ROOT]),
            "images": hash_files(used_images, [images]),
        },
        "package_versions": package_versions(),
        "augmentation_params": params,
    }
    tmp = output.with_name(output.name + ".tmp")
    joblib.dump(state, tmp)
    tmp.replace(output)
    print(dict(support))
    if cfg.get("augment"):
        print(f"аугментированных копий: {len(targets) - len(origins)}, всего примеров {len(targets)}")


class CropDino:
    workers = 1

    def __init__(self, cfg: dict, detector_cfg: dict | None, root_cfg: dict | None = None):
        from detector import Detector
        self.cfg_variant = dict(cfg)
        path = ROOT / cfg["checkpoint"]
        state = joblib.load(path)
        self.tax = (root_cfg or {}).get("taxonomy") or {}
        self.offline = is_offline(cfg, root_cfg)
        legacy = "error" if self.offline or cfg.get("strict_checkpoint") else cfg.get("legacy_checkpoint", "warn")
        validate_checkpoint(state, self.tax, legacy=legacy)
        self.state = state
        self.cfg, self.clf = state["config"], state["clf"]
        self.raw_labels = list(self.clf.classes_)
        self.classes = state.get("classes") or [json.loads(label) for label in self.raw_labels]
        if len(self.classes) != len(self.raw_labels):
            raise ValueError("Checkpoint: classes и clf.classes_ разной длины")
        self.support = state.get("support_original", state.get("support"))
        try:
            self.variant = effective_policy(state.get("policy"), cfg)
        except ValueError:
            raise
        self.policy_name = str(self.variant.get("class_policy", cfg.get("class_policy", "detector")))
        if self.policy_name not in ("joint", "detector"):
            raise ValueError("class_policy должен быть joint или detector")
        self.cascade = bool(self.variant.get("cascade", cfg.get("cascade", False)))
        self.variant["cascade"] = self.cascade
        self.variant["mode"] = "cascade" if self.cascade else "legacy"
        self.vlm = self._load_vlm(self.variant, root_cfg)
        self.vlm_ok, self.vlm_reason = vlm_ready(self.vlm, self.variant)
        self.encoder = Embedder(str(ROOT / self.cfg["encoder"]),
                                device=(detector_cfg or cfg).get("device", "cpu"), size=self.cfg["size"])
        dcfg = dict(detector_cfg or {})
        if self.offline:
            dcfg["offline"] = True
        self.det = Detector(dcfg) if dcfg else None
        tag = json.dumps([self.variant, detector_cfg], sort_keys=True, default=str)
        if self.vlm is not None:
            tag += self.vlm.fingerprint()  # поведение зависит от промпта/модели: кэш обязан обновиться
        import hashlib
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
        check_offline_vlm(root_cfg or {}, cfg, self.offline)
        from vlm import VLM
        from config_utils import resolve_variant
        return VLM(resolve_variant({"variants": variants_cfg}, str(name)), self.tax)

    def fingerprint(self):
        return "crop-dino-v3-" + self.signature

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

    def _legacy(self, obj: dict, scores) -> dict | None:
        typ = obj.get("type")
        decision = initial(self.raw_labels, self.classes, scores, mode="legacy", cfg=self.variant,
                           support=self.support, type_hint=typ, class_policy=self.policy_name)
        out = finalize(decision, cfg=self.variant, tax=self.tax, vlm=None)
        if out is None:
            return None
        return {**obj, "type": out.get("type") or typ or "other", "status": out["status"],
                "damage": out.get("damage"), "description": out.get("description", ""),
                "review_reasons": out.get("review_reasons", [])}

    def _cascade(self, obj: dict, scores, crop_img, telemetry: dict | None) -> dict | None:
        typ = obj.get("type")
        decision = initial(self.raw_labels, self.classes, scores, mode="cascade", cfg=self.variant,
                           support=self.support, type_hint=typ, class_policy=self.policy_name)
        vlm_norm = {"available": self.vlm_ok, "label": None, "answer": None, "description": "",
                    "error": None, "bad_json": False}
        if decision.get("needs_vlm") and self.vlm_ok:
            from cascade import classify_crop_safe
            call = classify_crop_safe(self.vlm, crop_img, typ or "")
            self._account(telemetry, call)
            vlm_norm.update(answer=call.get("answer"), error=call.get("error"),
                            bad_json=bool(call.get("bad_json")), available=True,
                            description=str((call.get("answer") or {}).get("description") or ""))
        elif decision.get("needs_vlm") and self.vlm_reason:
            vlm_norm["error"] = self.vlm_reason
        out = finalize(decision, cfg=self.variant, tax=self.tax, vlm=vlm_norm)
        if out is None:
            return None
        result = {**obj, "type": out.get("type") or typ or "other", "status": out["status"],
                  "damage": out.get("damage"), "description": out.get("description", ""),
                  "review_reasons": out.get("review_reasons", [])}
        if vlm_norm.get("error"):
            result["vlm_error"] = str(vlm_norm["error"])[:300]
            if telemetry is not None and not telemetry.get("error"):
                telemetry["error"] = str(vlm_norm["error"])[:300]
        if result["status"] == "review":
            result["damage"] = None
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
        return make_result(objects=objects, scope="mixed", seconds=time.monotonic() - start,
                           cost=telemetry["cost"], bad_json=telemetry["bad_json"], error=telemetry["error"],
                           detections=boxes, det_seconds=det_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--variant", default="det_dino")
    args = parser.parse_args()
    root = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    cfg = root["variants"][args.variant]
    train(cfg, args.annotations, args.images, args.split, ROOT / cfg["checkpoint"], root_cfg=root)
