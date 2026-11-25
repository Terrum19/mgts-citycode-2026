"""Аудит разметки и данных перед обучением/оценкой.

    python3 src/audit_data.py --labels data/labels.json --images data/raw/panoramas

Проверяет не только «файл читается», но и то, из-за чего метрики становятся ложными:
неизвестные типы/статусы, битые рамки, target_not_visible с объектами, утечку
location_id между split, дубли изображений по хэшу, отсутствующие/битые фото и уже
состоявшиеся прогоны по test. Ошибки -> exit 1; предупреждения не блокируют, но
попадают в JSON и печать.

Модуль stdlib-only; PIL нужен только для проверки размеров изображений.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results import valid_bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train", "calibration", "dev", "test", "holdout", "holdout_v1"}
STATUSES = {"ok", "bad", "unclear"}
FLAGS = {"bad_frame", "target_not_visible"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bug(errors: list[str], message: str) -> None:
    errors.append(message)


def audit_labels(
    labels: list[dict],
    *,
    taxonomy: dict | None = None,
    images: Path | None = None,
    protocol: dict | None = None,
    runs: Path | None = None,
    hash_images: bool = True,
) -> dict:
    """Вернуть отчёт аудита. Ошибки в ``errors``, предупреждения в ``warnings``."""
    errors: list[str] = []
    warnings: list[str] = []
    taxonomy = taxonomy or {}
    types = set((taxonomy.get("object_types") or {}).keys())
    damages = set((taxonomy.get("damage") or {}).keys())
    if not types:
        warnings.append("таксономия пуста: проверка типов объектов ограничена")

    seen_files: set[str] = set()
    locations: dict[str, set[str]] = defaultdict(set)
    image_hashes: dict[str, list[str]] = defaultdict(list)
    counts = {"frames": len(labels), "objects": 0, "by_split": Counter(), "by_type": Counter(),
              "by_status": Counter(), "by_damage": Counter(), "by_flags": Counter()}
    invalid_labels: list[dict] = []
    missing_images: list[str] = []
    bad_frames_with_objects: list[str] = []

    for index, lab in enumerate(labels):
        name = str(lab.get("file") or "")
        if not name:
            _bug(errors, f"labels[{index}]: пустое file")
            continue
        if name in seen_files:
            _bug(errors, f"{name}: повтор file")
        seen_files.add(name)
        split = lab.get("split")
        if split not in SPLITS:
            _bug(errors, f"{name}: неизвестный split={split!r}")
        counts["by_split"][str(split)] += 1
        flags = list(lab.get("frame_flags") or [])
        for flag in flags:
            if flag not in FLAGS:
                _bug(errors, f"{name}: неизвестный frame_flag={flag!r}")
            counts["by_flags"][flag] += 1
        location = str(lab.get("location_id") or "")
        if location:
            locations[location].add(str(split))
        width, height = lab.get("width"), lab.get("height")
        objects = list(lab.get("objects") or [])
        counts["objects"] += len(objects)
        if "target_not_visible" in flags and objects:
            _bug(errors, f"{name}: target_not_visible, но размечено {len(objects)} объектов")
            invalid_labels.append({"file": name, "reason": "target_not_visible_with_objects"})
        if "bad_frame" in flags and objects:
            warnings.append(f"{name}: bad_frame с {len(objects)} объектами (статус кадра всё равно не оценивается)")
            bad_frames_with_objects.append(name)
        for obj_index, obj in enumerate(objects):
            typ = obj.get("type")
            status = obj.get("status")
            damage = obj.get("damage")
            if types and typ not in types and typ != "other":
                _bug(errors, f"{name}[{obj_index}]: неизвестный type={typ!r}")
            if status not in STATUSES:
                _bug(errors, f"{name}[{obj_index}]: неизвестный status={status!r}")
            counts["by_type"][str(typ)] += 1
            counts["by_status"][str(status)] += 1
            if status == "bad":
                counts["by_damage"][str(damage)] += 1
                if damage not in damages and damage != "unknown":
                    warnings.append(f"{name}[{obj_index}]: damage={damage!r} нет в таксономии")
            elif damage:
                warnings.append(f"{name}[{obj_index}]: damage при status={status!r}")
            box = valid_bbox(obj.get("bbox"))
            if box is None:
                _bug(errors, f"{name}[{obj_index}]: некорректная рамка {obj.get('bbox')!r}")
                invalid_labels.append({"file": name, "index": obj_index, "reason": "invalid_bbox"})
                continue
            if width and height and (box[2] > width or box[3] > height or box[0] < 0 or box[1] < 0):
                _bug(errors, f"{name}[{obj_index}]: рамка вне кадра {box} ({width}x{height})")
                invalid_labels.append({"file": name, "index": obj_index, "reason": "bbox_outside_image"})

        if images is not None:
            path = images / name
            if not path.exists() or path.stat().st_size == 0:
                missing_images.append(name)
                continue
            if hash_images:
                digest = sha256_file(path)
                image_hashes[digest].append(name)
                if width and height:
                    try:
                        from PIL import Image
                        with Image.open(path) as im:
                            if (im.width, im.height) != (width, height):
                                warnings.append(
                                    f"{name}: width/height в labels {width}x{height}, у фото {im.width}x{im.height}")
                    except Exception as exc:  # noqa: BLE001
                        _bug(errors, f"{name}: изображение не читается ({str(exc)[:120]})")

    for location, splits in sorted(locations.items()):
        if len(splits) > 1:
            _bug(errors, f"location_id={location!r} встречается в split {sorted(splits)}")

    duplicate_images = {digest: names for digest, names in image_hashes.items() if len(names) > 1}
    if duplicate_images:
        for digest, names in list(duplicate_images.items())[:20]:
            _bug(errors, f"одинаковые изображения по SHA-256 {digest[:12]}: {names[:4]}")

    if images is not None:
        for name in missing_images:
            _bug(errors, f"{name}: файла изображения нет в {images}")

    bad_in_split = Counter()
    for lab in labels:
        if any(o.get("status") == "bad" for o in (lab.get("objects") or [])):
            bad_in_split[str(lab.get("split"))] += 1
    for split, n in sorted(bad_in_split.items()):
        if n < 3:
            warnings.append(f"{split}: всего {n} кадров с bad — отбор по recall будет нестабильным")

    contaminated: list[str] = []
    run_names: list[str] = []
    if runs is not None and runs.exists():
        for path in sorted(runs.iterdir()):
            if not path.is_dir():
                continue
            run_names.append(path.name)
            dataset = path.name.split("__", 1)[0]
            if protocol:
                ds = (protocol.get("datasets") or {}).get(dataset) or {}
                if ds.get("role") == "contaminated" or ds.get("selection_eligible") is False:
                    contaminated.append(path.name)
    if contaminated:
        warnings.append("есть прогоны по contaminated/test: " + ", ".join(contaminated[:6]))

    status = "failed" if errors else "ok"
    report = {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "frames": counts["frames"],
            "objects": counts["objects"],
            "by_split": dict(counts["by_split"]),
            "by_type": dict(counts["by_type"]),
            "by_status": dict(counts["by_status"]),
            "by_damage": dict(counts["by_damage"]),
            "by_flags": dict(counts["by_flags"]),
            "bad_frames_by_split": dict(bad_in_split),
        },
        "locations": len(locations),
        "locations_cross_split": {k: sorted(v) for k, v in locations.items() if len(v) > 1},
        "invalid_labels": invalid_labels,
        "missing_images": missing_images,
        "bad_frames_with_objects": bad_frames_with_objects,
        "duplicate_images": {k: v for k, v in duplicate_images.items()},
        "runs": run_names,
        "contaminated_runs": contaminated,
        "hashes": {"labels": None, "images_manifest": None},
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=ROOT / "data/labels.json")
    ap.add_argument("--images", type=Path, default=None)
    ap.add_argument("--protocol", type=Path, default=ROOT / "configs/evaluation_protocol.json")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-hash-images", action="store_true")
    ap.add_argument("--strict-warnings", action="store_true")
    args = ap.parse_args()

    import tomllib
    cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8")) if (ROOT / "config.toml").exists() else {}
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")) if args.protocol.exists() else None
    report = audit_labels(labels, taxonomy=cfg.get("taxonomy"), images=args.images, protocol=protocol,
                          runs=args.runs, hash_images=not args.no_hash_images)
    report["hashes"]["labels"] = sha256_file(args.labels)
    if args.images:
        names = [lab.get("file") for lab in labels if lab.get("file")]
        manifest = {name: sha256_file(args.images / name) for name in names if (args.images / name).exists()}
        report["hashes"]["images_manifest"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report["hashes"]["images_count"] = len(manifest)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if report["errors"]:
        sys.exit(1)
    if args.strict_warnings and report["warnings"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
