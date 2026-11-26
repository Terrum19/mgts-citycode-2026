"""Сборка объединённого чистого датасета ДСС (6 классов: ТКШ/РЩ × Норма/ЛКП/МП) и честный сплит.

    python src/prepare_dss.py

Объединяет две части выгрузки:
  - data/doc-1795639569/Датасет-ДСС
  - data/doc-1795639519/Датасет-ДСС
не изменяя исходники, исключает корневую схему Фото.1.jpg (копирует её в docs/dss_reference_photo1.jpg),
проставляет группы серий одной опоры (Группа), извлекает EXIF GPS и создаёт:
  - data/dss_full   (61 фото, 6 подпапок <Тип>/<Состояние>, описание.xlsx + meta.csv)
  - data/dss_train  (49 фото, 4 фолда StratifiedGroupKFold)
  - data/dss_test   (12 фото, 1 фолд StratifiedGroupKFold, 0 общих Группа с train)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import ROOT  # noqa: E402
from folders import load  # noqa: E402
from predict import exif_gps  # noqa: E402

DEFAULT_SOURCES = [
    ROOT / "data/doc-1795639569/Датасет-ДСС",
    ROOT / "data/doc-1795639519/Датасет-ДСС",
]

SITE_GROUPS = {
    "35453686": "site_3545368x",
    "35453688": "site_3545368x",
    "35453689": "site_3545368x",
    "42123333": "site_4212333x",
    "42123334": "site_4212333x",
    "53393563": "site_533935xx",
    "53393585": "site_533935xx",
    "36955093": "site_3695509x",
    "36955095": "site_3695509x",
}

TYPE_CODE = {"ТКШ": "tksh", "РЩ": "rsh"}
DEFECT_CODE = {
    "Удовлетворительное состояние": "ok",
    "Неудовлетворительное состояние ЛКП": "lkp",
    "МП": "mp",
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def group_for(typ: str, defect: str, filename: str) -> str:
    stem = Path(filename).stem
    if stem in SITE_GROUPS:
        return SITE_GROUPS[stem]
    tcode = TYPE_CODE.get(typ, typ)
    dcode = DEFECT_CODE.get(defect, defect)
    return f"{tcode}_{dcode}_{stem}"


def write_dataset(records: list[dict], out_root: Path) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    per_dir: dict[Path, list[dict]] = defaultdict(list)
    meta_rows: list[dict] = []
    for rec in records:
        rel = rec["canon_file"]
        dst = out_root / rel
        link_or_copy(rec["src_path"], dst)
        row = {
            "Имя файла": dst.name,
            "ID объекта": rec["object_id"],
            "Статус_ДСС": rec["dss_status"],
            "Описание": rec["description"],
            "Группа": rec["group"],
            "lat": rec["lat"],
            "lon": rec["lon"],
        }
        per_dir[dst.parent].append(row)
        meta_rows.append({
            "file": rel,
            "ID объекта": rec["object_id"],
            "Статус_ДСС": rec["dss_status"],
            "Описание_эталон": rec["description"],
            "Группа": rec["group"],
            "lat": rec["lat"],
            "lon": rec["lon"],
        })
    for folder, rows in sorted(per_dir.items()):
        pd.DataFrame(rows).to_excel(folder / "описание.xlsx", index=False)
    pd.DataFrame(meta_rows).to_csv(out_root / "meta.csv", index=False, encoding="utf-8-sig")


def prepare(sources: list[Path], out_dir: Path) -> dict:
    records = []
    for src_root in sources:
        if not src_root.is_dir():
            raise FileNotFoundError(f"Не найдена папка исходного датасета: {src_root}")
        ref_photo = src_root / "Фото.1.jpg"
        if ref_photo.is_file():
            ref_dst = ROOT / "docs/dss_reference_photo1.jpg"
            ref_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ref_photo, ref_dst)
        for item in folders_load(src_root):
            src_path = src_root / item["file"]
            typ = item["type"]
            defect = item["defect"]
            canon_label = f"{typ}/{defect}"
            filename = src_path.name
            gps = exif_gps(src_path)
            records.append({
                "src_path": src_path,
                "canon_label": canon_label,
                "canon_file": f"{canon_label}/{filename}",
                "type": typ,
                "defect": defect,
                "object_id": src_path.stem,
                "dss_status": str(item["extra"].get("Статус", "") or ""),
                "description": item["description"],
                "group": group_for(typ, defect, filename),
                "lat": gps.get("lat", ""),
                "lon": gps.get("lon", ""),
            })
    records.sort(key=lambda r: r["canon_file"])

    full_dir = out_dir / "dss_full"
    train_dir = out_dir / "dss_train"
    test_dir = out_dir / "dss_test"
    write_dataset(records, full_dir)

    y = np.array([r["canon_label"] for r in records])
    groups = np.array([r["group"] for r in records])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, te_idx = next(cv.split(np.zeros(len(records)), y, groups))
    train_records = [records[i] for i in tr_idx]
    test_records = [records[i] for i in te_idx]

    train_groups = {r["group"] for r in train_records}
    test_groups = {r["group"] for r in test_records}
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Обнаружена утечка групп между train и test: {overlap}")

    write_dataset(train_records, train_dir)
    write_dataset(test_records, test_dir)

    return {
        "total": len(records),
        "train": len(train_records),
        "test": len(test_records),
        "classes": dict(sorted(Counter(r["canon_label"] for r in records).items())),
        "train_classes": dict(sorted(Counter(r["canon_label"] for r in train_records).items())),
        "test_classes": dict(sorted(Counter(r["canon_label"] for r in test_records).items())),
        "group_overlap": len(overlap),
        "gps_count": sum(1 for r in records if r["lat"] != ""),
    }


def folders_load(root: Path) -> list[dict]:
    return load(root)


def main() -> None:
    ap = argparse.ArgumentParser(description="Объединение и честный сплит Датасет-ДСС")
    ap.add_argument("--sources", nargs="+", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    summary = prepare(args.sources, args.out_dir)
    print(f"Собран датасет ДСС: всего {summary['total']} фото, EXIF GPS у {summary['gps_count']}")
    print(f"Сплит: train={summary['train']}, test={summary['test']}, пересечение Группа={summary['group_overlap']}")
    print("Классы (full / train / test):")
    for cls, total in summary["classes"].items():
        tr_n = summary["train_classes"].get(cls, 0)
        te_n = summary["test_classes"].get(cls, 0)
        print(f"  {cls:45s} {total:2d}  ({tr_n:2d} / {te_n:2d})")


if __name__ == "__main__":
    main()
