"""Загрузка и разбор датасетов из фото, папок и ZIP-архивов для интерфейса."""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from PIL import Image

from folders import IMG_EXT, TABLE_EXT, match_table, read_table, split_label
from predict import read_boxes, read_meta


def exif_gps(path: Path) -> dict:
    """Широта/долгота из EXIF. Пустой dict, если GPS нет или координаты некорректны."""
    try:
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)
    except Exception:
        return {}
    if not gps:
        return {}

    def to_deg(value) -> float | None:
        try:
            deg, minutes, seconds = (float(v) for v in value)
        except Exception:
            return None
        return deg + minutes / 60 + seconds / 3600

    lat, lon = to_deg(gps.get(2) or ()), to_deg(gps.get(4) or ())
    if lat is None or lon is None:
        return {}
    if str(gps.get(1, "N")).strip().upper().startswith("S"):
        lat = -lat
    if str(gps.get(3, "E")).strip().upper().startswith("W"):
        lon = -lon
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {}
    return {"lat": round(lat, 7), "lon": round(lon, 7)}

IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
NON_DESC_COLS = {
    "группа", "group", "id объекта", "object_id", "asset_id",
    "lat", "lon", "latitude", "longitude", "широта", "долгота",
}
GT_COLS = ("истинный класс", "ground_truth", "gt_label", "класс", "class", "label")
OK_WORDS = ("норм", "исправ", "без дефект", "без поврежд", "нет дефект", "ok", "good", "normal")

DSS_LABELS = {
    "мп.рщ": ("РЩ", "МП"),
    "мп.ткш": ("ТКШ", "МП"),
    "неудовлетворительное состояние лкп.рщ": ("РЩ", "Неудовлетворительное состояние ЛКП"),
    "неудовлетворительное состояние лкп.ткш": ("ТКШ", "Неудовлетворительное состояние ЛКП"),
    "удовлетворительное состояние рщ": ("РЩ", "Удовлетворительное состояние"),
    "удовлетворительное состояние ткш": ("ТКШ", "Удовлетворительное состояние"),
}


def _dss_key(label: str) -> str:
    s = re.sub(r"\s*\.\s*", ".", label.strip().casefold())
    return re.sub(r"\s+", " ", s)


def split_dataset_label(label: str, sep: str | None = None) -> tuple[str, str]:
    """Разбор имени папки класса -> (тип объекта, состояние/дефект)."""
    dss = DSS_LABELS.get(_dss_key(label))
    if dss is not None:
        return dss
    return split_label(label, sep)


def is_ok_defect(defect: str) -> bool:
    """Определить, означает ли строка дефекта исправное состояние."""
    normalized = (defect or "").strip().casefold()
    if not normalized:
        return False
    if "неудовлетвор" in normalized:
        return False
    if "удовлетвор" in normalized:
        return True
    return "неисправ" not in normalized and any(w in normalized for w in OK_WORDS)


def _is_ignored_parts(parts: tuple[str, ...]) -> bool:
    for part in parts:
        low = part.casefold()
        if not part or part in (".", ".."):
            return True
        if part == "__MACOSX" or low in IGNORED_NAMES:
            return True
        if part.startswith("._") or part.startswith("~$") or part.startswith("."):
            return True
    return False


def safe_rel_path(raw_name: str) -> Path | None:
    """Безопасный относительный путь без обхода каталогов."""
    norm = str(raw_name).replace("\\", "/").strip().lstrip("/")
    if not norm:
        return None
    posix = PurePosixPath(norm)
    if posix.is_absolute() or ".." in posix.parts:
        return None
    if _is_ignored_parts(posix.parts):
        return None
    return Path(*posix.parts)


def unwrap_single_root(base_dir: Path) -> Path:
    """Если весь датасет лежит внутри одной корневой папки-обёртки, спуститься в неё."""
    cur = base_dir
    while True:
        try:
            entries = [p for p in cur.iterdir() if not _is_ignored_parts((p.name,))]
        except OSError:
            break
        files = [p for p in entries if p.is_file()]
        dirs = [p for p in entries if p.is_dir()]
        if files or len(dirs) != 1:
            break
        inner = dirs[0]
        inner_entries = [p for p in inner.iterdir() if not _is_ignored_parts((p.name,))]
        if not inner_entries:
            break
        cur = inner
    return cur


def _decode_zip_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    # ZIP без UTF-8 флага (0x800) с кириллицей на Windows часто хранит cp866 в виде cp437.
    if not (info.flag_bits & 0x800):
        try:
            raw = name.encode("cp437")
            for enc in ("cp866", "utf-8", "cp1251"):
                try:
                    decoded = raw.decode(enc)
                    if decoded != name:
                        return decoded
                except UnicodeDecodeError:
                    continue
        except UnicodeEncodeError:
            pass
    return name


def ingest_uploaded_photos(files, out_input_dir: Path) -> Path:
    """Сохранить набор загруженных фотографий в out_input_dir."""
    out_input_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    for f in files:
        raw_name = Path(str(f.name).replace("\\", "/")).name
        if _is_ignored_parts((raw_name,)):
            continue
        count = seen.get(raw_name, 0) + 1
        seen[raw_name] = count
        save_name = f"{Path(raw_name).stem}__{count}{Path(raw_name).suffix}" if count > 1 else raw_name
        (out_input_dir / save_name).write_bytes(f.getbuffer())
    return out_input_dir


def ingest_uploaded_directory(files, out_input_dir: Path) -> Path:
    """Сохранить папку, загруженную через st.file_uploader(..., accept_multiple_files='directory')."""
    out_input_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        rel = safe_rel_path(f.name)
        if rel is None:
            continue
        dst = out_input_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f.getbuffer())
    return unwrap_single_root(out_input_dir)


def ingest_zip_archive(zip_source, out_input_dir: Path) -> Path:
    """Безопасно распаковать ZIP-архив (UploadedFile, BytesIO или Path) в out_input_dir."""
    out_input_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_source) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = safe_rel_path(_decode_zip_name(info))
            if rel is None:
                continue
            dst = out_input_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
    return unwrap_single_root(out_input_dir)


def ingest_local_directory(src_dir: Path) -> Path:
    """Проверить локальную папку с датасетом."""
    resolved = Path(src_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Папка не найдена: {src_dir}")
    return resolved


def _description_column(rows: list[dict]) -> str | None:
    lens: Counter = Counter()
    for r in rows:
        for k, v in r.items():
            if str(k).strip().casefold() in NON_DESC_COLS:
                continue
            if isinstance(v, str) and v.strip():
                lens[k] += len(v)
    return lens.most_common(1)[0][0] if lens else None


def _read_any_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    df = read_table(path)
    return df.to_dict("records")


def _clean_meta_row(row: dict) -> dict:
    clean = {}
    for k, v in row.items():
        if v == v and v is not None:  # отсекает NaN
            clean[str(k)] = v
    return clean


@dataclass
class DatasetBundle:
    root: Path
    items: list[dict]
    has_gt: bool = False
    gt_kind: str | None = None
    classes: list[str] = field(default_factory=list)
    boxes: dict | None = None
    has_gps: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def paths(self) -> list[Path]:
        return [item["path"] for item in self.items]

    @property
    def meta_by_key(self) -> dict[str, dict]:
        return {item["key"]: item["meta"] for item in self.items if item["meta"]}


def _is_dss_scheme_root_file(path: Path, root: Path, has_subdirs: bool) -> bool:
    """Исключить корневой образец Фото.1.jpg из Датасет-ДСС, если рядом есть папки классов."""
    return has_subdirs and path.parent == root and path.name.casefold() == "фото.1.jpg"


def inspect_dataset(
    root: Path,
    extra_meta_file: Path | None = None,
    extra_boxes_file: Path | None = None,
) -> DatasetBundle:
    """Просканировать датасет: фото, метаданные, EXIF GPS, boxes.json и эталонную разметку."""
    import pandas as pd

    root = Path(root)
    warnings: list[str] = []
    all_files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and not _is_ignored_parts(p.relative_to(root).parts)
    )
    has_subdirs = any(p.parent != root and p.suffix.lower() in IMG_EXT for p in all_files)
    img_paths = [
        p for p in all_files
        if p.suffix.lower() in IMG_EXT and not _is_dss_scheme_root_file(p, root, has_subdirs)
    ]
    if not img_paths:
        raise ValueError("В загруженном датасете не найдено изображений (.jpg, .png, .webp, .bmp, .tif)")

    keys = [p.relative_to(root).as_posix() for p in img_paths]

    # Таблицы метаданных внутри датасета (корневые и внутри подпапок) + отдельно загруженная таблица
    table_paths = [
        p for p in all_files
        if p.suffix.lower() in (TABLE_EXT | {".jsonl"})
    ]
    meta_by_key: dict[str, dict] = {}
    for tp in table_paths:
        try:
            rows = _read_any_table(tp)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Не удалось прочитать таблицу {tp.name}: {exc}")
            continue
        if not rows:
            continue
        # Если таблица лежит в подпапке, сначала сопоставляем с файлами этой подпапки
        rel_dir = tp.parent.relative_to(root).as_posix()
        scope_keys = keys if tp.parent == root else [k for k in keys if k.startswith(rel_dir + "/")]
        if not scope_keys:
            scope_keys = keys
        matched = match_table(pd.DataFrame(rows), scope_keys)
        for k, row in matched.items():
            cleaned = _clean_meta_row(row)
            meta_by_key.setdefault(k, {}).update(cleaned)

    if extra_meta_file is not None:
        try:
            rows = read_meta(extra_meta_file)
            if rows:
                matched = match_table(pd.DataFrame(rows), keys)
                if len(matched) != len(keys):
                    warnings.append(
                        f"В таблице {extra_meta_file.name} сопоставлено {len(matched)} из {len(keys)} фото"
                    )
                for k, row in matched.items():
                    meta_by_key.setdefault(k, {}).update(_clean_meta_row(row))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Ошибка чтения таблицы метаданных: {exc}")

    # Рамки (boxes.json)
    boxes = None
    box_candidate = extra_boxes_file or next((p for p in all_files if p.name.casefold() == "boxes.json"), None)
    if box_candidate is not None:
        try:
            raw_boxes = read_boxes(box_candidate)
            # Поддерживаем ключи как по полному rel_path, так и по basename
            remapped = {}
            by_name = {}
            for k in keys:
                by_name.setdefault(Path(k).name, []).append(k)
            for k in keys:
                if k in raw_boxes:
                    remapped[k] = raw_boxes[k]
                elif Path(k).name in raw_boxes and len(by_name[Path(k).name]) == 1:
                    remapped[k] = raw_boxes[Path(k).name]
            boxes = remapped if len(remapped) == len(keys) else raw_boxes
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Ошибка чтения файла рамок: {exc}")

    # Разметка объектов (labels.json)
    labels_json_file = next((p for p in all_files if p.name.casefold() == "labels.json"), None)
    gt_objects_by_key: dict[str, list[dict]] = {}
    if labels_json_file is not None:
        try:
            raw_labels = json.loads(labels_json_file.read_text(encoding="utf-8-sig"))
            if isinstance(raw_labels, list):
                by_name = {Path(k).name: k for k in keys}
                for entry in raw_labels:
                    f_key = str(entry.get("file", "")).replace("\\", "/")
                    target_key = f_key if f_key in keys else by_name.get(Path(f_key).name)
                    if target_key:
                        gt_objects_by_key[target_key] = entry.get("objects") or []
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Ошибка чтения {labels_json_file.name}: {exc}")

    # Определение классовой разметки по папкам
    folder_labels = [
        p.parent.relative_to(root).as_posix() if p.parent != root else ""
        for p in img_paths
    ]
    non_empty_folders = sorted({fl for fl in folder_labels if fl})
    all_in_folders = all(bool(fl) for fl in folder_labels)
    recognized_dss = bool(non_empty_folders) and all(
        _dss_key(fl) in DSS_LABELS or "/" in fl for fl in non_empty_folders
    )
    has_folder_gt = all_in_folders and (len(non_empty_folders) >= 2 or recognized_dss)

    # Определение разметки из колонки таблицы
    gt_col = None
    if not gt_objects_by_key and not has_folder_gt and meta_by_key:
        for row in meta_by_key.values():
            for col in row:
                if str(col).strip().casefold() in GT_COLS:
                    gt_col = col
                    break
            if gt_col:
                break

    gt_kind: str | None = None
    if gt_objects_by_key:
        gt_kind = "objects"
    elif has_folder_gt:
        gt_kind = "folders"
    elif gt_col is not None:
        gt_kind = "table"

    dcol = _description_column(list(meta_by_key.values()))
    items: list[dict] = []
    has_gps = False
    classes_set: set[str] = set()

    for p, key, folder_lab in zip(img_paths, keys, folder_labels):
        meta = dict(meta_by_key.get(key, {}))
        if "lat" not in meta or "lon" not in meta:
            gps = exif_gps(p)
            if gps:
                meta.setdefault("lat", gps["lat"])
                meta.setdefault("lon", gps["lon"])
        if meta.get("lat") not in (None, "") and meta.get("lon") not in (None, ""):
            has_gps = True

        gt_label = gt_type = gt_defect = gt_status = None
        gt_objects = None
        if gt_kind == "objects":
            gt_objects = gt_objects_by_key.get(key)
            if gt_objects is not None:
                gt_status = "bad" if any(o.get("status") == "bad" for o in gt_objects) else "ok"
        elif gt_kind == "folders":
            gt_type, gt_defect = split_dataset_label(folder_lab)
            gt_label = f"{gt_type} / {gt_defect}" if gt_defect else gt_type
            gt_status = "ok" if is_ok_defect(gt_defect) else "bad"
            classes_set.add(gt_label)
        elif gt_kind == "table" and gt_col and gt_col in meta:
            raw_lab = str(meta[gt_col]).strip()
            if raw_lab:
                gt_type, gt_defect = split_dataset_label(raw_lab)
                gt_label = f"{gt_type} / {gt_defect}" if gt_defect else gt_type
                gt_status = "ok" if is_ok_defect(gt_defect or raw_lab) else "bad"
                classes_set.add(gt_label)

        items.append({
            "key": key,
            "path": p,
            "meta": meta,
            "meta_description": str(meta.get(dcol, "") or "") if dcol else "",
            "gt_label": gt_label,
            "gt_type": gt_type,
            "gt_defect": gt_defect,
            "gt_status": gt_status,
            "gt_objects": gt_objects,
        })

    return DatasetBundle(
        root=root,
        items=items,
        has_gt=gt_kind is not None,
        gt_kind=gt_kind,
        classes=sorted(classes_set),
        boxes=boxes,
        has_gps=has_gps,
        warnings=warnings,
    )
