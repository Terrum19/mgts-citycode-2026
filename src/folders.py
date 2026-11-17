"""Загрузчик формата финала: фото разложены по папкам «<тип объекта> <разделитель> <дефект>»,
в каждой папке Excel с описанием дефекта по каждому фото.

    python src/folders.py data/sim_final            # проверить, как распознались папки и колонки
    python src/folders.py data/final --sep "_"      # если разделитель в названиях папок другой

Класс = путь папки целиком (так не надо угадывать разделитель для обучения); тип и дефект
отделяются только для отчёта. Вложенные папки «Колодец/Сломана крышка» тоже поддерживаются.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TABLE_EXT = {".xlsx", ".xls", ".csv"}
SEPS = [" / ", " - ", " – ", " — ", "__", "_", "-", "."]


def read_table(path: Path):
    import pandas as pd
    if path.suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    return pd.read_excel(path)


def split_label(label: str, sep: str | None) -> tuple[str, str]:
    if "/" in label:
        t, _, d = label.partition("/")
        return t.strip(), d.strip()
    for s in [sep] if sep else SEPS:
        if s and s in label:
            t, _, d = label.partition(s)
            return t.strip(), d.strip()
    return label, ""


def match_table(df, names: list[str]) -> dict[str, dict]:
    """{имя фото -> строка таблицы}: колонку с файлом ищем по совпадениям (с расширением или без)."""
    keys = {n.lower(): n for n in names} | {Path(n).stem.lower(): n for n in names}

    def hit(v):
        v = Path(str(v).strip()).name.lower()
        return keys.get(v) or keys.get(Path(v).stem)

    if df.empty:
        return {}
    col = max(df.columns, key=lambda c: sum(bool(hit(v)) for v in df[c]))
    if not any(hit(v) for v in df[col]):  # имён файлов в таблице нет — надеемся на порядок строк
        return {n: r for n, r in zip(sorted(names), df.to_dict("records"))}
    return {hit(r[col]): {k: v for k, v in r.items() if k != col} for r in df.to_dict("records") if hit(r[col])}


def description_column(rows: list[dict]) -> str | None:
    """Колонка описания — текстовая с самой большой средней длиной."""
    lens = Counter()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, str):
                lens[k] += len(v)
    return lens.most_common(1)[0][0] if lens else None


def load(root: Path, sep: str | None = None) -> list[dict]:
    """-> [{"file": путь от root, "label", "type", "defect", "description", "extra": {...}}]."""
    root = Path(root)
    items = []
    for d in sorted(p for p in [root, *root.rglob("*")] if p.is_dir()):
        imgs = sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        if not imgs or d == root:
            continue
        label = d.relative_to(root).as_posix()
        meta = {}
        for t in sorted(p for p in d.iterdir() if p.suffix.lower() in TABLE_EXT and not p.name.startswith("~$")):
            meta |= match_table(read_table(t), imgs)
        dcol = description_column(list(meta.values()))
        typ, defect = split_label(label, sep)
        for name in imgs:
            extra = {k: v for k, v in meta.get(name, {}).items() if v == v}  # v == v отсекает NaN
            items.append({"file": f"{label}/{name}", "label": label, "type": typ, "defect": defect,
                          "description": str(extra.pop(dcol, "") or "") if dcol else "", "extra": extra})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sep")
    args = ap.parse_args()
    items = load(Path(args.root), args.sep)
    print(f"{len(items)} фото, {len({x['label'] for x in items})} классов")
    for label, n in sorted(Counter(x["label"] for x in items).items()):
        ex = next(x for x in items if x["label"] == label)
        has = sum(bool(x["description"]) for x in items if x["label"] == label)
        print(f"  {n:4d}  {label!r:40} -> тип {ex['type']!r}, дефект {ex['defect']!r}; описаний {has}/{n}")
    ex = next((x for x in items if x["description"]), None)
    if ex:
        print("пример:", ex["file"], "|", ex["description"][:100], "|", ex["extra"])
    no = re.sub(r"\s+", " ", ", ".join(sorted({x["label"] for x in items if not x["description"]})))
    if no:
        print("без описаний:", no[:300])


if __name__ == "__main__":
    main()
