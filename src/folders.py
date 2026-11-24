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
    from collections import defaultdict

    exact = {n.replace("\\", "/").casefold(): n for n in names}
    aliases = defaultdict(set)
    for n in names:
        aliases[Path(n).name.casefold()].add(n)
        aliases[Path(n).stem.casefold()].add(n)

    def hit(value):
        key = str(value).strip().replace("\\", "/").casefold()
        if key in exact:
            return exact[key]
        matches = aliases.get(Path(key).name) or aliases.get(Path(key).stem, set())
        return next(iter(matches)) if len(matches) == 1 else None

    if df.empty:
        return {}
    scores = {}
    for col in df:
        hits = {hit(v) for v in df[col]} - {None}
        explicit = {str(v).strip().replace("\\", "/").casefold() for v in df[col]} & exact.keys()
        scores[col] = (len(hits), len(explicit))
    best = max(scores.values(), default=(0, 0))
    columns = [col for col, score in scores.items() if score == best]
    if not best[0] or len(columns) != 1:
        print("Таблица: не удалось однозначно определить колонку имени фото")
        return {}
    col = columns[0]
    result, duplicates = {}, set()
    for row in df.to_dict("records"):
        name = hit(row[col])
        if name is not None:
            if name in result:
                duplicates.add(name)
            result[name] = {key: value for key, value in row.items() if key != col}
    for name in duplicates:
        result.pop(name)
    if len(result) != len(names):
        print(f"Таблица: однозначные строки для {len(result)} из {len(names)} фото")
    return result


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
