"""Имитация финала на чужих данных «с машины»: RDD2022 (смартфон на лобовом стекле, Чехия и Индия).

    python src/poc/make_rdd.py                       # -> data/rdd_train/, data/rdd_test/

Класс фото = тип повреждения дороги, если на фото повреждения ровно одного типа; фото без разметки —
«Норма»; фото с несколькими типами отбрасываются (у организаторов в папке один класс).
train — по --train фото на класс (как то, что дадут на финале), test — следующие --test фото на класс
(«расширенный» набор, который модель не видела). Формат как у финала: папки «Тип - Дефект» + Excel.
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/ext/rdd"
NAMES = {None: "Дорога - Норма", "D00": "Дорога - Продольная трещина", "D10": "Дорога - Поперечная трещина",
         "D20": "Дорога - Сетка трещин", "D40": "Дорога - Выбоина"}
WHAT = {"D00": "продольная трещина", "D10": "поперечная трещина", "D20": "сетка трещин", "D40": "выбоина"}


def scan(country: str):
    """-> {имя фото: (класс, описание)} из xml-разметки, без распаковки архива на диск."""
    z = zipfile.ZipFile(SRC / f"{country}.zip")
    names = z.namelist()
    imgs = {Path(n).stem: n for n in names if "/train/images/" in n and n.endswith(".jpg")}
    out = {}
    for n in names:
        if "/train/annotations/xmls/" not in n or not n.endswith(".xml"):
            continue
        r = ET.fromstring(z.read(n))
        W, H = int(r.findtext("size/width")), int(r.findtext("size/height"))
        objs = [(o.findtext("name"), [float(o.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")])
                for o in r.iter("object")]
        kinds = {k for k, _ in objs}
        if len(kinds) > 1 or (kinds and not kinds <= set(WHAT)) or Path(n).stem not in imgs:
            continue
        kind = next(iter(kinds), None)
        if kind is None:
            desc = "Покрытие без видимых повреждений."
        else:
            big = max((b[2] - b[0]) * (b[3] - b[1]) for _, b in objs) / (W * H)
            desc = f"{WHAT[kind].capitalize()}, {len(objs)} шт.; крупнейшая занимает ~{big:.0%} кадра."
        out[Path(n).stem] = (NAMES[kind], desc, imgs[Path(n).stem])
    return z, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=["Czech", "India"])
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--test", type=int, default=150)
    args = ap.parse_args()
    rnd = random.Random(42)
    pool = defaultdict(list)
    zips = {}
    for c in args.countries:
        zips[c], found = scan(c)
        for stem, (label, desc, member) in found.items():
            pool[label].append((c, stem, desc, member))
    for part in ("train", "test"):
        shutil.rmtree(ROOT / f"data/rdd_{part}", ignore_errors=True)
    for label, xs in sorted(pool.items()):
        rnd.shuffle(xs)
        print(f"{label}: всего {len(xs)}", end="")
        for part, chunk in (("train", xs[:args.train]), ("test", xs[args.train:args.train + args.test])):
            d = ROOT / f"data/rdd_{part}" / label
            d.mkdir(parents=True)
            rows = []
            for c, stem, desc, member in chunk:
                (d / f"{stem}.jpg").write_bytes(zips[c].read(member))
                rows.append({"Имя файла": f"{stem}.jpg", "Описание дефекта": desc, "Страна": c, "Группа": stem})
            pd.DataFrame(rows).to_excel(d / "описание.xlsx", index=False)
            print(f", {part} {len(chunk)}", end="")
        print()


if __name__ == "__main__":
    main()
