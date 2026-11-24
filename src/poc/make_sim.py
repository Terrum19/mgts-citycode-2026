"""Имитация формата финала: папки «<Тип> - <Дефект>» с фото и Excel-описанием в каждой.

    python src/poc/make_sim.py            # -> data/sim_final/

Источники: люки с Roboflow (jie-yu/manhole-cover-4, одна категория на фото, повторы по исходному имени
убраны) + вырезы опор и шкафов из наших размеченных панорам. Описания — заглушки по шаблону
(настоящих у нас нет); на финале их даст Excel организаторов.
"""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RF = ROOT / "data/roboflow/jie-yu_manhole-cover-4/train"
OUT = ROOT / "data/sim_final"
PER_CLASS = 120
RF_FOLDERS = {"good": "Колодец - Норма", "broke": "Колодец - Сломана крышка", "lose": "Колодец - Нет крышки",
              "uncovered": "Колодец - Крышка сдвинута", "circle": "Колодец - Разрушено кольцо"}
DESC = {"Колодец - Норма": "Крышка на месте, без повреждений.",
        "Колодец - Сломана крышка": "Крышка колодца с проломом, требуется замена.",
        "Колодец - Нет крышки": "Крышка отсутствует, колодец открыт — опасно для пешеходов и транспорта.",
        "Колодец - Крышка сдвинута": "Крышка сдвинута с посадочного места.",
        "Колодец - Разрушено кольцо": "Разрушено покрытие вокруг люка, крышка просела.",
        "Опора - Норма": "Опора стоит ровно, видимых повреждений нет.",
        "Шкаф - Норма": "Шкаф закрыт, корпус без повреждений."}


def main() -> None:
    random.seed(42)
    shutil.rmtree(OUT, ignore_errors=True)
    by_folder = defaultdict(list)  # папка -> [(имя, функция сохранения)]

    c = json.loads((RF / "_annotations.coco.json").read_text())
    cats = {x["id"]: x["name"] for x in c["categories"]}
    per = defaultdict(set)
    for a in c["annotations"]:
        per[a["image_id"]].add(cats[a["category_id"]])
    seen = set()
    for im in c["images"]:
        labs = per.get(im["id"], set())
        src = im["file_name"].split("_")[0]
        if len(labs) != 1 or src in seen or next(iter(labs)) not in RF_FOLDERS:
            continue
        seen.add(src)
        path = RF / im["file_name"]
        by_folder[RF_FOLDERS[next(iter(labs))]].append((f"rf_{src}.jpg", lambda d, p=path: shutil.copy(p, d)))

    labels = json.loads((ROOT / "data/labels.json").read_text())
    for lab in labels:
        for k, o in enumerate(lab["objects"]):
            folder = {"pole": "Опора - Норма", "cabinet": "Шкаф - Норма"}.get(o["type"])
            x1, y1, x2, y2 = o["bbox"]
            if not folder or o["status"] != "ok" or min(x2 - x1, y2 - y1) < 12 or max(x2 - x1, y2 - y1) < 60:
                continue
            side = max(x2 - x1, y2 - y1) * 1.2           # квадратный вырез: как фото «одного объекта»
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            box = (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
            path = ROOT / "data/raw/panoramas" / lab["file"]
            by_folder[folder].append((f"pano_{Path(lab['file']).stem}_{k}.jpg",
                                      lambda d, p=path, b=box: Image.open(p).convert("RGB").crop(b).save(d, quality=92)))

    import pandas as pd
    for folder, items in sorted(by_folder.items()):
        random.shuffle(items)
        items = items[:PER_CLASS]
        (OUT / folder).mkdir(parents=True)
        for name, save in items:
            save(OUT / folder / name)
        # «Группа» — только для честной проверки (вырезы одного кадра не делим между train и val)
        groups = [n.rsplit("_", 1)[0] if n.startswith("pano_") else n for n, _ in items]
        pd.DataFrame({"Имя файла": [n for n, _ in items], "Описание дефекта": DESC[folder], "Группа": groups}) \
            .to_excel(OUT / folder / "описание.xlsx", index=False)
        print(f"{folder}: {len(items)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
