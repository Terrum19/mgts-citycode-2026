"""Собирает набор для шага 0 из скачанного jie-yu/manhole-cover-4 (COCO) в наш формат.

    python src/collect/build_smoke.py

Вход:  data/roboflow/jie-yu_manhole-cover-4/train + configs/smoke_manholes.txt
Выход: data/smoke/manholes/images/*.jpg + data/smoke/manholes/labels.json
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/roboflow/jie-yu_manhole-cover-4/train"
LIST = ROOT / "configs/smoke_manholes.txt"
OUT = ROOT / "data/smoke/manholes"

# классы китайского набора -> наш статус и временный вид повреждения
MAP = {
    "good": ("ok", None),
    "broke": ("bad", "cover_broken"),
    "lose": ("bad", "cover_missing"),
    "uncovered": ("bad", "cover_open_or_shifted"),
    "circle": ("bad", "ring_or_surface_damaged"),
}


def main() -> None:
    coco = json.loads((SRC / "_annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_name = {i["file_name"]: i for i in coco["images"]}
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    labels = []
    for line in LIST.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _, fname = line.split("\t")
        img = by_name[fname]
        objs = []
        for a in anns.get(img["id"], []):
            x, y, w, h = a["bbox"]
            status, damage = MAP[cats[a["category_id"]]]
            objs.append({"type": "manhole", "bbox": [round(x), round(y), round(x + w), round(y + h)],
                         "status": status, "damage": damage})
        short = fname.split("_jpg")[0] + ".jpg"
        shutil.copy(SRC / fname, OUT / "images" / short)
        labels.append({"file": f"images/{short}", "width": img["width"], "height": img["height"],
                       "source": "roboflow:jie-yu/manhole-cover-4", "objects": objs})

    (OUT / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1))
    print(f"{len(labels)} фото -> {OUT}")


if __name__ == "__main__":
    main()
