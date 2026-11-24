"""Build crop training manifest from panorama dev and distinct public manhole sources."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect.build_smoke import MAP, SRC, ROOT


def main():
    labels = json.loads((ROOT / "data/labels.json").read_text())
    test_sites = {r["location_id"] for r in labels if r["split"] == "test"}
    rows = [{**r, "file": "raw/panoramas/" + r["file"]} for r in labels
            if r["split"] == "dev" and r["location_id"] not in test_sites]
    excluded = {Path(r["file"]).stem.split("_")[0] for r in json.loads((ROOT / "data/smoke/manholes/labels.json").read_text())}
    coco = json.loads((SRC / "_annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i["file_name"] for i in coco["images"]}
    counts = Counter()
    for a in coco["annotations"]:
        name = images[a["image_id"]]
        source = name.split("_")[0]
        cls = cats[a["category_id"]]
        if cls not in MAP or source in excluded or counts[cls] >= 100:
            continue
        x,y,w,h = a["bbox"]
        if min(w,h) < 1:
            continue
        status, damage = MAP[cls]
        rows.append({"file": str((SRC/name).relative_to(ROOT/"data")), "objects": [
            {"bbox":[x,y,x+w,y+h], "type":"manhole","status":status,"damage":damage}]})
        excluded.add(source)
        counts[cls] += 1
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
