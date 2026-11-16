"""Список целей для съёмки панорам: configs/pano_targets.json (коммитится, чтобы у всех был одинаковый).

    python3 src/collect/pano/make_targets.py

Берёт data/raw/osm_all.json (python3 src/collect/osm_points.py --all --out data/raw/osm_all.json):
все колодцы и шкафы + фонарные опоры, прореженные по сетке ~300 м (их 11 тысяч).
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAMP_CELL = 0.003  # ~300 м по широте


def main() -> None:
    random.seed(0)
    pts = json.loads((ROOT / "data/raw/osm_all.json").read_text())
    out, lamp_cells = [], set()
    random.shuffle(pts)
    for p in pts:
        if p["type"] == "street_lamp":
            cell = (round(p["lat"] / LAMP_CELL), round(p["lon"] / (LAMP_CELL * 1.8)))
            if cell in lamp_cells:
                continue
            lamp_cells.add(cell)
        out.append({"id": p["id"], "type": p["type"], "lat": round(p["lat"], 7), "lon": round(p["lon"], 7)})
    # колодцы первыми: их мало и они главные
    order = {"manhole": 0, "street_cabinet": 1, "street_lamp": 2}
    out.sort(key=lambda p: order[p["type"]])
    dst = ROOT / "configs/pano_targets.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    counts = {t: sum(p["type"] == t for p in out) for t in order}
    print(f"{len(out)} целей {counts} -> {dst.relative_to(ROOT)} ({dst.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
