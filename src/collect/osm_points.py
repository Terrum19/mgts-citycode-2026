"""Точки объектов инфраструктуры Москвы из OpenStreetMap (Overpass API).

    python3 src/collect/osm_points.py --per-type 150 --out data/raw/osm_points.json

Типы: manhole (кроме ливнёвок/канализации — нас интересуют связь и неизвестные),
street_cabinet, street_lamp. Отбор равномерный по городу: сетка ~1 км, не больше одной точки в ячейке.
"""
import argparse
import json
import random
from pathlib import Path

import requests

MIRRORS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
QUERY = """[out:json][timeout:300];
area["name"="Москва"]["admin_level"="4"]->.a;
(
  node["man_made"="manhole"](area.a);
  node["man_made"="street_cabinet"](area.a);
  node["highway"="street_lamp"](area.a);
);
out body;"""
SKIP_MANHOLE = {"drain", "sewer", "water", "heat", "gas"}


RAW_CACHE = Path("data/raw/osm_raw.json")


def fetch() -> list[dict]:
    if RAW_CACHE.exists():
        print(f"  кэш Overpass: {RAW_CACHE}")
        return json.loads(RAW_CACHE.read_text())
    last = None
    for url in MIRRORS:
        try:
            r = requests.post(url, data={"data": QUERY}, timeout=330,
                              headers={"User-Agent": "infra-dataset-research/0.1"})
            r.raise_for_status()
            elements = r.json()["elements"]
            RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
            RAW_CACHE.write_text(json.dumps(elements))
            return elements
        except (requests.RequestException, ValueError) as e:
            last = e
            print(f"  ! {url}: {e}")
    raise SystemExit(f"Overpass недоступен: {last}")


def kind(tags: dict) -> str | None:
    if tags.get("man_made") == "manhole":
        return None if tags.get("manhole") in SKIP_MANHOLE else "manhole"
    if tags.get("man_made") == "street_cabinet":
        return "street_cabinet"
    if tags.get("highway") == "street_lamp":
        return "street_lamp"
    return None


def spread(points: list[dict], n: int, cell_deg: float = 0.01) -> list[dict]:
    """Не больше одной точки на клетку ~1 км, чтобы кадры не были из одного двора."""
    random.shuffle(points)
    seen, out = set(), []
    for p in points:
        cell = (round(p["lat"] / cell_deg), round(p["lon"] / (cell_deg * 1.8)))
        if cell in seen:
            continue
        seen.add(cell)
        out.append(p)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=150)
    ap.add_argument("--out", default="data/raw/osm_points.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="все точки без прореживания по сетке")
    args = ap.parse_args()
    random.seed(args.seed)

    by_type: dict[str, list[dict]] = {}
    for e in fetch():
        k = kind(e.get("tags", {}))
        if k:
            by_type.setdefault(k, []).append({
                "id": f"osm_{e['id']}", "type": k, "lat": e["lat"], "lon": e["lon"],
                "tags": e.get("tags", {}),
            })
    out = []
    for k, pts in sorted(by_type.items()):
        sel = pts if args.all else spread(pts, args.per_type)
        print(f"{k}: всего {len(pts)}, отобрано {len(sel)}")
        out += sel
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{len(out)} точек -> {args.out}")


if __name__ == "__main__":
    main()
