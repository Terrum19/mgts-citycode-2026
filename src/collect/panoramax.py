"""Уличные кадры Москвы из Panoramax (открытые фото, CC BY-SA 4.0), в которых в поле зрения есть объект из OSM.

    python3 src/collect/panoramax.py index                 # метаданные всех кадров по Москве -> data/raw/panoramax/index.jsonl
    python3 src/collect/panoramax.py select --max 150      # отбор кадров с объектами + скачивание

Идём «от кадра к объекту»: у Panoramax кадров мало, поэтому для каждого кадра ищем объекты OSM
в секторе обзора камеры (по азимуту `view:azimuth`) на расстоянии 3..25 м.
Лицензия CC BY-SA: автор (producer) и ссылка сохраняются в meta.jsonl — указываем их при показе кадров.
"""
import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

API = "https://api.panoramax.xyz/api/search"
MOSCOW = (37.30, 55.50, 37.98, 56.00)   # lon_min, lat_min, lon_max, lat_max (без Новой Москвы)
OUT = Path("data/raw/panoramax")
MAX_DIST = {"manhole": 20, "street_cabinet": 25, "street_lamp": 30}
MIN_DIST = 3
HALF_FOV = 30          # градусов от оси камеры: объект должен быть ближе к центру кадра
MIN_SPACING = 40       # м между выбранными кадрами, чтобы не было 20 кадров одного двора


def get(url: str, **params) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 4:
                raise
            print(f"  ! {e}, повтор")
            time.sleep(5 * 2 ** attempt)
    return {}


def tile_photos(bbox: tuple, depth: int = 0) -> list[dict]:
    """Если в клетке >= лимита кадров, делим её на 4 (API отдаёт не больше 1000 за раз)."""
    data = get(API, bbox=",".join(f"{v:.5f}" for v in bbox), limit=1000)
    feats = data.get("features", [])
    if len(feats) >= 1000 and depth < 8:
        x0, y0, x1, y1 = bbox
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        out = []
        for b in [(x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)]:
            out += tile_photos(b, depth + 1)
        return out
    return feats


def slim(f: dict) -> dict:
    p = f["properties"]
    io = p.get("pers:interior_orientation") or {}
    return {
        "id": f["id"], "lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1],
        "azimuth": p.get("view:azimuth"), "datetime": p.get("datetime"), "fov": io.get("field_of_view"),
        "camera": io.get("camera_model"), "producer": p.get("geovisio:producer"),
        "license": p.get("license"), "collection": f.get("collection"),
        "sd": (f.get("assets", {}).get("sd") or {}).get("href"),
        "page": next((link["href"] for link in f.get("links", []) if link.get("rel") == "self"), None),
    }


def cmd_index(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = tile_photos(MOSCOW)
    seen, rows = set(), []
    for f in feats:
        if f["id"] not in seen:
            seen.add(f["id"])
            rows.append(slim(f))
    (OUT / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    flat = sum(1 for r in rows if r["fov"] != 360)
    print(f"{len(rows)} кадров ({flat} обычных, {len(rows) - flat} панорам 360) -> {OUT / 'index.jsonl'}")
    print("камеры:", Counter(r["camera"] for r in rows).most_common(6))
    print("годы:", Counter((r["datetime"] or "?")[:4] for r in rows).most_common(6))


def dist_bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    d = 2 * 6371000 * math.asin(math.sqrt(a))
    b = math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)))
    return d, (b + 360) % 360


def cmd_select(args) -> None:
    random.seed(0)
    photos = [json.loads(line) for line in (OUT / "index.jsonl").read_text().splitlines()]
    photos = [p for p in photos if p["fov"] != 360 and p["azimuth"] is not None and p["sd"]]
    objs = json.loads(Path(args.osm).read_text())
    grid = defaultdict(list)                       # ячейки ~100 м для быстрого поиска соседей
    for o in objs:
        grid[(round(o["lat"] * 1000), round(o["lon"] * 600))].append(o)

    cands = []
    for ph in photos:
        gy, gx = round(ph["lat"] * 1000), round(ph["lon"] * 600)
        seen_objs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for o in grid.get((gy + dy, gx + dx), []):
                    d, b = dist_bearing(ph["lat"], ph["lon"], o["lat"], o["lon"])
                    rel = (b - ph["azimuth"] + 540) % 360 - 180
                    if MIN_DIST <= d <= MAX_DIST[o["type"]] and abs(rel) <= HALF_FOV:
                        seen_objs.append({"id": o["id"], "type": o["type"], "dist": round(d, 1), "rel_bearing": round(rel, 1)})
        if seen_objs:
            cands.append({**ph, "objects_in_view": seen_objs})

    # приоритет: колодцы и шкафы (их мало) > фонари; один объект — один кадр; лимит на поездку
    prio = {"manhole": 0, "street_cabinet": 1, "street_lamp": 2}
    random.shuffle(cands)
    cands.sort(key=lambda c: min(prio[o["type"]] for o in c["objects_in_view"]))
    used_obj, chosen = set(), []
    for c in cands:
        ids = {o["id"] for o in c["objects_in_view"]}
        if ids & used_obj or any(dist_bearing(c["lat"], c["lon"], x["lat"], x["lon"])[0] < MIN_SPACING for x in chosen):
            continue
        chosen.append(c); used_obj |= ids
        if len(chosen) >= args.max:
            break
    print(f"кадров-кандидатов: {len(cands)}, выбрано: {len(chosen)}; "
          f"по типам: {Counter(min(c['objects_in_view'], key=lambda o: prio[o['type']])['type'] for c in chosen)}")

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta = OUT / "meta.jsonl"
    done = {json.loads(line)["id"] for line in meta.read_text().splitlines()} if meta.exists() else set()
    with meta.open("a") as fh:
        for k, c in enumerate(chosen, 1):
            if c["id"] in done:
                continue
            r = requests.get(c["sd"], timeout=120)
            if r.status_code != 200:
                print(f"  ! {c['id']}: http {r.status_code}")
                continue
            (img_dir / f"{c['id']}.jpg").write_bytes(r.content)
            # location_id = первый объект в кадре: кадры одного объекта всегда попадут в одну часть dev/test
            rec = {**c, "file": f"images/{c['id']}.jpg", "location_id": c["objects_in_view"][0]["id"]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 20 == 0:
                print(f"  {k}/{len(chosen)}")
    print(f"готово -> {img_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    s = sub.add_parser("select")
    s.add_argument("--max", type=int, default=150)
    s.add_argument("--osm", default="data/raw/osm_all.json")
    args = ap.parse_args()
    {"index": cmd_index, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()
