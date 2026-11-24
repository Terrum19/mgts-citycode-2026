"""Разбиение кадров на dev / test ДО разметки.

    python3 src/label/split.py            # -> configs/split.json (коммитится: разбиение одно на всех)

Группы: кадры, снятые ближе ~50 м друг к другу (или с одним location_id), всегда в одной части —
иначе почти одинаковые кадры одного места окажутся и в dev, и в test.
dev ~1/3 (размечаем с авторазметкой, подбираем промпт), test ~2/3 (размечаем с нуля, меряем).
Повторный запуск не перемешивает уже распределённые кадры: новые кадры добавляются, старые остаются.
"""
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data/raw/panoramas/meta.jsonl"
OUT = ROOT / "configs/split.json"
RADIUS_M = 50
DEV_SHARE = 1 / 3


def dist(a, b) -> float:
    dy = (a[0] - b[0]) * 111_320
    dx = (a[1] - b[1]) * 111_320 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


def main() -> None:
    rows = [json.loads(line) for line in META.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("file")]
    old = json.loads(OUT.read_text()) if OUT.exists() else {}

    # union-find по близости панорам и общему location_id
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["location_id"] == b["location_id"] or \
                    dist((a["pano_lat"], a["pano_lon"]), (b["pano_lat"], b["pano_lon"])) < RADIUS_M:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r["file"])

    random.seed(42)
    split = {}
    type_of = {r["file"]: r["type"] for r in rows}
    seen, dev = Counter(), Counter()                # по типам объектов: стратификация
    gl = list(groups.values())
    random.shuffle(gl)
    for files in gl:
        t = Counter(type_of[f] for f in files).most_common(1)[0][0]
        prev = {old[f] for f in files if f in old}
        if prev:                                    # группа уже распределена — не двигаем
            part = "test" if "test" in prev else "dev"
        else:
            part = "dev" if dev[t] < DEV_SHARE * (seen[t] + len(files)) else "test"
        seen[t] += len(files)
        dev[t] += sum(old.get(f, part) == "dev" for f in files)
        for f in files:
            # уже распределённый кадр не двигаем никогда: он уже лежит в своём проекте Label Studio
            # (новый кадр может «склеить» старые группы из разных частей)
            split[f] = old.get(f, part)

    OUT.write_text(json.dumps(dict(sorted(split.items())), indent=1))
    by_type = Counter((split[r["file"]], r["type"]) for r in rows)
    print(f"{len(rows)} кадров, {len(groups)} групп мест")
    for part in ("dev", "test"):
        print(f"  {part}: {sum(v == part for v in split.values())} "
              f"(колодцы {by_type[(part, 'manhole')]}, опоры {by_type[(part, 'street_lamp')]}, "
              f"шкафы {by_type[(part, 'street_cabinet')]})")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
