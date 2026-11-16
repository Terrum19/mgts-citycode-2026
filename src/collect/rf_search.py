"""Поиск датасетов на Roboflow Universe через API: классы и число картинок без скриншотов.

    python src/collect/rf_search.py "manhole" "broken manhole" --pages 3 --out data/rf_search.json

Ключ берётся из ROBOFLOW_API_KEY в .env.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

API = "https://api.roboflow.com/universe/search"
# слова в названиях классов, которые говорят о состоянии объекта
CONDITION = re.compile(
    r"broke|damag|crack|miss|lose|lost|open|uncover|circle|good|normal|defect|tilt|lean|"
    r"rust|corro|sunk|sink|displac|fall|hazard|bad|ok\b|intact",
    re.I,
)


def load_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    env = Path(__file__).resolve().parents[2] / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        sys.exit("ROBOFLOW_API_KEY не найден ни в окружении, ни в .env")
    return key


def search(query: str, key: str, pages: int) -> list[dict]:
    out = []
    for page in range(1, pages + 1):
        try:
            r = requests.get(API, params={"q": query, "page": page, "api_key": key}, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  ! {query!r} p{page}: {e}", file=sys.stderr)
            break
        res = r.json().get("results", [])
        out += res
        if len(res) < r.json().get("page_size", 12):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--out", default="data/rf_search.json")
    args = ap.parse_args()
    key = load_key()

    seen: dict[str, dict] = {}
    for q in args.queries:
        for d in search(q, key, args.pages):
            seen.setdefault(d["url"], d)

    rows = []
    for d in seen.values():
        classes = d.get("classes") or []
        cond = [c for c in classes if CONDITION.search(str(c))]
        rows.append({
            "url": d["url"], "name": d.get("name"), "type": d.get("type"),
            "images": d.get("images") or 0, "classes": classes, "condition_classes": cond,
        })
    rows.sort(key=lambda r: (len(r["condition_classes"]) > 0, r["images"]), reverse=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for r in rows:
        mark = "*" if r["condition_classes"] else " "
        cls = ", ".join(map(str, r["classes"]))[:110]
        print(f"{mark} {r['images']:>6}  {r['url'].removeprefix('https://universe.roboflow.com/'):<55} [{cls}]")
    print(f"\n{len(rows)} датасетов, с классами состояния: {sum(bool(r['condition_classes']) for r in rows)} -> {args.out}")


if __name__ == "__main__":
    main()
