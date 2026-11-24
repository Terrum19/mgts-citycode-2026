"""Подменяет авторазметку в проекте dev на свежую (data/label/tasks_dev.json), НЕ трогая размеченные кадры.

    python3 src/label/refresh_predictions.py            # показать, что изменится
    python3 src/label/refresh_predictions.py --apply    # применить

Нужно после смены классов в config.toml: make_tasks.py пересчитывает предсказания, этот скрипт
заливает их в задачи, где ещё нет ни одной разметки человека.
"""
import argparse
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
ENV = dict(re.findall(r"^(LABEL_STUDIO_\w+)=(.*)$", (ROOT / ".env").read_text(), re.M))
URL, H = ENV["LABEL_STUDIO_URL"], {"Authorization": "Token " + ENV["LABEL_STUDIO_TOKEN"]}
DEV_TITLE = "МГТС dev: проверка авторазметки"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    projects = {p["title"]: p["id"] for p in requests.get(f"{URL}/api/projects", headers=H, params={"page_size": 100}, timeout=60).json()["results"]}
    pid = projects[DEV_TITLE]
    fresh = {t["data"]["file"]: t.get("predictions", []) for t in json.loads((ROOT / "data/label/tasks_dev.json").read_text())}

    tasks = requests.get(f"{URL}/api/tasks", headers=H, params={"project": pid, "page_size": 1000, "fields": "all"}, timeout=120).json()
    tasks = tasks.get("tasks", tasks)
    skipped = changed = 0
    for t in tasks:
        if t.get("total_annotations", 0) or t.get("annotations"):
            skipped += 1
            continue
        new = fresh.get(t["data"]["file"])
        if not new:
            continue
        changed += 1
        if not args.apply:
            continue
        for p in t.get("predictions", []):
            pid_ = p["id"] if isinstance(p, dict) else p
            requests.delete(f"{URL}/api/predictions/{pid_}", headers=H, timeout=60).raise_for_status()
        for p in new:
            r = requests.post(f"{URL}/api/predictions", headers=H, timeout=60,
                              json={"task": t["id"], "model_version": p["model_version"], "result": p["result"]})
            r.raise_for_status()
    print(f"размечены людьми (не трогаю): {skipped}; {'обновлено' if args.apply else 'будет обновлено'}: {changed}")


if __name__ == "__main__":
    main()
