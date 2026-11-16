"""Создаёт/обновляет проекты Label Studio на хабе и импортирует задачи.

    python3 src/label/ls_setup.py

Нужно в .env: LABEL_STUDIO_URL, LABEL_STUDIO_TOKEN. Перед этим: make_tasks.py и rsync кадров на хаб
(data/raw/panoramas/*.jpg -> companion@hub:mgts/media/panoramas/).
Повторный запуск безопасен: обновляет конфиг интерфейса и добавляет только новые кадры.
"""
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
ENV = dict(re.findall(r"^(LABEL_STUDIO_\w+)=(.*)$", (ROOT / ".env").read_text(), re.M))
URL, H = ENV["LABEL_STUDIO_URL"], {"Authorization": "Token " + ENV["LABEL_STUDIO_TOKEN"]}
MEDIA_ON_HUB = "/home/companion/mgts/media/panoramas"
PROJECTS = {
    "dev": ("МГТС dev: проверка авторазметки",
            "Рамки уже нарисованы моделью. Поправь: удали лишние, подвинь неточные, ДОРИСУЙ пропущенные объекты, проверь статус."),
    "test": ("МГТС test: разметка с нуля",
             "Модели здесь нет: выдели ВСЕ видимые колодцы, опоры и шкафы и укажи их состояние. Это честная проверка, не подглядывай в dev."),
}


def api(method: str, path: str, **kw):
    r = requests.request(method, URL + path, headers=H, timeout=120, **kw)
    if r.status_code >= 400:
        raise SystemExit(f"{method} {path}: {r.status_code} {r.text[:300]}")
    return r.json() if r.text else {}


def main() -> None:
    config = (ROOT / "configs/label_config.xml").read_text()
    existing = {p["title"]: p for p in api("GET", "/api/projects", params={"page_size": 100})["results"]}
    for part, (title, descr) in PROJECTS.items():
        body = {"title": title, "description": descr, "label_config": config,
                "show_collab_predictions": part == "dev", "maximum_annotations": 1}
        if title in existing:
            pid = existing[title]["id"]
            api("PATCH", f"/api/projects/{pid}", json=body)
        else:
            pid = api("POST", "/api/projects", json=body)["id"]
        storages = api("GET", "/api/storages/localfiles", params={"project": pid})
        if not storages:
            api("POST", "/api/storages/localfiles", json={
                "project": pid, "title": "panoramas", "path": MEDIA_ON_HUB, "use_blob_urls": False})

        tasks = json.loads((ROOT / f"data/label/tasks_{part}.json").read_text())
        have = set()
        page = 1
        while True:
            r = api("GET", "/api/tasks", params={"project": pid, "page": page, "page_size": 500, "fields": "all"})
            batch = r.get("tasks", r if isinstance(r, list) else [])
            have |= {t["data"].get("file") for t in batch}
            if len(batch) < 500:
                break
            page += 1
        new = [t for t in tasks if t["data"]["file"] not in have]
        if new:
            api("POST", f"/api/projects/{pid}/import", json=new)
        print(f"{part}: проект #{pid} «{title}», задач было {len(have)}, добавлено {len(new)}")
    print(f"Открыть: {URL}")


if __name__ == "__main__":
    main()
