"""Выгрузка разметки из Label Studio в наш формат (тот, что читают evaluate.py и run.py).

    python3 src/label/export.py            # -> data/labels.json (+ статистика)

Берёт последнюю аннотацию человека по каждой задаче (предсказания модели игнорируются).
Формат: [{"file", "width", "height", "split", "location_id", "frame_flags": [...],
          "objects": [{"type", "bbox": [x1,y1,x2,y2] в пикселях, "status", "damage", "note"}]}]
"""
import json
import re
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
ENV = dict(re.findall(r"^(LABEL_STUDIO_\w+)=(.*)$", (ROOT / ".env").read_text(), re.M))
URL = ENV["LABEL_STUDIO_URL"].strip().strip("\"'").rstrip("/")
TOKEN = ENV["LABEL_STUDIO_TOKEN"].strip().strip("\"'")
TOKEN_TYPE = ENV.get("LABEL_STUDIO_TOKEN_TYPE", "auto").strip().lower()
if TOKEN_TYPE not in ("auto", "legacy", "pat"):
    raise SystemExit("LABEL_STUDIO_TOKEN_TYPE должен быть auto, legacy или pat")
IS_PAT = TOKEN_TYPE == "pat" or (TOKEN_TYPE == "auto" and TOKEN.count(".") == 2)
H = {} if IS_PAT else {"Authorization": "Token " + TOKEN}
TITLES = ("МГТС dev: проверка авторазметки", "МГТС test: разметка с нуля")


def convert(task: dict) -> dict | None:
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    ann = max(anns, key=lambda a: a.get("updated_at") or "")
    regions, flags = {}, []
    w = h = None
    for r in ann["result"]:
        if r["from_name"] == "frame":
            flags += r["value"]["choices"]
            continue
        reg = regions.setdefault(r["id"], {})
        w, h = r.get("original_width", w), r.get("original_height", h)
        v = r["value"]
        if r["type"] == "rectanglelabels":
            reg["type"] = v["rectanglelabels"][0]
            x, y = v["x"] * w / 100, v["y"] * h / 100
            reg["bbox"] = [round(x), round(y), round(x + v["width"] * w / 100), round(y + v["height"] * h / 100)]
        elif r["from_name"] in ("status", "damage"):
            reg[r["from_name"]] = v["choices"][0]
        elif r["from_name"] == "note":
            reg["note"] = " ".join(v.get("text", []))
    objects = []
    for reg in regions.values():
        if "bbox" not in reg:
            continue
        status = reg.get("status", "unclear")
        objects.append({"type": reg["type"], "bbox": reg["bbox"], "status": status,
                        "damage": reg.get("damage") if status == "bad" else None, "note": reg.get("note", "")})
    d = task["data"]
    return {"file": d["file"], "width": w or 1280, "height": h or 600, "split": d["split"],
            "location_id": d["location_id"], "frame_flags": flags, "objects": objects,
            "annotator": ann.get("completed_by")}


def refresh_access():
    try:
        response = requests.post(URL + "/api/token/refresh/", json={"refresh": TOKEN}, timeout=60,
                                 allow_redirects=False)
    except requests.RequestException:
        raise SystemExit("Не удалось подключиться к Label Studio для обмена PAT. Проверьте ZeroTier.")
    if not response.ok:
        raise SystemExit(f"Обмен Personal Access Token: HTTP {response.status_code}. "
                         "Проверьте, что PAT создан на этом сервере, не истёк и не отозван.")
    try:
        access = response.json()["access"]
        if not isinstance(access, str) or not access:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise SystemExit("Сервер не вернул access token при обмене PAT.")
    H["Authorization"] = "Bearer " + access


def get_json(path, **kwargs):
    if IS_PAT and not H:
        refresh_access()
    try:
        response = requests.get(URL.rstrip("/") + path, headers=H, **kwargs)
        if response.status_code == 401 and IS_PAT:
            refresh_access()
            response = requests.get(URL.rstrip("/") + path, headers=H, **kwargs)
    except requests.RequestException:
        raise SystemExit("Не удалось подключиться к Label Studio. Проверьте сервер и ZeroTier.")
    if not response.ok:
        hint = ("Проверьте токен и LABEL_STUDIO_TOKEN_TYPE (auto / legacy / pat)." if response.status_code == 401
                else "Проверьте права доступа к проектам." if response.status_code == 403
                else "Проверьте адрес сервера и доступность API.")
        raise SystemExit(f"Label Studio: HTTP {response.status_code} при GET {path}. {hint}")
    try:
        return response.json()
    except ValueError:
        raise SystemExit(f"Label Studio вернул не JSON при GET {path}. Проверьте адрес API.")


def project_list(payload):
    rows = payload if isinstance(payload, list) else payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(p, dict) or "title" not in p or "id" not in p for p in rows):
        raise SystemExit("Неожиданный формат списка проектов: ожидается массив или объект с массивом results. "
                         "Экспорт остановлен; существующая разметка сохранена.")
    return rows


def main() -> None:
    payload = get_json("/api/projects", params={"page_size": 100}, timeout=60)
    projects = {p["title"]: p["id"] for p in project_list(payload)}
    missing = [title for title in TITLES if title not in projects]
    if missing:
        raise SystemExit("Не найдены доступные проекты: " + "; ".join(missing))
    labels, total = [], 0
    for title in TITLES:
        tasks = get_json(f"/api/projects/{projects[title]}/export",
                         params={"exportType": "JSON", "download_all_tasks": "true"}, timeout=300)
        if not isinstance(tasks, list):
            raise SystemExit("Неожиданный формат экспорта задач; существующая разметка сохранена.")
        total += len(tasks)
        labels += [x for x in map(convert, tasks) if x]
    labels.sort(key=lambda x: x["file"])
    (ROOT / "data/labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1))
    objs = [o for lab in labels for o in lab["objects"]]
    print(f"размечено {len(labels)} из {total} кадров: {Counter(lab['split'] for lab in labels)}")
    print("объекты:", Counter(o["type"] for o in objs))
    print("статусы:", Counter(o["status"] for o in objs), "| повреждения:", Counter(o["damage"] for o in objs if o["damage"]))
    print("плохие кадры:", sum("bad_frame" in lab["frame_flags"] for lab in labels))
    print("-> data/labels.json")


if __name__ == "__main__":
    main()
