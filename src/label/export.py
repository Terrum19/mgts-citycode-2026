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
URL, H = ENV["LABEL_STUDIO_URL"], {"Authorization": "Token " + ENV["LABEL_STUDIO_TOKEN"]}
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


def main() -> None:
    projects = {p["title"]: p["id"] for p in requests.get(f"{URL}/api/projects", headers=H, params={"page_size": 100}, timeout=60).json()["results"]}
    labels, total = [], 0
    for title in TITLES:
        tasks = requests.get(f"{URL}/api/projects/{projects[title]}/export", headers=H,
                             params={"exportType": "JSON", "download_all_tasks": "true"}, timeout=300).json()
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
