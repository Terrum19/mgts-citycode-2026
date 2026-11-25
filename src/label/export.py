"""Выгрузка разметки и недостающих фото из Label Studio в data/labels.json.

    python3 src/label/export.py                  # выгрузить разметку + докачать недостающие фото
    python3 src/label/export.py --no-images      # только data/labels.json
    python3 src/label/export.py --all-projects   # все проекты с сервера, а не только dev/test
    python3 src/label/export.py --from-file raw_export.json  # конвертировать JSON, скачанный из браузера

Формат выхода: [{"file", "width", "height", "split", "location_id", "frame_flags": [...],
                 "objects": [{"type", "bbox": [x1,y1,x2,y2] в пикселях, "status", "damage", "note"}]}]
"""
import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TITLES = ("МГТС dev: проверка авторазметки", "МГТС test: разметка с нуля")
RETRY_CODES = (429, 500, 502, 503, 504)


def load_ls_env(root: Path = ROOT) -> tuple[str, str]:
    """Читает LABEL_STUDIO_URL и LABEL_STUDIO_TOKEN из окружения или .env."""
    vals = {}
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*(?:export\s+)?(LABEL_STUDIO_\w+)\s*=\s*(.*)$", line)
            if m:
                vals[m.group(1)] = m.group(2).strip().strip("'\"")
    url = (os.environ.get("LABEL_STUDIO_URL") or vals.get("LABEL_STUDIO_URL") or "").strip().rstrip("/")
    token = (os.environ.get("LABEL_STUDIO_TOKEN") or vals.get("LABEL_STUDIO_TOKEN") or "").strip()
    if not url or not token:
        raise SystemExit("Нужны LABEL_STUDIO_URL и LABEL_STUDIO_TOKEN в .env или переменных окружения")
    if "://" not in url:
        url = "http://" + url
    return url, token


def request_retry(session: requests.Session, method: str, url: str, timeout: int = 120, **kw) -> requests.Response:
    last_err = None
    for attempt in range(4):
        try:
            r = session.request(method, url, timeout=timeout, **kw)
            if r.status_code in RETRY_CODES and attempt < 3:
                time.sleep(2 * 2 ** attempt)
                continue
            return r
        except requests.RequestException as e:
            last_err = e
            if attempt < 3:
                time.sleep(2 * 2 ** attempt)
                continue
            raise
    raise last_err


def auth_headers(session: requests.Session, url: str, token: str) -> dict[str, str]:
    """Подбирает заголовок авторизации: Legacy Token или JWT Personal Access Token."""
    for headers in ({"Authorization": f"Token {token}"}, {"Authorization": f"Bearer {token}"}):
        r = request_retry(session, "GET", f"{url}/api/projects", headers=headers, params={"page_size": 1}, timeout=30)
        if r.ok:
            return headers
        if r.status_code != 401:
            raise SystemExit(f"Ошибка Label Studio {r.status_code}: {r.text[:300]}")
    # Label Studio >= 1.13 PAT: refresh JWT -> access JWT
    ref = request_retry(session, "POST", f"{url}/api/token/refresh", json={"refresh": token}, timeout=30)
    if ref.ok and ref.json().get("access"):
        return {"Authorization": f"Bearer {ref.json()['access']}"}
    raise SystemExit("Токен Label Studio отклонён (401). Проверь LABEL_STUDIO_TOKEN в .env")


def task_filename(task: dict) -> str | None:
    d = task.get("data") or {}
    raw = str(d.get("file") or "").strip()
    if raw:
        p = Path(raw)
        return raw if not p.is_absolute() and ".." not in p.parts else p.name
    img = str(d.get("image") or "").strip()
    if not img:
        return None
    parsed = urlparse(img)
    qs = parse_qs(parsed.query)
    if qs.get("d"):
        return Path(unquote(qs["d"][0])).name
    name = Path(unquote(parsed.path)).name
    return name or None


def _resolve_size(regions: dict, task: dict, images_dir: Path | None, fname: str) -> tuple[int, int]:
    for reg in regions.values():
        if reg.get("w") and reg.get("h"):
            return int(reg["w"]), int(reg["h"])
    if images_dir and fname and (images_dir / fname).exists():
        try:
            with Image.open(images_dir / fname) as im:
                return im.size
        except Exception:  # noqa: BLE001
            pass
    d = task.get("data") or {}
    return int(d.get("width") or 1280), int(d.get("height") or 600)


def convert(task: dict, default_split: str | None = None, split_map: dict[str, str] | None = None,
            images_dir: Path | None = None) -> dict | None:
    anns = [a for a in (task.get("annotations") or []) if not a.get("was_cancelled")]
    if not anns:
        return None
    ann = max(anns, key=lambda a: (a.get("updated_at") or a.get("created_at") or "", a.get("id") or 0))
    fname = task_filename(task)
    if not fname:
        return None

    regions, flags = {}, []
    for r in ann.get("result") or []:
        v = r.get("value") or {}
        if r.get("from_name") == "frame":
            flags += v.get("choices") or []
            continue
        rid = r.get("id")
        if rid is None:
            continue
        reg = regions.setdefault(rid, {})
        if r.get("original_width"):
            reg["w"] = r["original_width"]
        if r.get("original_height"):
            reg["h"] = r["original_height"]
        if r.get("type") == "rectanglelabels":
            labs = v.get("rectanglelabels") or []
            if labs and all(k in v for k in ("x", "y", "width", "height")):
                reg["type"] = labs[0]
                reg["pct"] = (float(v["x"]), float(v["y"]), float(v["width"]), float(v["height"]))
        elif r.get("from_name") in ("status", "damage"):
            choices = v.get("choices") or []
            if choices:
                reg[r["from_name"]] = choices[0]
        elif r.get("from_name") == "note":
            txt = v.get("text")
            reg["note"] = txt if isinstance(txt, str) else " ".join(txt or [])

    w, h = _resolve_size(regions, task, images_dir, fname)
    objects = []
    for reg in regions.values():
        if "pct" not in reg or "type" not in reg:
            continue
        rw, rh = float(reg.get("w") or w), float(reg.get("h") or h)
        px, py, pw, ph = reg["pct"]
        x, y = px * rw / 100, py * rh / 100
        bbox = [round(x), round(y), round(x + pw * rw / 100), round(y + ph * rh / 100)]
        status = reg.get("status", "unclear")
        objects.append({"type": reg["type"], "bbox": bbox, "status": status,
                        "damage": reg.get("damage") if status == "bad" else None, "note": reg.get("note", "")})

    d = task.get("data") or {}
    split = d.get("split") or (split_map or {}).get(fname) or default_split
    return {"file": fname, "width": w, "height": h, "split": split,
            "location_id": d.get("location_id"), "frame_flags": flags, "objects": objects,
            "annotator": ann.get("completed_by")}


def download_image(session: requests.Session, base_url: str, headers: dict[str, str],
                   image_ref: str, dest: Path) -> str:
    """Скачивает фото, если его ещё нет на диске; проверяет, что это валидная картинка."""
    if dest.exists() and dest.stat().st_size > 0:
        return "exists"
    if "://" in image_ref:
        url = image_ref
        req_headers = headers if urlparse(url).netloc == urlparse(base_url).netloc else {}
    else:
        url = f"{base_url}/{image_ref.lstrip('/')}"
        req_headers = headers
    r = request_retry(session, "GET", url, headers=req_headers, timeout=60)
    r.raise_for_status()
    with Image.open(io.BytesIO(r.content)) as im:
        im.verify()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    tmp.write_bytes(r.content)
    os.replace(tmp, dest)
    return "downloaded"


def infer_split(title: str) -> str | None:
    low = title.lower()
    if "dev" in low:
        return "dev"
    if "test" in low:
        return "test"
    return None


def fetch_projects(session: requests.Session, url: str, headers: dict[str, str]) -> list[dict]:
    out, page = [], 1
    while True:
        r = request_retry(session, "GET", f"{url}/api/projects", headers=headers,
                          params={"page": page, "page_size": 100}, timeout=60)
        r.raise_for_status()
        data = r.json()
        batch = data.get("results", data) if isinstance(data, dict) else data
        out.extend(batch)
        if not isinstance(data, dict) or not data.get("next"):
            break
        page += 1
    return out


def fetch_tasks(session: requests.Session, url: str, headers: dict[str, str], pid: int) -> list[dict]:
    try:
        r = request_retry(session, "GET", f"{url}/api/projects/{pid}/export", headers=headers,
                          params={"exportType": "JSON", "download_all_tasks": "true"}, timeout=300)
        if r.ok:
            return r.json()
    except requests.RequestException:
        pass
    r = request_retry(session, "GET", f"{url}/api/projects/{pid}/export", headers=headers,
                      params={"exportType": "JSON"}, timeout=300)
    r.raise_for_status()
    return r.json()


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/labels.json")
    ap.add_argument("--images-dir", type=Path, default=ROOT / "data/raw/panoramas")
    ap.add_argument("--no-images", action="store_true", help="не докачивать отсутствующие фото")
    ap.add_argument("--project", nargs="+", help="точные названия или ID проектов Label Studio")
    ap.add_argument("--all-projects", action="store_true", help="выгрузить все доступные проекты")
    ap.add_argument("--from-file", type=Path, help="сконвертировать готовый JSON-экспорт из браузера")
    ap.add_argument("--split", help="split по умолчанию, если в задачах он не указан")
    args = ap.parse_args()

    split_path = ROOT / "configs/split.json"
    split_map = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}

    if args.from_file:
        tasks = json.loads(args.from_file.read_text(encoding="utf-8"))
        labels = [x for t in tasks if (x := convert(t, args.split, split_map, args.images_dir))]
        total = len(tasks)
        download_tasks = []
    else:
        url, token = load_ls_env()
        session = requests.Session()
        headers = auth_headers(session, url, token)
        all_projects = fetch_projects(session, url, headers)
        by_title = {p["title"]: p for p in all_projects}
        by_id = {str(p["id"]): p for p in all_projects}

        if args.all_projects:
            chosen = all_projects
        elif args.project:
            chosen = [by_title[q] if q in by_title else by_id[q]
                      for q in args.project if q in by_title or q in by_id]
            unknown = [q for q in args.project if q not in by_title and q not in by_id]
            if unknown and chosen:
                print(f"Предупреждение: не найдены проекты {unknown}", file=sys.stderr)
        else:
            chosen = [by_title[t] for t in TITLES if t in by_title]
            missing = [t for t in TITLES if t not in by_title]
            if missing and chosen:
                print(f"Предупреждение: не найдены проекты {missing}", file=sys.stderr)
        if not chosen:
            avail = ", ".join(f"#{p['id']} «{p['title']}»" for p in all_projects) or "нет проектов"
            raise SystemExit(f"Не найдены нужные проекты. Доступные: {avail}")

        labels, total, download_tasks = [], 0, []
        for p in chosen:
            tasks = fetch_tasks(session, url, headers, p["id"])
            total += len(tasks)
            def_split = args.split or infer_split(p["title"])
            for t in tasks:
                item = convert(t, def_split, split_map, args.images_dir)
                if item:
                    labels.append(item)
                    download_tasks.append((item["file"], (t.get("data") or {}).get("image")))

        if not args.no_images and download_tasks:
            stats, failed = Counter(), []
            for fname, img_ref in download_tasks:
                if not img_ref:
                    continue
                try:
                    stats[download_image(session, url, headers, str(img_ref), args.images_dir / fname)] += 1
                except Exception as e:  # noqa: BLE001 — одна битая ссылка не должна срывать экспорт
                    failed.append((fname, str(e)[:120]))
            print(f"фото: было {stats['exists']}, скачано {stats['downloaded']}, ошибок {len(failed)} -> {args.images_dir}")
            for fname, err in failed[:5]:
                print(f"  не скачалось {fname}: {err}", file=sys.stderr)

    # При дублях одного файла между проектами оставляем последнюю запись
    dedup = {x["file"]: x for x in labels}
    labels = sorted(dedup.values(), key=lambda x: x["file"])
    write_json_atomic(args.out, labels)
    objs = [o for lab in labels for o in lab["objects"]]
    print(f"размечено {len(labels)} из {total} кадров: {Counter(lab['split'] for lab in labels)}")
    print("объекты:", Counter(o["type"] for o in objs))
    print("статусы:", Counter(o["status"] for o in objs), "| повреждения:", Counter(o["damage"] for o in objs if o["damage"]))
    print("плохие кадры:", sum("bad_frame" in lab["frame_flags"] for lab in labels))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
