"""Перепроверка разметки sim_final вслепую и пересчёт метрик связки на очищенных метках.

    python src/label/review_sim.py make     # выбрать фото, залить на хаб, создать проект в Label Studio
    python src/label/review_sim.py eval     # забрать ответы, пересчитать F1 до/после

На проверку идут: все ошибки связки (DINOv2-base 448 + logreg + 0.3*Qwen), фото, где и классификатор,
и Qwen не согласны с меткой, и контроль — 60 случайных фото, где все согласны (по нему видно,
сколько шума осталось среди непроверенных). Исходная метка проверяющему НЕ показывается.
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/poc"))
from classify import groups_of, is_ok  # noqa: E402
from folders import load  # noqa: E402
from sweep import cv_probs, feats, score  # noqa: E402

DATA = ROOT / "data/sim_final"
QWEN = ROOT / "runs/cls/sim_final__vlm_qwen235.json"
MAP = ROOT / "data/label/sim_review.json"
HUB, MEDIA = "companion@10.180.132.80", "mgts/media/sim_review"
TITLE = "МГТС: перепроверка sim_final (вслепую)"
SKIP = ["Непонятно", "Не тот объект / мусор"]
ENV = dict(re.findall(r"^(LABEL_STUDIO_\w+)=(.*)$", (ROOT / ".env").read_text(), re.M))
URL, H = ENV["LABEL_STUDIO_URL"], {"Authorization": "Token " + ENV["LABEL_STUDIO_TOKEN"]}


def api(method, path, **kw):
    r = requests.request(method, URL + path, headers=H, timeout=120, **kw)
    if r.status_code >= 400:
        raise SystemExit(f"{method} {path}: {r.status_code} {r.text[:300]}")
    return r.json() if r.text else {}


def predict(items, labels):
    """Связка на CV: возвращает (предсказания классификатора, Qwen, связки)."""
    y = np.array(labels, dtype=object)
    C = np.array(sorted(set(y)), dtype=object)
    P = cv_probs(feats("dinov2-base:448", [DATA / x["file"] for x in items]), y, groups_of(items), 32.0)
    ans = {r["file"]: r["label"] for r in json.loads(QWEN.read_text())}
    Q = np.zeros_like(P)
    for i, x in enumerate(items):
        if ans.get(x["file"]) in C:
            Q[i, list(C).index(ans[x["file"]])] = 1
    qw = np.array([ans.get(x["file"]) for x in items], dtype=object)
    return C[P.argmax(1)], qw, C[(P + 0.3 * Q).argmax(1)]


def make():
    items = load(DATA)
    y = np.array([x["label"] for x in items], dtype=object)
    cls, qw, comb = predict(items, y)
    suspect = [i for i in range(len(y)) if comb[i] != y[i] or (cls[i] != y[i] and qw[i] != y[i])]
    agree = [i for i in range(len(y)) if cls[i] == y[i] and qw[i] == y[i] and comb[i] == y[i]]
    rnd = random.Random(42)
    control = rnd.sample(agree, 60)
    chosen = suspect + control
    rnd.shuffle(chosen)
    mapping = [{"id": f"r{k:04d}.jpg", "file": items[i]["file"], "label": y[i],
                "control": i in control} for k, i in enumerate(chosen)]
    MAP.write_text(json.dumps(mapping, ensure_ascii=False, indent=1))
    print(f"на проверку {len(chosen)}: подозрительных {len(suspect)}, контроль {len(control)}")

    tmp = ROOT / "data/label/sim_review_imgs"
    tmp.mkdir(exist_ok=True)
    for m in mapping:
        dst = tmp / m["id"]
        if not dst.exists():
            dst.symlink_to(DATA / m["file"])
    subprocess.run(["ssh", "-o", "BatchMode=yes", HUB, f"mkdir -p {MEDIA}"], check=True)
    subprocess.run(["rsync", "-aL", f"{tmp}/", f"{HUB}:{MEDIA}/"], check=True)

    classes = sorted(set(y)) + SKIP
    choices = "\n".join(f'    <Choice value="{c}" hotkey="{k}"/>' for k, c in enumerate(classes, 1))
    config = f"""<View>
  <Header value="Что на фото? Выбери один класс (клавиши 1–{len(classes)}). Если дефектов несколько — самый серьёзный."/>
  <Image name="image" value="$image" zoom="true"/>
  <Choices name="label" toName="image" choice="single-radio" required="true">
{choices}
  </Choices>
</View>"""
    existing = {p["title"]: p for p in api("GET", "/api/projects", params={"page_size": 100})["results"]}
    body = {"title": TITLE, "label_config": config, "maximum_annotations": 1,
            "description": "Перепроверка меток вслепую: смотрим фото и выбираем правильный класс."}
    if TITLE in existing:
        pid = existing[TITLE]["id"]
        api("PATCH", f"/api/projects/{pid}", json=body)
    else:
        pid = api("POST", "/api/projects", json=body)["id"]
        api("POST", "/api/storages/localfiles", json={
            "project": pid, "title": "sim_review", "path": f"/home/companion/{MEDIA}", "use_blob_urls": False})
        api("POST", f"/api/projects/{pid}/import", json=[
            {"data": {"image": f"/data/local-files/?d=sim_review/{m['id']}", "rid": m["id"]}} for m in mapping])
    print(f"проект #{pid}: {URL}/projects/{pid}")


def eval_():
    mapping = {m["id"]: m for m in json.loads(MAP.read_text())}
    pid = {p["title"]: p for p in api("GET", "/api/projects", params={"page_size": 100})["results"]}[TITLE]["id"]
    tasks = api("GET", f"/api/projects/{pid}/export", params={"exportType": "JSON"})
    review = {}
    for t in tasks:
        res = [r for a in t.get("annotations", []) if not a.get("was_cancelled") for r in a["result"]]
        if res:
            review[mapping[t["data"]["rid"]]["file"]] = res[0]["value"]["choices"][0]
    print(f"проверено {len(review)} из {len(mapping)}")
    by_file = {m["file"]: m for m in mapping.values()}
    for part, name in ((False, "подозрительные"), (True, "контроль")):
        done = [f for f, m in by_file.items() if m["control"] == part and f in review]
        if done:
            changed = sum(review[f] != by_file[f]["label"] for f in done)
            print(f"  {name}: проверено {len(done)}, метка изменена у {changed} ({changed / len(done):.0%})")

    items = load(DATA)
    total = len(items)
    old = [x["label"] for x in items]
    keep = [i for i, x in enumerate(items) if review.get(x["file"]) not in SKIP]
    items = [items[i] for i in keep]
    new = [review.get(x["file"], x["label"]) for x in items]
    old = [old[i] for i in keep]
    ok_of = {lab: is_ok(lab.split(" - ", 1)[-1]) for lab in set(new) | set(old)} | {"?": True}
    print(f"выкинуто как непонятные/мусор: {total - len(items)}")

    # 1) старая модель (обучена на шумных метках), оценка по чистым меткам
    cls, qw, comb = predict(items, old)
    yn = np.array(new, dtype=object)
    print("\nобучение на старых метках, проверка по очищенным:")
    for n, p in (("классификатор", cls), ("Qwen", qw), ("связка", comb)):
        print(f"  {n:14}", {k: round(v, 3) for k, v in score(yn, np.array([v or "?" for v in p], dtype=object), ok_of).items()})
    # 2) обучение и проверка на очищенных
    cls, qw, comb = predict(items, new)
    print("обучение и проверка на очищенных:")
    for n, p in (("классификатор", cls), ("Qwen", qw), ("связка", comb)):
        print(f"  {n:14}", {k: round(v, 3) for k, v in score(yn, np.array([v or "?" for v in p], dtype=object), ok_of).items()})
    print("\nвнимание: проверяли в основном ошибки модели, поэтому цифры на очищенных метках слегка завышены;"
          " насколько — видно по доле изменённых меток в контроле.")


if __name__ == "__main__":
    {"make": make, "eval": eval_}[sys.argv[1]]()
