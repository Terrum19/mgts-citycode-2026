"""Задачи для Label Studio + конфиг интерфейса разметки.

    python3 src/label/make_tasks.py                    # dev с авторазметкой (Qwen), test без неё
    python3 src/label/make_tasks.py --no-predict       # без вызовов модели (dev тоже пустой)

Выход:
  data/label/tasks_dev.json, data/label/tasks_test.json  — импортировать в два проекта Label Studio
  configs/label_config.xml                               — интерфейс разметки (классы из config.toml)
  data/label/autolabel_dev.json                          — сырые ответы модели (кэш: повторно не платим)
Картинки в задачах ссылаются на /data/local-files/?d=panoramas/<file> — так их отдаёт Label Studio с хаба.
"""
import argparse
import json
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vlm import VLM  # noqa: E402

PANO = ROOT / "data/raw/panoramas"
OUT = ROOT / "data/label"
W, H = 1280, 600
TYPE_RU = {"manhole": "колодец", "street_lamp": "опора", "street_cabinet": "шкаф"}


def label_config(tax: dict) -> str:
    colors = {"manhole": "#e6194b", "pole": "#3cb44b", "cabinet": "#4363d8", "aerial_equipment": "#911eb4"}
    labels = "\n".join(
        f'    <Label value="{k}" hint={quoteattr(v)} background="{colors.get(k, "#f58231")}" hotkey="{i}"/>'
        for i, (k, v) in enumerate(tax["object_types"].items(), 1))
    damage = "\n".join(f'      <Choice value="{k}" hint={quoteattr(v)}/>' for k, v in tax["damage"].items())
    return f"""<View>
  <Header value="Цель съёмки: $target_ru (по OSM; на кадре может быть не видна — размечай всё, что видно)"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="object" toName="image" strokeWidth="2">
{labels}
    <Label value="other" hint="другой объект инфраструктуры связи (ввод в здание, КИП-столбик…) — подпиши в комментарии" background="#f58231" hotkey="{len(tax['object_types']) + 1}"/>
  </RectangleLabels>
  <View visibleWhen="region-selected">
    <Header value="Состояние выделенного объекта"/>
    <Choices name="status" toName="image" perRegion="true" choice="single-radio" required="true" showInline="true">
      <Choice value="ok" hint="исправен"/>
      <Choice value="bad" hint="есть признак неудовлетворительного состояния"/>
      <Choice value="unclear" hint="не видно, чтобы решить"/>
    </Choices>
    <Choices name="damage" toName="image" perRegion="true" choice="single-radio"
             visibleWhen="choice-selected" whenTagName="status" whenChoiceValue="bad">
{damage}
      <Choice value="other_damage" hint="другое — опиши в комментарии"/>
    </Choices>
    <TextArea name="note" toName="image" perRegion="true" placeholder="комментарий (необязательно)" maxSubmissions="1"/>
  </View>
  <Choices name="frame" toName="image" choice="multiple" showInline="true">
    <Choice value="bad_frame" hint="размыто / засвечено / ничего не разобрать"/>
    <Choice value="target_not_visible" hint="объекта, на который целились, на кадре нет"/>
  </Choices>
</View>
"""


def to_prediction(objs: list[dict], model: str) -> dict:
    result = []
    for k, o in enumerate(objs):
        x1, y1, x2, y2 = o["bbox"]
        x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        rid = f"p{k}"
        common = {"id": rid, "to_name": "image", "original_width": W, "original_height": H, "image_rotation": 0}
        label = o["type"] if o["type"] in ("manhole", "pole", "cabinet", "aerial_equipment") else "other"
        result.append({**common, "from_name": "object", "type": "rectanglelabels", "value": {
            "x": 100 * x1 / W, "y": 100 * y1 / H, "width": 100 * (x2 - x1) / W, "height": 100 * (y2 - y1) / H,
            "rotation": 0, "rectanglelabels": [label]}})
        result.append({**common, "from_name": "status", "type": "choices", "value": {"choices": [o["status"]]}})
        if o["status"] == "bad":
            dmg = o["damage"] if o["damage"] and o["damage"] != "unknown" else "other_damage"
            result.append({**common, "from_name": "damage", "type": "choices", "value": {"choices": [dmg]}})
        if o.get("description"):
            result.append({**common, "from_name": "note", "type": "textarea", "value": {"text": [o["description"]]}})
    return {"model_version": model, "result": result}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="qwen235")
    ap.add_argument("--no-predict", action="store_true")
    args = ap.parse_args()
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "configs/label_config.xml").write_text(label_config(cfg["taxonomy"]))

    split = json.loads((ROOT / "configs/split.json").read_text())
    meta = {r["file"]: r for r in map(json.loads, (PANO / "meta.jsonl").read_text().splitlines()) if r.get("file")}

    cache_path = OUT / "autolabel_dev.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    dev_files = sorted(f for f, p in split.items() if p == "dev")
    todo = [f for f in dev_files if f not in cache]
    if todo and not args.no_predict:
        model = VLM(cfg["variants"][args.variant], cfg["taxonomy"])
        with ThreadPoolExecutor(cfg["variants"][args.variant].get("workers", 4)) as ex:
            for k, (f, p) in enumerate(zip(todo, ex.map(lambda f: model.predict(PANO / f, W, H), todo)), 1):
                cache[f] = p
                print(f"  {k}/{len(todo)} {f}: {'ERR ' + p['error'] if p['error'] else str(len(p['objects'])) + ' obj'}")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))

    for part in ("dev", "test"):
        tasks = []
        for f in sorted(x for x, p in split.items() if p == part):
            m = meta[f]
            task = {"data": {"image": f"/data/local-files/?d=panoramas/{f}", "file": f, "split": part,
                             "target_type": m["type"], "target_ru": TYPE_RU.get(m["type"], m["type"]),
                             "location_id": m["location_id"], "dist_m": round(m["dist"], 1)}}
            if part == "dev" and f in cache and not cache[f].get("error"):
                task["predictions"] = [to_prediction(cache[f]["objects"], cfg["variants"][args.variant]["model"])]
            tasks.append(task)
        (OUT / f"tasks_{part}.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=1))
        n_pred = sum(bool(t.get("predictions")) for t in tasks)
        print(f"{part}: {len(tasks)} задач, с авторазметкой {n_pred} -> data/label/tasks_{part}.json")
    print("-> configs/label_config.xml")


if __name__ == "__main__":
    main()
