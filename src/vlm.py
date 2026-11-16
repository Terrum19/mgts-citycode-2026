"""Вариант «только VLM»: одна модель находит объекты и оценивает их состояние за один запрос.

Работает с любым OpenAI-совместимым API (OpenRouter, mlx_vlm.server, llama.cpp).
"""
import base64
import json
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def env(name: str) -> str:
    if not name:
        return ""
    if os.environ.get(name):
        return os.environ[name]
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def build_prompt(tax: dict) -> str:
    types = "\n".join(f'- "{k}": {v}' for k, v in tax["object_types"].items())
    damage = "\n".join(f'- "{k}": {v}' for k, v in tax["damage"].items())
    return f"""Ты инспектор городской инфраструктуры. Найди на фото все объекты следующих типов:
{types}

Для каждого найденного объекта определи состояние:
- "ok" — объект исправен;
- "bad" — есть признак неудовлетворительного состояния. Тогда укажи вид повреждения ровно из списка:
{damage}

Не считай объектом ливневую решётку (решётку водостока).

Ответь ТОЛЬКО JSON без пояснений:
{{"objects": [{{"type": "<тип>", "bbox": [x1, y1, x2, y2], "status": "ok|bad", "damage": "<вид или null>", "description": "<одно предложение по-русски>"}}]}}
Координаты bbox — в относительной шкале 0..1000 по ширине и высоте изображения (0,0 — левый верхний угол).
Если объектов нет, верни {{"objects": []}}."""


def parse(text: str) -> list[dict] | None:
    """Достаёт список объектов из ответа модели; None, если JSON не разобрать."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    objs = data.get("objects") if isinstance(data, dict) else None
    return objs if isinstance(objs, list) else None


def normalize(objs: list[dict], w: int, h: int, tax: dict) -> list[dict]:
    """bbox 0..1000 -> пиксели; мусорные поля приводим к допустимым значениям."""
    out = []
    for o in objs:
        try:
            x1, y1, x2, y2 = (float(v) for v in o["bbox"])
        except (KeyError, TypeError, ValueError):
            continue
        status = "bad" if str(o.get("status", "")).lower() == "bad" else "ok"
        damage = o.get("damage") if status == "bad" else None
        if damage not in tax["damage"]:
            damage = None if status == "ok" else "unknown"
        out.append({
            "type": o.get("type", "unknown"),
            "bbox": [round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)],
            "status": status, "damage": damage, "description": o.get("description", ""),
        })
    return out


class VLM:
    def __init__(self, cfg: dict, tax: dict):
        self.cfg, self.tax = cfg, tax
        self.prompt = build_prompt(tax)
        self.key = env(cfg.get("key_env", ""))

    def call(self, image: Path) -> str:
        b64 = base64.b64encode(image.read_bytes()).decode()
        body = {
            "model": self.cfg["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": self.prompt},
            ]}],
            "temperature": 0,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        for attempt in range(5):
            r = requests.post(f"{self.cfg['base_url']}/chat/completions", json=body, headers=headers, timeout=180)
            if r.status_code in (429, 502, 503) and attempt < 4:
                time.sleep(5 * 2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data["choices"][0]["message"].get("content") or ""
        raise RuntimeError("retries exhausted")

    def predict(self, image: Path, w: int, h: int) -> dict:
        t0 = time.time()
        raw, objs, err = "", None, None
        for _ in range(2):  # второй шанс при битом JSON
            try:
                raw = self.call(image)
            except Exception as e:  # noqa: BLE001 — логируем любую ошибку API и идём дальше
                err = str(e)[:300]
                break
            objs = parse(raw)
            if objs is not None:
                break
        return {
            "objects": normalize(objs, w, h, self.tax) if objs is not None else [],
            "bad_json": objs is None and err is None,
            "error": err, "seconds": round(time.time() - t0, 2), "raw": raw[:2000],
        }
