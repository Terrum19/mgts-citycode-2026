"""Клиент VLM с OpenAI-совместимым API (OpenRouter, mlx_vlm.server, llama.cpp) и промпты.

Два режима:
  predict(img)               — вариант «только VLM»: модель сама находит объекты и оценивает их за один запрос;
  classify_crop(crop, type)  — вариант 1: детектор уже нашёл объект, VLM оценивает вырез.
"""
import base64
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import requests
from PIL import Image

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


def _lists(tax: dict) -> tuple[str, str]:
    types = "\n".join(f'- "{k}": {v}' for k, v in tax["object_types"].items())
    damage = "\n".join(f'- "{k}": {v}' for k, v in tax["damage"].items())
    return types, damage


def build_prompt(tax: dict) -> str:
    types, damage = _lists(tax)
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


def build_crop_prompt(tax: dict, type_hint: str) -> str:
    types, damage = _lists(tax)
    return f"""Ты инспектор городской инфраструктуры. Это вырез из кадра с камеры автомобиля.
Детектор считает, что в центре выреза объект типа "{type_hint}". Типы объектов:
{types}

Определи:
- is_object: true, если в центре действительно объект одного из типов выше (ливневая решётка, дорожный знак,
  светофор, рекламный щит — это false);
- type: тип объекта из списка;
- status: "ok" — исправен, "bad" — есть признак неудовлетворительного состояния;
- damage: при "bad" — вид повреждения ровно из списка, иначе null:
{damage}
- description: одно предложение по-русски о состоянии объекта.

Ответь ТОЛЬКО JSON без пояснений:
{{"is_object": true, "type": "<тип>", "status": "ok|bad", "damage": "<вид или null>", "description": "..."}}"""


def parse_json(text: str) -> dict | None:
    """Первый JSON-объект из ответа модели; None, если не разобрать."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse(text: str) -> list[dict] | None:
    data = parse_json(text)
    objs = data.get("objects") if data else None
    return objs if isinstance(objs, list) else None


def clean_state(o: dict, tax: dict) -> tuple[str, str | None]:
    """status/damage к допустимым значениям: bad без вида из списка -> damage "unknown"."""
    status = "bad" if str(o.get("status", "")).lower() == "bad" else "ok"
    damage = o.get("damage") if status == "bad" else None
    if status == "bad" and damage not in tax["damage"]:
        damage = "unknown"
    return status, damage


def normalize(objs: list, w: int, h: int, tax: dict) -> list[dict]:
    """bbox 0..1000 -> пиксели; объекты неизвестного типа или без рамки отбрасываются."""
    out = []
    for o in objs:
        if not isinstance(o, dict) or o.get("type") not in tax["object_types"]:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in o["bbox"])
        except (KeyError, TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        status, damage = clean_state(o, tax)
        out.append({
            "type": o["type"],
            "bbox": [round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)],
            "status": status, "damage": damage, "description": str(o.get("description") or ""),
        })
    return out


def to_jpeg(img: Image.Image, max_side: int) -> bytes:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


class VLM:
    def __init__(self, cfg: dict, tax: dict):
        self.cfg, self.tax = cfg, tax
        self.prompt = build_prompt(tax)
        self.key = env(cfg.get("key_env", ""))
        self.max_side = cfg.get("max_side", 1600)
        self.retries = cfg.get("retries", 5)
        if not isinstance(self.retries, int) or isinstance(self.retries, bool) or self.retries < 1:
            raise ValueError(f"retries должен быть целым >= 1, получено {self.retries!r}")

    def fingerprint(self) -> str:
        """Меняется при смене модели или таксономии/промпта — кэш по нему понимает, что старые ответы не годятся."""
        text = self.cfg["model"] + self.prompt + build_crop_prompt(self.tax, "")
        return hashlib.sha1(text.encode()).hexdigest()[:10]

    def call(self, jpeg: bytes, prompt: str) -> tuple[str, float, int]:
        """-> (текст ответа, стоимость в $; 0, если API её не сообщает; число HTTP-попыток)."""
        b64 = base64.b64encode(jpeg).decode()
        body = {
            "model": self.cfg["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0,
            "max_tokens": self.cfg.get("max_tokens", 4000),
            "response_format": {"type": "json_object"},
        }
        if "openrouter.ai" in self.cfg["base_url"]:
            body["usage"] = {"include": True}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        attempts = 0
        for attempt in range(self.retries):
            attempts += 1
            r = requests.post(f"{self.cfg['base_url']}/chat/completions", json=body, headers=headers,
                              timeout=self.cfg.get("timeout", 180))
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            cost = float((data.get("usage") or {}).get("cost") or 0)
            return data["choices"][0]["message"].get("content") or "", cost, attempts
        raise RuntimeError("retries exhausted")

    def _ask(self, jpeg: bytes, prompt: str, extract) -> dict:
        """Запрос с одним повтором при битом JSON."""
        t0, raw, val, err, cost, attempts = time.time(), "", None, None, 0.0, 0
        for _ in range(2):
            try:
                raw, c, a = self.call(jpeg, prompt)
                cost += c
                attempts += a
            except Exception as e:  # noqa: BLE001 — любую ошибку API записываем и идём дальше
                err = str(e)[:300]
                break
            val = extract(raw)
            if val is not None:
                break
        return {"value": val, "bad_json": val is None and err is None, "error": err,
                "seconds": round(time.time() - t0, 2), "cost": cost, "attempts": attempts, "raw": raw[:2000]}

    def predict(self, img: Image.Image) -> dict:
        w, h = img.size
        r = self._ask(to_jpeg(img, self.max_side), self.prompt, parse)
        objs = r.pop("value")
        r["objects"] = normalize(objs, w, h, self.tax) if objs is not None else []
        return r

    def classify_crop(self, crop: Image.Image, type_hint: str) -> dict:
        r = self._ask(to_jpeg(crop, self.max_side), build_crop_prompt(self.tax, type_hint), parse_json)
        r["answer"] = r.pop("value")
        return r
