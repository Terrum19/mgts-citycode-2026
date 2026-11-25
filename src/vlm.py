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
from math import isfinite
from pathlib import Path

import requests
from PIL import Image

from cascade import clean_state  # noqa: F401 — оставляем прежний импорт from vlm import clean_state

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


BBOX_FORMATS = ("xyxy", "yxyx")
PROMPT_MODES = ("default", "recall")

NON_TARGETS = (
    "- ливневые решётки (решётки водостока);\n"
    "- временные строительные и дорожные ограждения, перфорированные металлические щиты "
    "(в том числе секции с надписями вроде «КГХ»);\n"
    "- информационные стенды, навигационные стелы, рекламные щиты, урны и скамейки;\n"
    "- секции и прутья обычных заборов, бетонные вентшахты и киоски;\n"
    "- отдельные дорожные знаки и светофоры (стойки только со знаком, без светильника и без оборудования связи)."
)

NON_TARGETS_RECALL = (
    "- ливневые решётки (решётки водостока);\n"
    "- временные строительные и дорожные ограждения, перфорированные металлические щиты "
    "(в том числе секции с надписями вроде «КГХ»);\n"
    "- информационные стенды, навигационные стелы, рекламные щиты, урны и скамейки;\n"
    "- секции и прутья обычных заборов, бетонные вентшахты и киоски;\n"
    "- отдельные дорожные знаки и светофоры (стойки только со знаком или пешеходным светофором, без уличного фонаря наверху и без оборудования/кабелей связи); "
    "если на фонарном столбе или опоре связи дополнительно висит знак или светофор — это «pole»."
)


def bbox_format_for(cfg: dict) -> str:
    """Порядок координат рамки: для Google Gemma/Gemini — [ymin, xmin, ymax, xmax], для остальных — [x1, y1, x2, y2]."""
    fmt = cfg.get("bbox_format")
    if fmt is not None:
        if fmt not in BBOX_FORMATS:
            raise ValueError(f"неизвестный bbox_format={fmt!r}, допустимы {BBOX_FORMATS}")
        return fmt
    model = (cfg.get("model") or "").lower()
    if "gemma" in model or "gemini" in model:
        return "yxyx"
    return "xyxy"


def build_prompt(tax: dict, bbox_format: str = "xyxy", prompt_mode: str = "default") -> str:
    if bbox_format not in BBOX_FORMATS:
        raise ValueError(f"неизвестный bbox_format={bbox_format!r}, допустимы {BBOX_FORMATS}")
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"неизвестный prompt_mode={prompt_mode!r}, допустимы {PROMPT_MODES}")
    types, damage = _lists(tax)
    if bbox_format == "yxyx":
        bbox_example = "[ymin, xmin, ymax, xmax]"
        bbox_note = (
            "Координаты bbox — строго в порядке [ymin, xmin, ymax, xmax] в относительной шкале 0..1000: "
            "сначала вертикаль y сверху вниз (ymin, ymax), затем горизонталь x слева направо (xmin, xmax); "
            "(0, 0) — левый верхний угол."
        )
    else:
        bbox_example = "[x1, y1, x2, y2]"
        bbox_note = (
            "Координаты bbox — строго в порядке [x1, y1, x2, y2] в относительной шкале 0..1000: "
            "x1, x2 по горизонтали (слева направо), y1, y2 по вертикали (сверху вниз); "
            "(0, 0) — левый верхний угол."
        )
    scan = " Внимательно осмотри весь кадр, включая обочины и газоны." if prompt_mode == "recall" else ""
    non_targets = NON_TARGETS_RECALL if prompt_mode == "recall" else NON_TARGETS
    return f"""Ты инспектор городской инфраструктуры.{scan} Найди на фото все объекты следующих типов:
{types}

Для каждого найденного объекта определи состояние:
- "ok" — объект исправен;
- "bad" — есть признак неудовлетворительного состояния. Тогда укажи вид повреждения ровно из списка:
{damage}
Ставь "bad" только при явном дефекте; кабель в защитном кабель-канале или аккуратно закреплённая бухта на опоре без провисания и обрыва — это "ok".

Не считай объектами:
{non_targets}

Ответь ТОЛЬКО JSON без пояснений:
{{"objects": [{{"type": "<тип>", "bbox": {bbox_example}, "status": "ok|bad", "damage": "<вид или null>", "description": "<одно предложение по-русски>"}}]}}
{bbox_note}
Если объектов нет, верни {{"objects": []}}."""


def build_crop_prompt(tax: dict, type_hint: str) -> str:
    types, damage = _lists(tax)
    return f"""Ты инспектор городской инфраструктуры. Это вырез из кадра с камеры автомобиля.
Детектор считает, что в центре выреза объект типа "{type_hint}". Типы объектов:
{types}

Определи:
- is_object: true, если в центре действительно объект одного из типов выше (ливневая решётка, временное
  строительное ограждение или щит, информационный стенд, забор, дорожный знак, светофор, рекламный щит — это false);
- type: тип объекта из списка;
- status: "ok" — исправен, "bad" — есть признак неудовлетворительного состояния;
- damage: при "bad" — вид повреждения ровно из списка, иначе null:
{damage}
  (аккуратный кабель-канал или закреплённая без провисания бухта на опоре — это "ok");
- description: одно предложение по-русски о состоянии объекта.

Ответь ТОЛЬКО JSON без пояснений:
{{"is_object": true, "type": "<тип>", "status": "ok|bad", "damage": "<вид или null>", "description": "..."}}"""


def build_context_crop_prompt(tax: dict, type_hint: str, has_crop: bool = True) -> str:
    types, damage = _lists(tax)
    view_note = (
        "На первом изображении показан полный кадр с камеры автомобиля, целевой объект обведён яркой пурпурной рамкой. "
        "На втором изображении — увеличенный вырез этого объекта. Оценивай ТОЛЬКО объект внутри яркой рамки "
        "(полный кадр используй для понимания типа, вертикали и окружения, а вырез — для деталей состояния)."
        if has_crop else
        "На изображении показан полный кадр с камеры автомобиля, целевой объект обведён яркой пурпурной рамкой. "
        "Оценивай ТОЛЬКО объект внутри яркой рамки."
    )
    return f"""Ты инспектор городской инфраструктуры. {view_note}
Детектор считает, что в яркой рамке объект типа "{type_hint}". Типы объектов:
{types}

Определи:
- is_object: true, если внутри яркой рамки действительно объект одного из типов выше. Ставь false для ливневой решётки, временного строительного ограждения или щита, информационного стенда, забора, столбика ограждения, тумбы, урны, рекламного щита, а также отдельной стойки только с дорожным знаком или пешеходным светофором (без уличного светильника и без оборудования/кабелей связи). Если на фонарном столбе или опоре связи дополнительно висит знак или светофор — это true ("pole").
- type: тип объекта из списка;
- status: "ok" — исправен, "bad" — есть признак неудовлетворительного состояния.
  Правила против ложных срабатываний:
  • Наклон опоры/шкафа ("pole_tilted", "cabinet_tilted") ставь только при реальном отклонении относительно вертикали зданий и соседних столбов на полном кадре — перспективный завал у краёв широкоугольной панорамы это "ok".
  • Для колодца ("manhole") оценивай только крышку и непосредственное кольцо вокруг неё: не путай соседнюю ливневую решётку, стыки плитки, тени или траву вокруг люка с "ring_or_surface_damaged" или "cover_open_or_shifted".
  • Аккуратный кабель-канал или закреплённая без провисания бухта на опоре — это "ok".
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


def normalize(objs: list, w: int, h: int, tax: dict, bbox_format: str = "xyxy") -> list[dict]:
    """bbox 0..1000 -> пиксели; объекты неизвестного типа или без рамки отбрасываются."""
    if bbox_format not in BBOX_FORMATS:
        raise ValueError(f"неизвестный bbox_format={bbox_format!r}, допустимы {BBOX_FORMATS}")
    out = []
    for o in objs:
        if not isinstance(o, dict) or o.get("type") not in tax["object_types"]:
            continue
        raw_box = o.get("bbox") if "bbox" in o else o.get("box_2d")
        try:
            a, b, c, d = (float(v) for v in raw_box)
        except (KeyError, TypeError, ValueError):
            continue
        if not all(isfinite(v) for v in (a, b, c, d)):
            continue
        x1, y1, x2, y2 = (b, a, d, c) if bbox_format == "yxyx" else (a, b, c, d)
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        bx1, by1, bx2, by2 = round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)
        if bx2 <= bx1 or by2 <= by1:
            continue
        status, damage = clean_state(o, tax)
        out.append({
            "type": o["type"],
            "bbox": [bx1, by1, bx2, by2],
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


OPENROUTER_QUANTIZATIONS = ["bf16", "fp16", "fp8", "mxfp8", "int8", "fp32"]


class VLM:
    def __init__(self, cfg: dict, tax: dict):
        self.cfg, self.tax = cfg, tax
        self.bbox_format = bbox_format_for(cfg)
        self.prompt_mode = cfg.get("prompt_mode", "default")
        self.prompt = build_prompt(tax, self.bbox_format, self.prompt_mode)
        self.key = env(cfg.get("key_env", ""))
        self.max_side = cfg.get("max_side", 1600)

    def fingerprint(self) -> str:
        """Меняется при смене модели, формата bbox или таксономии/промпта — кэш по нему понимает, что старые ответы не годятся."""
        text = self.cfg["model"] + self.bbox_format + self.prompt + build_crop_prompt(self.tax, "")
        return hashlib.sha1(text.encode()).hexdigest()[:10]

    def _body(self, jpeg: bytes | list[bytes] | tuple[bytes, ...], prompt: str) -> dict:
        images = [jpeg] if isinstance(jpeg, (bytes, bytearray)) else list(jpeg)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(im).decode()}"}}
            for im in images
        ]
        content.append({"type": "text", "text": prompt})
        body = {
            "model": self.cfg["model"],
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": self.cfg.get("max_tokens", 4000),
            "response_format": {"type": "json_object"},
        }
        if "openrouter.ai" in self.cfg["base_url"]:
            body["usage"] = {"include": True}
            # Берём самого дешёвого провайдера, но отсекаем урезанные 4-bit (int4/fp4) и unlabelled квантования.
            provider = self.cfg.get("provider")
            if provider is None:
                provider = {"sort": "price", "allow_fallbacks": True}
                if not self.cfg["model"].endswith(":free"):
                    provider["quantizations"] = list(OPENROUTER_QUANTIZATIONS)
            if provider:
                body["provider"] = provider
        return body

    def call(self, jpeg: bytes | list[bytes] | tuple[bytes, ...], prompt: str) -> tuple[str, float]:
        """-> (текст ответа, стоимость в $; 0, если API её не сообщает)."""
        body = self._body(jpeg, prompt)
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        attempts = int(self.cfg.get("retries", 5))
        base_delay = float(self.cfg.get("retry_delay", 3.0))
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                r = requests.post(
                    f"{self.cfg['base_url']}/chat/completions",
                    json=body, headers=headers, timeout=self.cfg.get("timeout", 180),
                )
                if r.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                    time.sleep(base_delay * 2 ** attempt)
                    continue
                if not r.ok:
                    try:
                        msg = r.json().get("error") or r.text
                    except ValueError:
                        msg = r.text
                    raise requests.HTTPError(f"{r.status_code}: {msg}", response=r)
                data = r.json()
            except requests.HTTPError:
                raise
            except requests.RequestException as e:
                last_err = e
                if attempt < attempts - 1:
                    time.sleep(base_delay * 2 ** attempt)
                    continue
                raise
            if "error" in data:
                raise RuntimeError(data["error"])
            cost = float((data.get("usage") or {}).get("cost") or 0)
            return data["choices"][0]["message"].get("content") or "", cost
        raise RuntimeError(f"retries exhausted: {last_err}")

    def _ask(self, jpeg: bytes | list[bytes] | tuple[bytes, ...], prompt: str, extract) -> dict:
        """Запрос с одним повтором при битом JSON."""
        t0, raw, val, err, cost = time.time(), "", None, None, 0.0
        for _ in range(2):
            try:
                raw, c = self.call(jpeg, prompt)
                cost += c
            except Exception as e:  # noqa: BLE001 — любую ошибку API записываем и идём дальше
                err = str(e)[:300]
                break
            val = extract(raw)
            if val is not None:
                break
        return {"value": val, "bad_json": val is None and err is None, "error": err,
                "seconds": round(time.time() - t0, 2), "cost": cost, "raw": raw[:2000]}

    def predict(self, img: Image.Image) -> dict:
        w, h = img.size
        r = self._ask(to_jpeg(img, self.max_side), self.prompt, parse)
        objs = r.pop("value")
        r["objects"] = normalize(objs, w, h, self.tax, self.bbox_format) if objs is not None else []
        return r

    def classify_crop(self, crop: Image.Image, type_hint: str) -> dict:
        r = self._ask(to_jpeg(crop, self.max_side), build_crop_prompt(self.tax, type_hint), parse_json)
        r["answer"] = r.pop("value")
        return r

    def classify_crop_in_context(self, context_img: Image.Image, crop_img: Image.Image | None,
                                 type_hint: str) -> dict:
        """Оценка находки детектора на полном кадре с выделенной рамкой (+ опциональный крупный вырез)."""
        images = [to_jpeg(context_img, self.max_side)]
        if crop_img is not None:
            images.append(to_jpeg(crop_img, self.max_side))
        prompt = build_context_crop_prompt(self.tax, type_hint, has_crop=crop_img is not None)
        r = self._ask(images, prompt, parse_json)
        r["answer"] = r.pop("value")
        return r
