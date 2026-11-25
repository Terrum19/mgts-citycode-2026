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
from results import make_result, normalize_reasons
from taxonomy import taxonomy_fingerprint

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
    "- лужи, грязь, тени и дорожная разметка без чёткой крышки колодца (не считать их за «manhole»);\n"
    "- отдельные дорожные знаки и светофоры (стойки только со знаком, без светильника и без оборудования связи)."
)

NON_TARGETS_RECALL = (
    "- ливневые решётки (решётки водостока);\n"
    "- временные строительные и дорожные ограждения, перфорированные металлические щиты "
    "(в том числе секции с надписями вроде «КГХ»);\n"
    "- информационные стенды, навигационные стелы, рекламные щиты, урны и скамейки;\n"
    "- секции и прутья обычных заборов, бетонные вентшахты и киоски;\n"
    "- лужи, грязь, тени и дорожная разметка без чёткой крышки колодца (не считать их за «manhole»);\n"
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


def _bbox_spec(bbox_format: str) -> tuple[str, str]:
    if bbox_format not in BBOX_FORMATS:
        raise ValueError(f"неизвестный bbox_format={bbox_format!r}, допустимы {BBOX_FORMATS}")
    if bbox_format == "yxyx":
        return (
            "[ymin, xmin, ymax, xmax]",
            "Координаты bbox — строго в порядке [ymin, xmin, ymax, xmax] в относительной шкале 0..1000: "
            "сначала вертикаль y сверху вниз (ymin, ymax), затем горизонталь x слева направо (xmin, xmax); "
            "(0, 0) — левый верхний угол.",
        )
    return (
        "[x1, y1, x2, y2]",
        "Координаты bbox — строго в порядке [x1, y1, x2, y2] в относительной шкале 0..1000: "
        "x1, x2 по горизонтали (слева направо), y1, y2 по вертикали (сверху вниз); "
        "(0, 0) — левый верхний угол.",
    )


def build_prompt(tax: dict, bbox_format: str = "xyxy", prompt_mode: str = "default",
                 allow_review: bool = False, strict_bad: bool = True) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"неизвестный prompt_mode={prompt_mode!r}, допустимы {PROMPT_MODES}")
    types, damage = _lists(tax)
    bbox_example, bbox_note = _bbox_spec(bbox_format)
    scan = (
        " Внимательно осмотри весь кадр, включая обочины и газоны, и перечисли все видимые люки, опоры и шкафы."
        if prompt_mode == "recall" else ""
    )
    non_targets = NON_TARGETS_RECALL if prompt_mode == "recall" else NON_TARGETS
    schema_status = "ok|bad|review" if allow_review else "ok|bad"
    review_rule = (
        '\n- "review" — ТОЛЬКО если заметен подозрительный признак дефекта, который невозможно уверенно подтвердить; '
        'просто удалённый, мелкий или видимый сбоку объект без признаков повреждения — это "ok", а не "review".'
        if allow_review else ""
    )
    if strict_bad:
        bad_rule = (
            'Ставь "bad" только при явном дефекте; кабель в защитном кабель-канале или аккуратно закреплённая бухта на опоре '
            'без провисания и обрыва, а также заводские технологические стыки и швы декоративных чугунных кожухов у основания опоры — '
            'это "ok" (не "pole_damaged"). Лужи, грязь, тени и дорожная разметка без чёткой крышки колодца — '
            'не "manhole" и не "ring_or_surface_damaged".'
        )
    else:
        bad_rule = (
            'Заметное отслоение краски или следы вандализма на корпусе шкафа ("cabinet_damaged") и просадка или разрушение '
            'плитки/покрытия вокруг люка ("ring_or_surface_damaged") тоже считаются "bad". '
            'При этом кабель в защитном кабель-канале, аккуратно закреплённая бухта на опоре без провисания и заводские '
            'технологические стыки/швы декоративных чугунных кожухов у основания опоры — это "ok" (не "pole_damaged"). '
            'Лужи, грязь, тени и дорожная разметка без чёткой крышки колодца — не "manhole" и не "ring_or_surface_damaged".'
        )
    return f"""Ты инспектор городской инфраструктуры.{scan} Найди на фото все объекты следующих типов:
{types}

Для каждого найденного объекта определи состояние:
- "ok" — объект исправен;
- "bad" — есть признак неудовлетворительного состояния. Тогда укажи вид повреждения ровно из списка:
{damage}{review_rule}
{bad_rule}

Не считай объектами:
{non_targets}

Ответь ТОЛЬКО JSON без пояснений (если целевых объектов на кадре действительно нет, верни {{"objects": []}}):
{{"objects": [{{"type": "<тип>", "bbox": {bbox_example}, "status": "{schema_status}", "damage": "<вид или null>", "description": "<одно предложение по-русски>"}}]}}
{bbox_note}"""


def build_empty_retry_prompt(tax: dict, bbox_format: str = "xyxy", allow_review: bool = False) -> str:
    """Короткий повторный промпт при пустом ответе {"objects": []} на основном проходе."""
    types, damage = _lists(tax)
    bbox_example, bbox_note = _bbox_spec(bbox_format)
    schema_status = "ok|bad|review" if allow_review else "ok|bad"
    return f"""На предыдущем проходе список объектов оказался пустым. Перепроверь фото внимательно: точно ли нет ни одной опоры освещения/связи, кабельного люка на тротуаре/дороге или уличного шкафа?
Не выдумывай объекты — если кадр действительно пуст или на нём только дорожные знаки/светофоры/ограждения/лужи, верни {{"objects": []}}.
Если хотя бы один целевой объект есть, верни его в JSON:
Типы:
{types}
Виды повреждений (при "bad"):
{damage}
Ответь ТОЛЬКО JSON без пояснений:
{{"objects": [{{"type": "<тип>", "bbox": {bbox_example}, "status": "{schema_status}", "damage": "<вид или null>", "description": "<одно предложение по-русски>"}}]}}
{bbox_note}"""


def build_crop_prompt(tax: dict, type_hint: str, prompt_mode: str = "default",
                      allow_review: bool = False) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"неизвестный prompt_mode={prompt_mode!r}, допустимы {PROMPT_MODES}")
    types, damage = _lists(tax)
    if allow_review:
        status_rule = (
            '- status: "ok" — исправен, "bad" — есть явный признак неудовлетворительного состояния,\n'
            '  "review" — ТОЛЬКО если заметен подозрительный признак дефекта, который невозможно уверенно подтвердить\n'
            '  (удалённый, мелкий или видимый сбоку объект без признаков повреждения — это "ok").'
        )
    else:
        status_rule = (
            '- status: "ok" — исправен, "bad" — есть признак неудовлетворительного состояния;\n'
            '  при сомнении не ставь "bad" без явного признака.'
        )
    schema_status = "ok|bad|review" if allow_review else "ok|bad"
    damage_tail = 'при "review" и "ok" — null' if allow_review else "иначе null"
    return f"""Ты инспектор городской инфраструктуры. Это вырез из кадра с камеры автомобиля.
Детектор считает, что в центре выреза объект типа "{type_hint}". Типы объектов:
{types}

Определи:
- is_object: true, если в центре действительно объект одного из типов выше (ливневая решётка, временное
  строительное ограждение или щит, информационный стенд, забор, дорожный знак, светофор, рекламный щит — это false);
- type: тип объекта из списка;
{status_rule}
- damage: при "bad" — вид повреждения ровно из списка, {damage_tail}:
{damage}
  (аккуратный кабель-канал или закреплённая без провисания бухта на опоре — это "ok");
- description: одно предложение по-русски о состоянии объекта.

Ответь ТОЛЬКО JSON без пояснений:
{{"is_object": true, "type": "<тип>", "status": "{schema_status}", "damage": "<вид или null>", "description": "..."}}"""


def build_context_crop_prompt(tax: dict, type_hint: str, has_crop: bool = True,
                              prompt_mode: str = "default", allow_review: bool = False) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"неизвестный prompt_mode={prompt_mode!r}, допустимы {PROMPT_MODES}")
    types, damage = _lists(tax)
    if allow_review:
        status_rule = (
            '- status: "ok" — исправен, "bad" — есть явный признак неудовлетворительного состояния;\n'
            '  "review" — ТОЛЬКО если заметен подозрительный признак дефекта, который невозможно уверенно подтвердить '
            '(удалённый или мелкий объект без признаков повреждения — это "ok").'
        )
    else:
        status_rule = (
            '- status: "ok" — исправен, "bad" — есть явный и серьёзный признак неудовлетворительного состояния '
            '(не ставь "bad" из-за низкого разрешения, артефактов сжатия, потёртостей крышки или лёгкого перспективного угла).'
        )
    schema_status = "ok|bad|review" if allow_review else "ok|bad"
    damage_tail = 'при "review" и "ok" — null' if allow_review else "иначе null"
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
- is_object: true, если внутри яркой рамки действительно объект одного из типов выше. Ставь false для ливневой решётки, временного строительного ограждения или щита, информационного стенда, забора, столбика ограждения, тумбы, урны, рекламного щита, лужи/дорожной разметки без крышки колодца, а также отдельной стойки только с дорожным знаком или пешеходным светофором (без уличного светильника и без оборудования/кабелей связи). Если на фонарном столбе или опоре связи дополнительно висит знак или светофор — это true ("pole").
- type: тип объекта из списка;
{status_rule}
  Правила против ложных срабатываний:
  • Наклон опоры/шкафа ("pole_tilted", "cabinet_tilted") ставь только при сильном, явном отклонении относительно вертикали зданий и соседних столбов на полном кадре — перспективный завал у краёв широкоугольной панорамы или лёгкий наклон съёмки это "ok".
  • Для колодца ("manhole") оценивай только крышку и непосредственное кольцо/покрытие вокруг неё: явный провал или разрушенное покрытие вокруг люка — это "ring_or_surface_damaged", но не путай соседнюю ливневую решётку, ровные стыки плитки, тени, лужи, грязь или дорожную разметку с дефектом ("ok").
  • Заводские технологические стыки и швы декоративных чугунных кожухов у основания опоры — это "ok", не "pole_damaged".
  • Аккуратный кабель-канал или закреплённая без провисания бухта на опоре — это "ok".
- damage: при "bad" — вид повреждения ровно из списка, {damage_tail}:
{damage}
- description: одно предложение по-русски о состоянии объекта.

Ответь ТОЛЬКО JSON без пояснений:
{{"is_object": true, "type": "<тип>", "status": "{schema_status}", "damage": "<вид или null>", "description": "..."}}"""


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


def normalize(objs: list, w: int, h: int, tax: dict, bbox_format: str = "xyxy",
              *, invalid_bbox_objects: list | None = None) -> list[dict]:
    """bbox 0..1000 -> пиксели; объект без рамки/неизвестного типа отбрасывается.

    Некорректная рамка не роняет кадр: она попадает в invalid_bbox_objects. Валидная
    рамка с неизвестным статусом остаётся в выдаче как review (не превращается в ok).
    """
    if bbox_format not in BBOX_FORMATS:
        raise ValueError(f"неизвестный bbox_format={bbox_format!r}, допустимы {BBOX_FORMATS}")
    out = []
    for index, o in enumerate(objs or []):
        if not isinstance(o, dict) or o.get("type") not in tax["object_types"]:
            continue
        raw_box = o.get("bbox") if "bbox" in o else o.get("box_2d")
        try:
            a, b, c, d = (float(v) for v in raw_box)
        except (KeyError, TypeError, ValueError):
            if invalid_bbox_objects is not None:
                invalid_bbox_objects.append({"index": index, "bbox": raw_box, "reason": "not_four_numbers"})
            continue
        if not all(isfinite(v) for v in (a, b, c, d)):
            if invalid_bbox_objects is not None:
                invalid_bbox_objects.append({"index": index, "bbox": raw_box, "reason": "non_finite"})
            continue
        x1, y1, x2, y2 = (b, a, d, c) if bbox_format == "yxyx" else (a, b, c, d)
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        bx1, by1, bx2, by2 = round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)
        if bx2 <= bx1 or by2 <= by1:
            if invalid_bbox_objects is not None:
                invalid_bbox_objects.append({"index": index, "bbox": raw_box, "reason": "degenerate"})
            continue
        status, damage = clean_state(o, tax)
        item = {
            "type": o["type"],
            "bbox": [bx1, by1, bx2, by2],
            "status": status, "damage": damage, "description": str(o.get("description") or ""),
            "review_reasons": normalize_reasons(["low_confidence"]) if status == "review" else [],
        }
        out.append(item)
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
        if self.prompt_mode not in PROMPT_MODES:
            raise ValueError(f"неизвестный prompt_mode={self.prompt_mode!r}, допустимы {PROMPT_MODES}")
        self.allow_review = bool(cfg.get("allow_review", False))
        self.retry_empty = bool(cfg.get("retry_empty", False))
        self.strict_bad = bool(cfg.get("strict_bad", True))
        self.max_tokens = int(cfg.get("max_tokens", 2000))
        self.timeout = cfg.get("timeout", 60)
        self.prompt = build_prompt(tax, self.bbox_format, self.prompt_mode,
                                   allow_review=self.allow_review, strict_bad=self.strict_bad)
        self.empty_retry_prompt = build_empty_retry_prompt(tax, self.bbox_format, allow_review=self.allow_review)
        self.crop_prompt = build_crop_prompt(tax, "", self.prompt_mode, allow_review=self.allow_review)
        self.context_prompt = build_context_crop_prompt(tax, "", has_crop=True, prompt_mode=self.prompt_mode,
                                                        allow_review=self.allow_review)
        self.context_prompt_no_crop = build_context_crop_prompt(tax, "", has_crop=False,
                                                                prompt_mode=self.prompt_mode,
                                                                allow_review=self.allow_review)
        self.key = env(cfg.get("key_env", ""))
        self.max_side = cfg.get("max_side", 1600)
        self.context_max_side = cfg.get("context_max_side", self.max_side)
        self.crop_max_side = cfg.get("crop_max_side", self.max_side)

    def fingerprint(self) -> str:
        """SHA-256 смыслового запроса: смена модели/промпта/размера/таксономии обновляет кэш."""
        payload = {
            "cache_schema": 2,
            "model": self.cfg.get("model"),
            "base_url": self.cfg.get("base_url"),
            "provider": self.cfg.get("provider"),
            "bbox_format": self.bbox_format,
            "prompt_mode": self.prompt_mode,
            "allow_review": self.allow_review,
            "retry_empty": self.retry_empty,
            "strict_bad": self.strict_bad,
            "max_side": self.max_side,
            "context_max_side": self.context_max_side,
            "crop_max_side": self.crop_max_side,
            "max_tokens": self.max_tokens,
            "taxonomy": taxonomy_fingerprint(self.tax),
            "prompt": self.prompt,
            "empty_retry_prompt": self.empty_retry_prompt if self.retry_empty else "",
            "crop_prompt": self.crop_prompt,
            "context_prompt": self.context_prompt,
            "context_prompt_no_crop": self.context_prompt_no_crop,
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

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
            "max_tokens": self.max_tokens,
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
                    json=body, headers=headers, timeout=self.timeout,
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
        jpeg = to_jpeg(img, self.max_side)
        r = self._ask(jpeg, self.prompt, parse)
        objs = r.pop("value")
        cost = float(r.get("cost") or 0.0)
        seconds = float(r.get("seconds") or 0.0)
        error = r.get("error")
        bad_json = bool(r.get("bad_json"))
        if self.retry_empty and error is None and not bad_json and objs == []:
            r2 = self._ask(jpeg, self.empty_retry_prompt, parse)
            cost += float(r2.get("cost") or 0.0)
            seconds = round(seconds + float(r2.get("seconds") or 0.0), 2)
            error = r2.get("error")
            bad_json = bool(r2.get("bad_json"))
            if r2.get("value") is not None:
                objs = r2["value"]
        invalid: list = []
        objects = normalize(objs, w, h, self.tax, self.bbox_format, invalid_bbox_objects=invalid) if objs is not None else []
        return make_result(objects=objects, scope="mixed", invalid_bbox_objects=invalid,
                           error=error, bad_json=bad_json, seconds=seconds, cost=cost)

    def classify_crop(self, crop: Image.Image, type_hint: str) -> dict:
        prompt = build_crop_prompt(self.tax, type_hint, self.prompt_mode, allow_review=self.allow_review)
        r = self._ask(to_jpeg(crop, self.crop_max_side), prompt, parse_json)
        r["answer"] = r.pop("value")
        return r

    def classify_crop_in_context(self, context_img: Image.Image, crop_img: Image.Image | None,
                                 type_hint: str) -> dict:
        """Оценка находки детектора на полном кадре с выделенной рамкой (+ опциональный крупный вырез)."""
        images = [to_jpeg(context_img, self.context_max_side)]
        if crop_img is not None:
            images.append(to_jpeg(crop_img, self.crop_max_side))
        prompt = build_context_crop_prompt(self.tax, type_hint, has_crop=crop_img is not None,
                                           prompt_mode=self.prompt_mode, allow_review=self.allow_review)
        r = self._ask(images, prompt, parse_json)
        r["answer"] = r.pop("value")
        return r
