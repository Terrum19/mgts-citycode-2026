"""Погодные аугментации для обучения: свет, цвет, смаз, шум, лёгкий JPEG.

Зачем: на финале и в эксплуатации съёмка идёт с движущейся машины в ноябрьской Москве —
низкий контраст, мокрый асфальт, слепящие фары, сумерки, смаз. Без аугментаций линейная голова
на 5-10 примерах редкого дефекта «запоминает» дневное освещение и цвет конкретного асфальта.

Правила (сознательные ограничения):
  * только PIL/numpy — без новых зависимостей;
  * геометрия кадра не меняется: ни поворотов, ни отражений, ни обрезки. Сильный поворот ломает
    признак `pole_tilted` (наклон опоры), перевёрнутый люк на дороге не встречается;
  * вызывать ТОЛЬКО для train-выборки, после сплита. Валидация и тест считаются на оригиналах,
    иначе метрики завышаются;
  * всё детерминировано: один и тот же (seed, cfg) даёт один и тот же результат.

    from augment import augment, n_copies, sample_seed
    copy = augment(crop, cfg["augment"], sample_seed(seed, "photo.jpg:3", 0))
"""
import io
import random
import zlib

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Границы по умолчанию (их же дублирует [augment] в config.toml). Умеренные: задача — расширить
# разнообразие освещения, а не уничтожить мелкую фактуру дефекта.
DEFAULTS = {
    "brightness": [0.75, 1.25],
    "contrast": [0.75, 1.25],
    "saturation": [0.70, 1.30],
    "color_gain": 0.08,
    "blur_prob": 0.20,
    "blur_radius": [0.3, 1.0],
    "motion_prob": 0.20,
    "motion_length": [3, 9],
    "motion_angle": [-30.0, 30.0],
    "noise_prob": 0.25,
    "noise_std": [2.0, 8.0],
    "jpeg_prob": 0.20,
    "jpeg_quality": [60, 90],
}


def settings(cfg: dict | None) -> dict:
    """Параметры аугментации: недостающие ключи берём из DEFAULTS, лишние игнорируем."""
    out = dict(DEFAULTS)
    for key, value in (cfg or {}).items():
        if key in DEFAULTS and value is not None:
            out[key] = value
    return out


def sample_seed(base: int, index, copy: int = 0) -> int:
    """Стабильный seed для копии: не зависит от PYTHONHASHSEED и порядка запуска."""
    return (int(base) + zlib.crc32(f"{index}:{copy}".encode())) & 0x7FFFFFFF


def _uniform(rng: random.Random, value) -> float:
    if isinstance(value, (list, tuple)):
        return rng.uniform(float(value[0]), float(value[-1]))
    return float(value)


def _randint(rng: random.Random, value) -> int:
    if isinstance(value, (list, tuple)):
        return rng.randint(int(value[0]), int(value[-1]))
    return int(value)


def _shift(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Сдвиг с повтором крайних пикселей (без заворота кадра)."""
    height, width = array.shape[:2]
    ys = np.clip(np.arange(height) - dy, 0, height - 1)
    xs = np.clip(np.arange(width) - dx, 0, width - 1)
    return array[ys][:, xs]


def motion_blur(img: Image.Image, length: int, angle: float) -> Image.Image:
    """Направленный 1D смаз — имитация скорости машины и капель на стекле."""
    length = max(1, int(round(length)))
    if length <= 1:
        return img
    array = np.asarray(img, dtype=np.float32)
    rad = np.deg2rad(float(angle))
    dx, dy = np.cos(rad), np.sin(rad)
    total = np.zeros_like(array)
    offsets = [k - (length - 1) / 2 for k in range(length)]
    for offset in offsets:
        total += _shift(array, int(round(dx * offset)), int(round(dy * offset)))
    return Image.fromarray(np.clip(total / len(offsets), 0, 255).astype(np.uint8))


def _color_gain(img: Image.Image, amount: float, rng: np.random.Generator) -> Image.Image:
    """Небольшой сдвиг баланса белого: жёлтый свет фонарей, серый снег, синие сумерки."""
    if amount <= 0:
        return img
    gains = np.array([1 + rng.uniform(-amount, amount) for _ in range(3)], dtype=np.float32)
    array = np.asarray(img, dtype=np.float32) * gains
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as rounded:
        return rounded.convert("RGB")


def augment(img: Image.Image, cfg: dict | None, seed: int) -> Image.Image:
    """Одна погодная копия кропа/фото: RGB, исходный размер, uint8.

    Порядок как у реальной деградации кадра: свет и цвет -> оптический смаз -> шум сенсора ->
    сжатие видеорегистратора.
    """
    rng = random.Random(int(seed))
    nrng = np.random.default_rng(int(seed))
    params = settings(cfg)
    out = img.convert("RGB")

    out = ImageEnhance.Brightness(out).enhance(_uniform(rng, params["brightness"]))
    out = ImageEnhance.Contrast(out).enhance(_uniform(rng, params["contrast"]))
    out = ImageEnhance.Color(out).enhance(_uniform(rng, params["saturation"]))
    out = _color_gain(out, float(params["color_gain"]), nrng)

    if rng.random() < float(params["blur_prob"]):
        out = out.filter(ImageFilter.GaussianBlur(_uniform(rng, params["blur_radius"])))
    if rng.random() < float(params["motion_prob"]):
        out = motion_blur(out, _randint(rng, params["motion_length"]), _uniform(rng, params["motion_angle"]))

    if rng.random() < float(params["noise_prob"]):
        sigma = _uniform(rng, params["noise_std"])
        array = np.asarray(out, dtype=np.float32) + nrng.normal(0.0, sigma, (out.height, out.width, 3))
        out = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    if rng.random() < float(params["jpeg_prob"]):
        out = _jpeg(out, _randint(rng, params["jpeg_quality"]))
    return out


def n_copies(status: str, original_support: int, cfg: dict | None) -> int:
    """Сколько аугментированных копий нужно объекту.

    support — число ОРИГИНАЛОВ класса (не копий): редкий дефект аугментируется сильнее, чтобы
    сгладить дисбаланс. `augment_rare_bad_copies` — общее число копий, а не добавка сверху.
    """
    if not cfg or not cfg.get("augment"):
        return 0
    copies = max(0, int(cfg.get("augment_copies", 1)))
    rare = max(0, int(cfg.get("augment_rare_bad_copies", copies)))
    threshold = int(cfg.get("augment_rare_threshold", 20))
    if status == "bad" and int(original_support) < threshold:
        return rare
    return copies
