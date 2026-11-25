"""VGCopy-paste: вставка дефектного объекта на реальный уличный фон (резерв при дефиците класса).

    python3 src/poc/copy_paste.py --defects data/copy_paste/defects.json \
        --backgrounds data/raw/panoramas --out data/synthetic/copy_paste --count 100 --seed 42

Зачем: на финале редкого дефекта может быть 1-5 фото. Из пары реальных объектов получаются
десятки разнообразных кадров с готовыми bbox для обучения быстрого классификатора/детектора.

Вход `defects.json` — список доноров; пути от каталога манифеста:
    [{"file": "donors/broken_01.jpg", "mask": "donors/broken_01_mask.png",
      "type": "manhole", "status": "bad", "damage": "cover_broken"}]
Маска — PNG того же размера, что donor; ненулевые пиксели задают объект. Автосегментации нет:
контур готовит человек (или SAM), здесь только вставка.

Выход: OUT/images/NNNNNN.jpg + OUT/labels.json в формате, который понимает `crop_dino.py`
(`[{"file", "objects": [{bbox, type, status, damage}]}]`).

Что делаем: вырез по маске, масштаб от ширины фона, умеренная перспектива, случайное валидное
положение, feathering краёв, подгонка средней яркости/цвета под локальный фон, bbox по фактической
маске. Чего не делаем: поиск семантически правильного места, перекрытия, тени, обучение генератора.

Главный риск (почему инструмент «на крайний случай»): при грубой вклейке классификатор выучивает
шов как признак поломки, а не сам дефект. Поэтому синтетика не подмешивается в train автоматически,
а качество проверяется только на настоящих невклеенных фото.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_defects(path: Path) -> list[dict]:
    """Манифест доноров -> записи с абсолютными путями к фото и маске."""
    path = Path(path)
    base = path.resolve().parent
    rows = json.loads(path.read_text(encoding="utf-8"))
    defects = []
    for row in rows:
        donor = (base / row["file"]).resolve()
        mask = (base / row["mask"]).resolve()
        for item in (donor, mask):
            if not item.exists():
                raise FileNotFoundError(f"нет файла донора: {item}")
        defects.append({**row, "donor": donor, "mask": mask})
    return defects


def background_list(path: Path) -> list[Path]:
    """Все фото в каталоге фонов (по умолчанию — уже собранные панорамы)."""
    path = Path(path)
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        raise ValueError(f"в {path} нет фото для фона")
    return files


def cut(donor: Path, mask: Path) -> Image.Image:
    """Вырез объекта по маске -> RGBA, обрезанный до ненулевой области маски."""
    with Image.open(donor) as source, Image.open(mask) as alpha_source:
        rgb = source.convert("RGB")
        alpha = alpha_source.convert("L")
        if alpha.size != rgb.size:
            raise ValueError(f"маска {mask} не совпадает по размеру с {donor}: {alpha.size} != {rgb.size}")
        binary = alpha.point(lambda v: 255 if v >= 128 else 0)
        box = binary.getbbox()
        if box is None:
            raise ValueError(f"пустая маска: {mask}")
        rgba = rgb.convert("RGBA")
        rgba.putalpha(binary)
        return rgba.crop(box)


def _coeffs(src: list, dst: list) -> list[float]:
    """Коэффициенты PIL PERSPECTIVE: отображение точки результата (dst) в исходник (src)."""
    rows, values = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)
    return np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64)).tolist()


def perspective(rgba: Image.Image, rng: random.Random, strength: float = 0.08) -> Image.Image:
    """Умеренное перспективное искажение: люк внизу кадра сплющен в эллипс, вдалеке — мельче."""
    width, height = rgba.size
    dx, dy = strength * width, strength * height
    source = [(0, 0), (width, 0), (width, height), (0, height)]
    shifted = [(rng.uniform(0, dx), rng.uniform(0, dy)),
               (width - rng.uniform(0, dx), rng.uniform(0, dy)),
               (width - rng.uniform(0, dx), height - rng.uniform(0, dy)),
               (rng.uniform(0, dx), height - rng.uniform(0, dy))]
    return rgba.transform((width, height), Image.PERSPECTIVE, _coeffs(source, shifted),
                          resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))


def match_colors(piece: Image.Image, background: Image.Image, position: tuple[int, int]) -> Image.Image:
    """Подогнать средний цвет/яркость объекта под локальный фон (ограниченными gain/offset)."""
    x, y = position
    array = np.asarray(piece, dtype=np.float32)
    rgb, alpha = array[:, :, :3], array[:, :, 3]
    region = np.asarray(background.crop((x, y, x + piece.width, y + piece.height)), dtype=np.float32)
    mask = alpha > 8
    if not mask.any():
        return piece
    donor_mean = rgb[mask].mean(axis=0)
    background_mean = region[mask].mean(axis=0)
    gain = np.clip(background_mean / np.maximum(donor_mean, 1.0), 0.7, 1.4)
    offset = np.clip(background_mean - gain * donor_mean, -30, 30)
    adjusted = np.clip(rgb * gain + offset, 0, 255)
    return Image.fromarray(np.dstack([adjusted, alpha]).astype(np.uint8), "RGBA")


def feather(piece: Image.Image, ratio: float = 0.04) -> Image.Image:
    """Мягкие края: резкая граница вклейки — это тот самый шов, который нельзя выучить.

    Кроп по маске непрозрачен целиком, поэтому сначала добавляем прозрачные поля: иначе
    размытие постоянной alpha ничего не сглаживает.
    """
    radius = max(1.0, min(piece.size) * ratio)
    margin = int(np.ceil(radius * 3))
    canvas = Image.new("RGBA", (piece.width + 2 * margin, piece.height + 2 * margin), (0, 0, 0, 0))
    canvas.paste(piece, (margin, margin))
    canvas.putalpha(canvas.getchannel("A").filter(ImageFilter.GaussianBlur(radius)))
    return canvas


def compose(background: Image.Image, piece: Image.Image, rng: random.Random, *,
            scale: tuple[float, float] = (0.06, 0.18), y_range: tuple[float, float] = (0.30, 0.95),
            strength: float = 0.08) -> tuple[Image.Image, list[int] | None]:
    """Вставить объект в кадр -> (кадр, bbox в пикселях кадра)."""
    width, height = background.size
    target_width = max(8, int(round(width * rng.uniform(*scale))))
    ratio = target_width / piece.width
    target_height = max(8, int(round(piece.height * ratio)))
    piece = piece.resize((target_width, target_height), Image.BICUBIC)
    piece = feather(perspective(piece, rng, strength))

    x = rng.randint(0, max(0, width - piece.width))
    low = min(int(height * y_range[0]), max(0, height - piece.height))
    high = min(max(low, int(height * y_range[1]) - piece.height), max(low, height - piece.height))
    y = rng.randint(low, high)

    piece = match_colors(piece, background, (x, y))
    out = background.copy()
    out.paste(piece, (x, y), piece)
    box = piece.getchannel("A").point(lambda v: 255 if v >= 8 else 0).getbbox()
    if box is None:
        return out, None
    return out, [max(0, x + box[0]), max(0, y + box[1]),
                 min(width, x + box[2]), min(height, y + box[3])]


def generate(defects: list[dict], backgrounds: list[Path], out: Path, count: int, seed: int = 42, *,
             scale: tuple[float, float] = (0.06, 0.18), strength: float = 0.08) -> list[dict]:
    """Сгенерировать count кадров и labels.json -> манифест (как у crop_dino.py)."""
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pieces = [(cut(row["donor"], row["mask"]), row) for row in defects]
    if not pieces:
        raise ValueError("нет доноров")
    backgrounds = list(backgrounds)
    rows = []
    for index in range(count):
        piece, meta = pieces[rng.randrange(len(pieces))]
        with Image.open(backgrounds[rng.randrange(len(backgrounds))]) as source:
            background = source.convert("RGB")
        image, bbox = compose(background, piece.copy(), rng, scale=scale, strength=strength)
        if bbox is None:
            continue
        name = f"images/{index:06d}.jpg"
        image.save(out / name, quality=92)
        rows.append({"file": name, "objects": [{"bbox": bbox, "type": meta.get("type"),
                                                "status": meta.get("status", "bad"),
                                                "damage": meta.get("damage")}]})
    (out / "labels.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="VGCopy-paste: вставка дефектов на реальные фоны")
    ap.add_argument("--defects", required=True, help="JSON доноров {file, mask, type, status, damage}")
    ap.add_argument("--backgrounds", required=True, help="каталог фонов (например, data/raw/panoramas)")
    ap.add_argument("--out", required=True, help="каталог результата: images/ + labels.json")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale-min", type=float, default=0.06, help="доля ширины кадра, нижняя граница")
    ap.add_argument("--scale-max", type=float, default=0.18)
    ap.add_argument("--perspective", type=float, default=0.08, help="амплитуда перспективы, доля размера")
    args = ap.parse_args()

    defects = read_defects(Path(args.defects))
    backgrounds = background_list(Path(args.backgrounds))
    rows = generate(defects, backgrounds, Path(args.out), args.count, args.seed,
                    scale=(args.scale_min, args.scale_max), strength=args.perspective)
    print(f"{len(rows)} кадров, {len(defects)} доноров, {len(backgrounds)} фонов -> {args.out}/labels.json")
    print("Валидировать и оценивать — только на настоящих фото: шов вклейки не должен стать признаком.")


if __name__ == "__main__":
    sys.exit(main())
