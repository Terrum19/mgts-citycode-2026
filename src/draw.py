"""Рисование рамок: зелёная — исправен, красная — повреждён (подпись — вид повреждения).
Если передана разметка, она рисуется жёлтым пунктиром — для разбора ошибок.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OK, BAD, GT = (40, 200, 70), (230, 40, 40), (255, 210, 0)
_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
          "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"]


def font(size: int):
    for f in _FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def _dashed(d: ImageDraw.ImageDraw, box, color, width, dash=8):
    x1, y1, x2, y2 = box
    for a, b, horiz, c in ((x1, x2, True, y1), (x1, x2, True, y2), (y1, y2, False, x1), (y1, y2, False, x2)):
        for s in range(int(a), int(b), dash * 2):
            e = min(s + dash, b)
            d.line((s, c, e, c) if horiz else (c, s, c, e), fill=color, width=width)


def annotate(img: Image.Image, objects: list[dict], gt: list[dict] | None = None, names: dict | None = None) -> Image.Image:
    """names: {код -> подпись} для типов и повреждений (по умолчанию — коды)."""
    names = names or {}
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    lw = max(2, round(max(img.size) / 400))
    f = font(max(12, round(max(img.size) / 70)))
    for g in gt or []:
        _dashed(d, g["bbox"], GT, lw)
        tag = g["type"] + ("" if g["status"] == "ok" else f" [{g.get('damage') or g['status']}]")
        d.text((g["bbox"][0], g["bbox"][3] + 2), tag, fill=GT, font=f, stroke_width=2, stroke_fill="black")
    for k, o in enumerate(objects, 1):
        color = BAD if o["status"] == "bad" else OK
        d.rectangle(o["bbox"], outline=color, width=lw)
        tag = f"{k}. {names.get(o['type'], o['type'])}"
        if o["status"] == "bad":
            tag += ": " + names.get(o["damage"], str(o["damage"]))
        x, y = o["bbox"][0], max(0, o["bbox"][1] - f.size - 4)
        d.text((x, y), tag, fill=color, font=f, stroke_width=2, stroke_fill="black")
    return img
