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


STATUS_RU = {"ok": "НОРМА", "bad": "ПОВРЕЖДЁН", "review": "НА ПРОВЕРКЕ"}


def frame_banner(img: Image.Image, frame: dict, names: dict) -> Image.Image:
    """Плашка кадрового результата. Рамку на всё изображение не рисуем: локализация не выполнялась."""
    objects = frame.get("objects") or []
    first = objects[0] if objects else {}
    status = str(first.get("status") or frame.get("status") or "review")
    typ = names.get(first.get("type"), first.get("type") or "")
    damage = names.get(first.get("damage"), first.get("damage") or "") if status == "bad" else ""
    asset = frame.get("asset") or {}
    assessment = frame.get("assessment") or {}
    color = {"bad": BAD, "ok": OK}.get(status, GT)

    lines = [" · ".join(x for x in (f"КАДР: {STATUS_RU.get(status, status.upper())}", typ, damage) if x)]
    if asset.get("asset_id"):
        lines[0] = f"{asset['asset_id']} · " + lines[0]
    if assessment.get("recommendation"):
        lines.append(f"{assessment['recommendation']} · {assessment.get('urgency', '')}".strip(" ·"))
    lines.append("Локализация объекта не выполнялась")

    img = img.convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    f = font(max(14, round(max(img.size) / 45)))
    step = f.size + 6
    d.rectangle((0, 0, img.width, step * len(lines) + 8), fill=(0, 0, 0, 165))
    for i, line in enumerate(lines):
        d.text((6, 4 + i * step), line, fill=color if i == 0 else (255, 255, 255), font=f)
    return img


def annotate(img: Image.Image, objects: list[dict], gt: list[dict] | None = None, names: dict | None = None,
             frame: dict | None = None) -> Image.Image:
    """names: {код -> подпись} для типов и повреждений (по умолчанию — коды).
    frame: результат уровня кадра — тогда рисуется плашка, а рамки не имитируются."""
    names = names or {}
    if frame is not None and frame.get("scope") == "frame":
        return frame_banner(img, frame, names)
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    lw = max(2, round(max(img.size) / 400))
    f = font(max(12, round(max(img.size) / 70)))
    for g in gt or []:
        _dashed(d, g["bbox"], GT, lw)
        tag = g["type"] + ("" if g["status"] == "ok" else f" [{g.get('damage') or g['status']}]")
        d.text((g["bbox"][0], g["bbox"][3] + 2), tag, fill=GT, font=f, stroke_width=2, stroke_fill="black")
    for k, o in enumerate(objects, 1):
        status = o["status"]
        color = {"bad": BAD, "ok": OK}.get(status, GT)
        box = o["bbox"]
        if box is not None:
            d.rectangle(box, outline=color, width=lw)
        tag = f"{k}. {names.get(o['type'], o['type'])}"
        if o["status"] == "bad":
            tag += ": " + names.get(o["damage"], str(o["damage"]))
        if status == "review":
            tag += " · на проверку"
        x, y = (4, 4) if box is None else (box[0], max(0, box[1] - f.size - 4))
        d.text((x, y), tag, fill=color, font=f, stroke_width=2, stroke_fill="black")
    return img
