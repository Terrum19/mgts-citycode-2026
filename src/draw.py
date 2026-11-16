"""Сетка картинок: разметка зелёным, предсказание красным (подпись — вид повреждения).

    python src/draw.py data/smoke/manholes runs/manholes__qwen235/preds.json runs/manholes__qwen235/sheet.jpg

Нужен только ffmpeg (Pillow не обязателен).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SHORT = {None: "ok", "cover_broken": "broken", "cover_missing": "missing",
         "cover_open_or_shifted": "open", "ring_or_surface_damaged": "ring", "unknown": "?"}


def boxes(objs, color, y_text):
    out = []
    for o in objs:
        x1, y1, x2, y2 = o["bbox"]
        out.append(f"drawbox=x={x1}:y={y1}:w={max(1, x2 - x1)}:h={max(1, y2 - y1)}:color={color}:t=5")
    tag = ",".join(SHORT.get(o.get("damage"), str(o.get("damage"))) for o in objs) or "-"
    out.append(f"drawtext=text='{tag}':x=8:y={y_text}:fontsize=28:fontcolor={color}:box=1:boxcolor=black@0.6")
    return out


def main() -> None:
    data, preds_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    labels = json.loads((data / "labels.json").read_text())
    preds = json.loads(preds_path.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        for k, lab in enumerate(labels):
            p = preds.get(lab["file"], {"objects": []})
            vf = ",".join(boxes(lab["objects"], "lime", 8) + boxes(p["objects"], "red", 44) + ["scale=320:320"])
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(data / lab["file"]), "-vf", vf,
                            f"{tmp}/{k:03d}.jpg"], check=True)
        cols = 6
        rows = -(-len(labels) // cols)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{tmp}/%03d.jpg", "-vf",
                        f"tile={cols}x{rows}:padding=4:color=white", "-frames:v", "1", str(out)], check=True)
    print(out)


if __name__ == "__main__":
    main()
