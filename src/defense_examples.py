import json
from pathlib import Path


def load_examples():
    directory = Path(__file__).resolve().parents[1] / "demo" / "inspections"
    records = json.loads((directory / "predictions.json").read_text(encoding="utf-8"))
    return [(directory / record["filename"], record["prediction"]) for record in records]
