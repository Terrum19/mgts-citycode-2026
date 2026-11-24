"""Прогон варианта на наборе + метрики.

    python src/run.py --variant qwen235 --data data/smoke/manholes
    python src/run.py --variant qwen235 gemma31_free --data data/smoke/manholes   # несколько подряд

Результат: runs/<набор>__<вариант>/preds.json и metrics.json, сводка в консоли.
"""
import argparse
import json
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate  # noqa: E402
from vlm import ROOT, VLM  # noqa: E402

KEYS = ["det_precision", "det_recall", "img_accuracy", "img_bad_recall", "img_bad_precision",
        "obj_bad_recall", "damage_acc", "bad_json_rate", "error_rate", "sec_per_image"]


def run_variant(name: str, cfg: dict, data: Path) -> dict:
    labels = json.loads((data / "labels.json").read_text())
    model = VLM(cfg["variants"][name], cfg["taxonomy"])
    out = ROOT / "runs" / f"{data.name}__{name}"
    out.mkdir(parents=True, exist_ok=True)

    def one(lab):
        return lab["file"], model.predict(data / lab["file"], lab["width"], lab["height"])

    preds = {}
    with ThreadPoolExecutor(cfg["variants"][name].get("workers", 2)) as ex:
        for k, (f, p) in enumerate(ex.map(one, labels), 1):
            preds[f] = p
            flag = "ERR" if p["error"] else ("BADJSON" if p["bad_json"] else f"{len(p['objects'])} obj")
            print(f"  [{name}] {k}/{len(labels)} {f}: {flag} {p['seconds']}s", file=sys.stderr)
    (out / "preds.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1))
    m = evaluate(labels, preds)
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    data = Path(args.data).resolve()

    results = {v: run_variant(v, cfg, data) for v in args.variant}
    print("\n" + "metric".ljust(18) + "".join(v[:16].rjust(17) for v in results))
    for k in KEYS:
        print(k.ljust(18) + "".join(f"{r[k]:17.2f}" for r in results.values()))
    for v, r in results.items():
        print(f"\n{v} confusion (gt -> pred):", json.dumps(r["confusion"], ensure_ascii=False))


if __name__ == "__main__":
    main()
