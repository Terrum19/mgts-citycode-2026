"""VLM без обучения: выбрать класс (папку) из списка и написать описание.
Если даны примеры описаний похожих фото (из Excel организаторов), модель пишет в их стиле.

    python src/vlm_classify.py data/sim_final --variant qwen235 --per-class 20   # сравнение с classify.py
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folders import load  # noqa: E402
from run import load_config  # noqa: E402
from vlm import VLM, parse_json, to_jpeg  # noqa: E402


def build_prompt(labels: list[str], hint: str | None = None, examples: list[str] | None = None,
                 prompt_mode: str = "default") -> str:
    if prompt_mode not in ("default", "recall"):
        raise ValueError(f"неизвестный prompt_mode={prompt_mode!r}")
    opts = "\n".join(f"{k}. {lab}" for k, lab in enumerate(labels, 1))
    p = f"""Ты инспектор сетевой инфраструктуры. На фото — объект городской инфраструктуры.
Выбери ровно один класс (тип объекта и состояние) из списка:
{opts}
"""
    if isinstance(hint, str):
        p += f"\nКлассификатор считает, что это «{hint}». Проверь и исправь, если он ошибся.\n"
    elif hint:  # [[класс, вероятность], ...] — переранжирование
        top = "\n".join(f"- {lab}: {pr:.0%}" for lab, pr in hint)
        p += ("\nКлассификатор, обученный на размеченных фото (верен примерно в 70% случаев), "
              f"даёт самые вероятные классы:\n{top}\nПосмотри на фото внимательно и выбери правильный класс; "
              "обычно это один из этих, но можно выбрать и другой из списка.\n")
    if examples:
        p += "\nПримеры описаний экспертов для похожих фото (пиши в таком же стиле):\n" + \
             "\n".join(f"- {e}" for e in examples) + "\n"
    if prompt_mode == "recall":
        p += """
Если ни один класс нельзя выбрать надёжно, верни class=null и review=true.
Если объект исправен — выбирай класс с исправным состоянием, не ставь дефект «на всякий случай»."""
    p += """
Ответь ТОЛЬКО JSON: {"class": <номер класса или null>, "review": false, "review_reason": null,
"description": "<1–2 предложения по-русски: что за объект и в чём дефект>"}"""
    return p


def classify(vlm: VLM, img: Image.Image, labels: list[str], hint=None, examples=None) -> dict:
    prompt = build_prompt(labels, hint, examples, prompt_mode=getattr(vlm, "prompt_mode", "default"))
    r = vlm._ask(to_jpeg(img, vlm.max_side), prompt, parse_json)
    a = r.pop("value") or {}
    try:
        k = int(a.get("class")) - 1
        label = labels[k] if 0 <= k < len(labels) else None
    except (TypeError, ValueError):
        label = None
    review = bool(a.get("review"))
    error = r.get("error")
    if label is None and not review and not error:
        # Схема не позволяет молча выбрать первый класс: отсутствие класса без review — ошибка.
        error = "class=null без review"
    return {**r, "error": error, "label": label, "review": review or label is None,
            "review_reason": a.get("review_reason"),
            "description": str(a.get("description") or "")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--variant", default="qwen235")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--sep")
    ap.add_argument("--hints", help="metrics.json от classify.py eval (поле cv_pred) или top3.json от sweep.py --dump")
    args = ap.parse_args()
    hints = json.loads(Path(args.hints).read_text()) if args.hints else {}
    hints = hints.get("cv_pred", hints)
    cfg = load_config()
    root = Path(args.root)
    items = load(root, args.sep)
    labels = sorted({x["label"] for x in items})
    random.seed(0)
    sample = [x for lab in labels for x in random.sample([i for i in items if i["label"] == lab],
                                                         min(args.per_class, sum(i["label"] == lab for i in items)))]
    vlm = VLM(cfg["variants"][args.variant], cfg["taxonomy"])
    t0 = time.time()
    with ThreadPoolExecutor(cfg["variants"][args.variant].get("workers", 4)) as ex:
        res = list(ex.map(lambda x: classify(vlm, Image.open(root / x["file"]), labels, hints.get(x["file"])), sample))
    acc = sum(r["label"] == x["label"] for r, x in zip(res, sample)) / len(sample)
    if hints:
        top1 = {f: (h if isinstance(h, str) else h[0][0]) for f, h in hints.items()}
        h_acc = sum(top1[x["file"]] == x["label"] for x in sample) / len(sample)
        print(f"классификатор на той же выборке: {h_acc:.3f}")
    print(f"{args.variant}{' + подсказка' if hints else ''}: {len(sample)} фото, точность {acc:.3f}, сек/фото {sum(r['seconds'] for r in res) / len(res):.1f}, "
          f"${sum(r['cost'] for r in res):.3f}, битых ответов {sum(r['label'] is None for r in res)}, всего {time.time() - t0:.0f} с")
    print("частые ошибки:", *Counter(f"{x['label']} -> {r['label']}" for r, x in zip(res, sample)
                                      if r["label"] != x["label"]).most_common(8), sep="\n  ")
    out = Path("runs/cls") / f"{root.name}__vlm_{args.variant}{'_' + Path(args.hints).stem if hints else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{"file": x["file"], "gt": x["label"], **r} for x, r in zip(sample, res)],
                              ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
