"""Классификатор «тип + дефект» по папкам: признаки DINOv2 + логистическая регрессия.

    python src/classify.py eval  data/sim_final            # перекрёстная проверка: точность, F1, ошибки
    python src/classify.py train data/sim_final --name sim # обучить на всём -> weights/cls_sim.joblib

Класс = папка. «Исправным» считается класс, где дефект похож на «норма/исправен/без дефектов» —
для метрики «нашли ли все дефектные объекты».
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import ROOT, Embedder  # noqa: E402
from folders import load  # noqa: E402

OK_WORDS = ("норм", "исправ", "без дефект", "без поврежд", "нет дефект", "ok", "good", "normal")


def is_ok(defect: str) -> bool:
    normalized = defect.lower()
    return "неисправ" not in normalized and any(w in normalized for w in OK_WORDS)


def make_clf(C: float = 2.0):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced")


def groups_of(items: list[dict]) -> list[str]:
    return [str(x["extra"].get("Группа", x["file"])) for x in items]


def evaluate(items, X, y, folds: int, C: float) -> dict:
    """Перекрёстная проверка с группами (повторы одного кадра не попадают в train и val одновременно)."""
    labels = sorted(set(y))
    pred = np.empty(len(y), dtype=object)
    knn_pred = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    fit_sec = []
    for tr, va in cv.split(X, y, groups_of(items)):
        t0 = time.time()
        clf = make_clf(C).fit(X[tr], y[tr])
        fit_sec.append(time.time() - t0)
        pred[va] = clf.predict(X[va])
        sim = X[va] @ X[tr].T                         # kNN по косинусу: для сравнения и для поиска похожих
        top = np.argsort(-sim, axis=1)[:, :5]
        knn_pred[va] = [Counter(y[tr][t]).most_common(1)[0][0] for t in top]
    defect = {lab: not is_ok(it["defect"]) for lab, it in ((x["label"], x) for x in items)}
    gt_bad = np.array([defect[v] for v in y])
    pr_bad = np.array([defect[v] for v in pred])
    return {
        "images": len(y), "classes": len(labels), "folds": folds,
        "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
        "knn5_accuracy": accuracy_score(y, knn_pred),
        "defect_recall": (gt_bad & pr_bad).sum() / max(1, gt_bad.sum()),
        "defect_precision": (gt_bad & pr_bad).sum() / max(1, pr_bad.sum()),
        "type_accuracy": float(np.mean([a.split(" - ")[0] == b.split(" - ")[0] for a, b in zip(y, pred)])),
        "fit_seconds": float(np.mean(fit_sec)),
        "report": classification_report(y, pred, zero_division=0),
        "confusions": Counter(f"{a} -> {b}" for a, b in zip(y, pred) if a != b).most_common(10),
        "errors": [(it["file"], p) for it, a, p in zip(items, y, pred) if a != p],
        "cv_pred": {it["file"]: p for it, p in zip(items, pred)},  # подсказки для гибрида (vlm_classify --hints)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "train"])
    ap.add_argument("root")
    ap.add_argument("--name", default="final")
    ap.add_argument("--sep")
    ap.add_argument("--model", default="weights/dinov2-base", help="папка с весами или имя на huggingface")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=2.0)
    args = ap.parse_args()

    root = Path(args.root)
    items = load(root, args.sep)
    y = np.array([x["label"] for x in items], dtype=object)
    t0 = time.time()
    model = str(ROOT / args.model) if (ROOT / args.model).exists() else args.model
    X = Embedder(model, args.device)([root / x["file"] for x in items])
    emb_sec = time.time() - t0
    print(f"{len(items)} фото, {len(set(y))} классов; признаки {emb_sec:.0f} с", file=sys.stderr)

    if args.mode == "eval":
        m = evaluate(items, X, y, args.folds, args.C)
        out = ROOT / "runs/cls" / f"{root.name}__{args.model.split('/')[-1]}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
        print(m["report"])
        for k in ("accuracy", "macro_f1", "knn5_accuracy", "type_accuracy", "defect_recall", "defect_precision", "fit_seconds"):
            print(f"{k:18} {m[k]:.3f}")
        print("частые ошибки:", *m["confusions"], sep="\n  ")
        print(f"-> {out}/metrics.json (там же список ошибочных фото)")
    else:
        t1 = time.time()
        clf = make_clf(args.C).fit(X, y)
        (ROOT / "weights").mkdir(exist_ok=True)
        path = ROOT / "weights" / f"cls_{args.name}.joblib"
        joblib.dump({"clf": clf, "model": args.model, "X": X, "items": items, "root": str(root.resolve())}, path)
        print(f"обучение {time.time() - t1:.1f} с -> {path}")


if __name__ == "__main__":
    main()
