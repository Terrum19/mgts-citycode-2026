"""Перебор: сеть для признаков × разрешение × классификатор × склейки (в т.ч. с ответом Qwen).

    python src/poc/sweep.py data/sim_final --feats dinov2-base:224 dinov2-base:448 convnextv2_base...:384
    python src/poc/sweep.py data/sim_final --feats dinov2-base:448 --qwen runs/cls/sim_final__vlm_qwen235.json

Признак задаётся как <папка в weights/>:<размер>[:tta]. Несколько --feats можно склеить (--concat).
Таблица: macro-F1, точность, полнота и точность по «есть дефект» — на одной и той же
перекрёстной проверке с группами (5 фолдов, seed 42). Результат дописывается в runs/cls/sweep.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classify import groups_of, is_ok  # noqa: E402
from embed import ROOT, Embedder  # noqa: E402
from folders import load  # noqa: E402


def feats(spec: str, paths: list[Path]) -> np.ndarray:
    parts = spec.split(":")
    size = int(parts[1]) if len(parts) > 1 and parts[1] else None
    return Embedder(str(ROOT / "weights" / parts[0]), size=size, tta="tta" in parts[2:])(paths)


def cv_probs(X, y, groups, C=2.0, folds=5) -> np.ndarray:
    """Вероятности классов для каждого фото, предсказанные моделью, которая это фото не видела."""
    classes = sorted(set(y))
    P = np.zeros((len(y), len(classes)))
    for tr, va in StratifiedGroupKFold(folds, shuffle=True, random_state=42).split(X, y, groups):
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[tr], y[tr])
        P[np.ix_(va, [classes.index(c) for c in clf.classes_])] = clf.predict_proba(X[va])
    return P


def score(y, pred, ok_of) -> dict:
    gt_bad = np.array([not ok_of[v] for v in y])
    pr_bad = np.array([not ok_of[v] for v in pred])
    return {"macro_f1": f1_score(y, pred, average="macro"), "acc": accuracy_score(y, pred),
            "defect_recall": (gt_bad & pr_bad).sum() / gt_bad.sum(), "defect_prec": (gt_bad & pr_bad).sum() / max(1, pr_bad.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--feats", nargs="+", required=True)
    ap.add_argument("--concat", action="store_true", help="также склеить все --feats в один вектор")
    ap.add_argument("--C", nargs="+", type=float, default=[0.5, 2.0, 8.0])
    ap.add_argument("--dump", help="сохранить top-3 классов с вероятностями лучшей строки без Qwen (для vlm_classify --hints)")
    ap.add_argument("--qwen", help="ответы vlm_classify.py: добавить one-hot ответа Qwen к признакам и усреднить")
    args = ap.parse_args()

    root = Path(args.root)
    items = load(root)
    y = np.array([x["label"] for x in items], dtype=object)
    classes = sorted(set(y))
    ok_of = {x["label"]: is_ok(x["defect"]) for x in items}
    groups = groups_of(items)
    paths = [root / x["file"] for x in items]

    F = {}
    for spec in args.feats:
        t0 = time.time()
        F[spec] = feats(spec, paths)
        print(f"{spec}: dim {F[spec].shape[1]}, {time.time() - t0:.0f} с", file=sys.stderr)
    if args.concat and len(F) > 1:
        F["+".join(args.feats)] = np.hstack([F[s] for s in args.feats])

    Q = None
    if args.qwen:
        ans = {r["file"]: r["label"] for r in json.loads(Path(args.qwen).read_text())}
        Q = np.zeros((len(items), len(classes)))
        for i, x in enumerate(items):
            if ans.get(x["file"]) in classes:
                Q[i, classes.index(ans[x["file"]])] = 1
        has = np.array([x["file"] in ans for x in items])
        print(f"Qwen: ответы на {has.sum()} из {len(items)}; одна Qwen: "
              + json.dumps({k: round(v, 3) for k, v in score(y[has], np.array(classes, dtype=object)[Q[has].argmax(1)], ok_of).items()}),
              file=sys.stderr)

    rows, best = [], (-1, None)
    for name, X in F.items():
        for C in args.C:
            P = cv_probs(X, y, groups, C)
            rows.append({"feats": name, "C": C, "head": "logreg", **score(y, np.array(classes, dtype=object)[P.argmax(1)], ok_of)})
            if rows[-1]["macro_f1"] > best[0]:
                best = (rows[-1]["macro_f1"], P)
            if Q is not None:
                # стэкинг: ответ Qwen как дополнительные признаки (вес подобран один раз, не по фолдам)
                for w in (0.5, 1.0):
                    Ps = cv_probs(np.hstack([X, w * Q]), y, groups, C)
                    rows.append({"feats": name, "C": C, "head": f"logreg+qwen_feat{w}",
                                 **score(y, np.array(classes, dtype=object)[Ps.argmax(1)], ok_of)})
                for w in (0.3, 0.5):
                    rows.append({"feats": name, "C": C, "head": f"avg(p, {w}*qwen)",
                                 **score(y, np.array(classes, dtype=object)[(P + w * Q).argmax(1)], ok_of)})
    rows.sort(key=lambda r: -r["macro_f1"])
    print(f"\n{'признаки':55} {'C':>4} {'голова':22} {'F1':>6} {'acc':>6} {'деф.R':>6} {'деф.P':>6}")
    for r in rows:
        print(f"{r['feats'][:55]:55} {r['C']:4} {r['head']:22} {r['macro_f1']:6.3f} {r['acc']:6.3f} "
              f"{r['defect_recall']:6.3f} {r['defect_prec']:6.3f}")
    if args.dump:
        P = best[1]
        top = {x["file"]: [[classes[j], round(float(P[i, j]), 3)] for j in np.argsort(-P[i])[:3]] for i, x in enumerate(items)}
        Path(args.dump).write_text(json.dumps(top, ensure_ascii=False))
        print(f"top-3 лучшей строки (F1 {best[0]:.3f}) -> {args.dump}", file=sys.stderr)
    out = ROOT / "runs/cls/sweep.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for r in rows:
            f.write(json.dumps({"root": root.name, **r}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
