"""Честная проверка как на финале: учим на одной папке, проверяем на другой, которую модель не видела.

    python src/poc/holdout.py data/rdd_train data/rdd_test --qwen runs/cls/rdd_test__vlm_qwen235.json

Печатает: оценку перекрёстной проверкой на train (это то, что мы увидим на хакатоне до сдачи) и
настоящие метрики на test для классификатора, одной Qwen и связки (p + w*Qwen).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import groups_of, is_ok  # noqa: E402
from folders import load  # noqa: E402
from sweep import cv_probs, feats, score  # noqa: E402


def fmt(d):
    return "  ".join(f"{k} {v:.3f}" for k, v in d.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--feats", default="dinov2-base:448")
    ap.add_argument("--C", type=float, default=32.0)
    ap.add_argument("--w", type=float, default=0.3)
    ap.add_argument("--qwen")
    args = ap.parse_args()

    tr_root, te_root = Path(args.train), Path(args.test)
    tr, te = load(tr_root), load(te_root)
    ytr = np.array([x["label"] for x in tr], dtype=object)
    yte = np.array([x["label"] for x in te], dtype=object)
    C = np.array(sorted(set(ytr)), dtype=object)
    ok_of = {x["label"]: is_ok(x["defect"]) for x in tr + te} | {"?": True}
    Xtr = feats(args.feats, [tr_root / x["file"] for x in tr])
    Xte = feats(args.feats, [te_root / x["file"] for x in te])

    print(f"train {len(tr)}, test {len(te)}, признаки {args.feats}, C={args.C}")
    Pcv = cv_probs(Xtr, ytr, groups_of(tr), args.C)
    print(f"\nоценка на хакатоне (CV на train):  {fmt(score(ytr, C[Pcv.argmax(1)], ok_of))}")

    clf = LogisticRegression(C=args.C, max_iter=5000, class_weight="balanced").fit(Xtr, ytr)
    P = clf.predict_proba(Xte)
    res = {"классификатор": C[P.argmax(1)]}
    if args.qwen:
        ans = {r["file"]: r["label"] for r in json.loads(Path(args.qwen).read_text())}
        Q = np.zeros_like(P)
        for i, x in enumerate(te):
            if ans.get(x["file"]) in C:
                Q[i, list(C).index(ans[x["file"]])] = 1
        res["Qwen"] = np.array([ans.get(x["file"]) or "?" for x in te], dtype=object)
        for w in sorted({args.w, 0.3, 0.5, 1.0}):
            res[f"связка w={w}"] = C[(P + w * Q).argmax(1)]
    print("настоящие метрики на test:")
    for name, pred in res.items():
        print(f"  {name:16} {fmt(score(yte, pred, ok_of))}")

    best = res.get(f"связка w={args.w}", res["классификатор"])
    print(f"\nпо классам (связка w={args.w}):")
    print(classification_report(yte, best, digits=3, zero_division=0))
    country = [x["extra"].get("Страна") for x in te]
    for c in sorted(set(filter(None, country))):
        m = np.array([k == c for k in country])
        print(f"  {c}: {fmt(score(yte[m], best[m], ok_of))}")
    print("частые ошибки:", *Counter(f"{a} -> {b}" for a, b in zip(yte, best) if a != b).most_common(6), sep="\n  ")


if __name__ == "__main__":
    main()
