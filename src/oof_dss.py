"""Честная оценка на 61 фото ДСС: out-of-fold предсказания и базовый признак «геометрия кадра».

    python src/oof_dss.py [data/dss_full] [--out out/dss_oof.json]

Зачем это отдельным скриптом: final.py выбирает связку по отложенной доле train и потому не даёт
единого честного числа. Здесь каждый кадр оценивается моделью, которая его не видела, а рядом
считается baseline только по размерам кадра — проверка гипотезы «широкий кадр = норма».

Протокол жёстко зафиксирован: один и тот же логрег (C=32), одни и те же фолды
(StratifiedGroupKFold, 5, seed 42), группы — «Группа» из таблицы, иначе имя файла без пути.
Никакого подбора порогов по результату и никакого VLM.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import groups_of, is_ok  # noqa: E402
from embed import ROOT  # noqa: E402
from final import Encoder, lr_probs, pick_device  # noqa: E402
from folders import load  # noqa: E402


def oof(F: np.ndarray, y_idx: np.ndarray, groups, n_cls: int, C: float = 32.0, folds: int = 5):
    """Индекс класса для каждого объекта: модель обучается без фолда этого объекта."""
    pred = np.empty(len(y_idx), dtype=int)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    for tr, va in cv.split(F, y_idx, groups):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(F[tr], y_idx[tr])
        pred[va] = lr_probs(clf, F[va], n_cls).argmax(1)
    return pred


def metrics(y, pred, ok_of: dict) -> dict:
    gt_bad = np.array([not ok_of[v] for v in y])
    pr_bad = np.array([not ok_of[v] for v in pred])
    return {
        "images": int(len(y)),
        "classes": len(set(y)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "type_accuracy": float(np.mean([a.split("/")[0] == b.split("/")[0] for a, b in zip(y, pred)])),
        "defect_recall": float((gt_bad & pr_bad).sum() / max(1, gt_bad.sum())),
        "defect_precision": float((gt_bad & pr_bad).sum() / max(1, pr_bad.sum())),
        "bad_to_ok": int((gt_bad & ~pr_bad).sum()),      # главная опасная ошибка: дефект закрыт как норма
        "ok_to_bad": int((~gt_bad & pr_bad).sum()),
        "true_bad": int(gt_bad.sum()),
        "confusions": Counter(f"{a} -> {b}" for a, b in zip(y, pred) if a != b).most_common(8),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(ROOT / "data/dss_full"))
    ap.add_argument("--out", default=str(ROOT / "out/dss_oof.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--C", type=float, default=32.0)
    args = ap.parse_args()

    root = Path(args.root)
    items = load(root)
    if not items:
        raise SystemExit(f"нет фото в {root}")
    classes = sorted({x["label"] for x in items})
    ok_of = {lab: is_ok(lab.split("/")[1]) for lab in classes}
    y = np.array([x["label"] for x in items], dtype=object)
    y_idx = np.array([classes.index(v) for v in y])
    groups = np.array(groups_of(items))
    print(f"{len(items)} фото, {len(classes)} классов, групп {len(set(groups))}", file=sys.stderr)

    paths = [root / x["file"] for x in items]
    enc = Encoder(pick_device(args.device), args.size, 2)
    F = enc.feats(enc.frozen([Image.open(p).convert("RGB") for p in paths]))

    pred = np.array(classes)[oof(F, y_idx, groups, len(classes), args.C)]
    result = {"protocol": {"folds": 5, "seed": 42, "C": args.C, "size": args.size,
                           "grouping": "Группа из таблицы, иначе имя файла",
                           "note": "exploratory grouped OOF, не независимый тест"},
              "dinov2": metrics(y, pred, ok_of)}

    # Baseline: только геометрия кадра. Если он конкурирует с DINO — это признак короткого замыкания.
    sizes = np.array([Image.open(p).size for p in paths], dtype=np.float64)
    G = np.log(np.c_[sizes[:, 0], sizes[:, 1], sizes[:, 0] / sizes[:, 1]])
    G = (G - G.mean(0)) / (G.std(0) + 1e-9)
    result["composition_baseline"] = metrics(y, np.array(classes)[oof(G, y_idx, groups, len(classes), args.C)], ok_of)
    # Тривиальные baseline: без них «деф.R 0.585» звучит лучше, чем есть (41 из 61 фото — дефектные).
    n_bad = int(sum(not ok_of[v] for v in y))
    result["baselines"] = {
        "all_bad": {"defect_recall": 1.0, "defect_precision": n_bad / len(y), "bad_to_ok": 0},
        "all_ok": {"defect_recall": 0.0, "defect_precision": 0.0, "bad_to_ok": n_bad},
        "prevalence": {"bad": n_bad, "ok": len(y) - n_bad, "share_bad": n_bad / len(y)},
    }
    result["oof_predictions"] = [{"file": x["file"], "true": t, "pred": p} for x, t, p in zip(items, y, pred)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for name in ("dinov2", "composition_baseline"):
        m = result[name]
        print(f"{name:22} F1 {m['macro_f1']:.3f}  acc {m['accuracy']:.3f}  тип {m['type_accuracy']:.3f}  "
              f"деф.R {m['defect_recall']:.3f}  деф.P {m['defect_precision']:.3f}  bad->ok {m['bad_to_ok']}/{m['true_bad']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
