"""Признаки для мелких дефектов на большом кадре: кроме [CLS, среднее по патчам] берём максимум по патчам
и признаки 4 четвертей кадра (2×2). Сравнение на holdout (train -> test).

    python src/poc/local_feats.py data/rdd_train data/rdd_test
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import groups_of, is_ok  # noqa: E402
from embed import CACHE, ROOT, Embedder  # noqa: E402
from folders import load  # noqa: E402
from sweep import cv_probs, score  # noqa: E402


def compute(emb: Embedder, paths, size_full=448, size_tile=224):
    out = []
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        W, H = im.size
        tiles = [im.crop((x * W // 2, y * H // 2, (x + 1) * W // 2, (y + 1) * H // 2)) for y in range(2) for x in range(2)]
        with torch.inference_mode():
            emb.size = size_full
            h = emb.model(pixel_values=emb._tensor([im])).last_hidden_state[0]
            cls, pt = h[0], h[1:]
            emb.size = size_tile
            t = emb.model(pixel_values=emb._tensor(tiles)).last_hidden_state
            tf = torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)
        n = torch.nn.functional.normalize
        out.append(torch.cat([n(torch.cat([cls, pt.mean(0)]), dim=0), n(pt.max(0).values, dim=0),
                              n(tf.mean(0), dim=0), n(tf.max(0).values, dim=0)]).numpy())
        print(f"\r  {k + 1}/{len(paths)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return np.stack(out).astype(np.float32)


def main():
    tr_root, te_root = Path(sys.argv[1]), Path(sys.argv[2])
    tr, te = load(tr_root), load(te_root)
    cache = CACHE / f"local_{tr_root.name}_{te_root.name}.npz"
    if cache.exists():
        d = np.load(cache)
        Xtr, Xte = d["tr"], d["te"]
    else:
        emb = Embedder(str(ROOT / "weights/dinov2-base"))
        Xtr = compute(emb, [tr_root / x["file"] for x in tr])
        Xte = compute(emb, [te_root / x["file"] for x in te])
        np.savez(cache, tr=Xtr, te=Xte)
    ytr = np.array([x["label"] for x in tr], dtype=object)
    yte = np.array([x["label"] for x in te], dtype=object)
    ok_of = {x["label"]: is_ok(x["defect"]) for x in tr + te}
    C = np.array(sorted(set(ytr)), dtype=object)
    d = 1536
    blocks = {"cls+mean (как сейчас)": slice(0, d), "max патчей": slice(d, 2 * d),
              "четверти mean": slice(2 * d, 3 * d), "четверти max": slice(3 * d, 4 * d)}
    combos = {**{k: [v] for k, v in blocks.items()},
              "cls+mean + max": [blocks["cls+mean (как сейчас)"], blocks["max патчей"]],
              "cls+mean + четверти": [blocks["cls+mean (как сейчас)"], blocks["четверти mean"], blocks["четверти max"]],
              "всё": list(blocks.values())}
    for name, sl in combos.items():
        A, B = np.hstack([Xtr[:, s] for s in sl]), np.hstack([Xte[:, s] for s in sl])
        for c in (2.0, 32.0):
            cv = score(ytr, C[cv_probs(A, ytr, groups_of(tr), c).argmax(1)], ok_of)
            clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(A, ytr)
            t = score(yte, clf.predict(B), ok_of)
            print(f"{name:24} C={c:4}  CV F1 {cv['macro_f1']:.3f}  test F1 {t['macro_f1']:.3f}  acc {t['acc']:.3f}  "
                  f"деф.R {t['defect_recall']:.3f}  деф.P {t['defect_prec']:.3f}")


if __name__ == "__main__":
    main()
