"""Дообучение хвоста DINOv2: первые блоки заморожены (их выход считаем один раз и кэшируем),
учим последние K блоков + линейную голову. Быстро даже на CPU. Holdout: train -> test.

    python src/poc/finetune_tail.py data/rdd_train data/rdd_test --k 2 --epochs 12
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import is_ok  # noqa: E402
from embed import CACHE, ROOT, Embedder  # noqa: E402
from folders import load  # noqa: E402
from sweep import score  # noqa: E402


def layer_out(layer, h):
    o = layer(h)
    return o[0] if isinstance(o, (tuple, list)) else o


def frozen_tokens(emb, model, paths, cut, flip, tag):
    f = CACHE / f"tokens_{tag}_cut{cut}{'_flip' if flip else ''}.pt"
    if NOSAVE[0] is False and f.exists():
        return torch.load(f)
    out = []
    for s in range(0, len(paths), 16):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + 16]]
        if flip:
            imgs = [ImageOps.mirror(im) for im in imgs]
        with torch.inference_mode():
            h = model.embeddings(emb._tensor(imgs))
            for layer in model.encoder.layer[:cut]:
                h = layer_out(layer, h)
        out.append(h.half())
        print(f"\r  токены {tag}{' flip' if flip else ''}: {min(s + 16, len(paths))}/{len(paths)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    t = torch.cat(out)
    if not NOSAVE[0]:
        torch.save(t, f)
    return t


NOSAVE = [False]


class Tail(torch.nn.Module):
    def __init__(self, model, cut, n_cls):
        super().__init__()
        self.blocks = copy.deepcopy(model.encoder.layer[cut:])
        self.norm = copy.deepcopy(model.layernorm)
        d = model.config.hidden_size
        self.head = torch.nn.Linear(2 * d, n_cls)

    def forward(self, h):
        for b in self.blocks:
            h = layer_out(b, h)
        h = self.norm(h)
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--k", type=int, default=2, help="сколько последних блоков учить")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--no-save", action="store_true", help="не писать токены на диск (держать в памяти)")
    args = ap.parse_args()
    NOSAVE[0] = args.no_save
    torch.manual_seed(0)

    tr_root, te_root = Path(args.train), Path(args.test)
    tr, te = load(tr_root), load(te_root)
    classes = sorted({x["label"] for x in tr})
    ok_of = {x["label"]: is_ok(x["defect"]) for x in tr + te}
    ytr = torch.tensor([classes.index(x["label"]) for x in tr])
    yte = np.array([x["label"] for x in te], dtype=object)

    emb = Embedder(str(ROOT / "weights/dinov2-base"), size=args.size)
    model = emb.model
    cut = len(model.encoder.layer) - args.k
    tag = f"{tr_root.name}_s{args.size}"
    Htr = frozen_tokens(emb, model, [tr_root / x["file"] for x in tr], cut, False, tag)
    Htr_f = frozen_tokens(emb, model, [tr_root / x["file"] for x in tr], cut, True, tag)
    Hte = frozen_tokens(emb, model, [te_root / x["file"] for x in te], cut, False, f"{te_root.name}_s{args.size}")

    net = Tail(model, cut, len(classes))
    opt = torch.optim.AdamW([{"params": list(net.blocks.parameters()) + list(net.norm.parameters()), "lr": args.lr},
                             {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    w = torch.tensor(len(ytr) / (len(classes) * np.bincount(ytr.numpy(), minlength=len(classes))), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    steps = args.epochs * ((len(ytr) + 31) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, 1e-3], total_steps=steps, pct_start=0.1)

    def predict(H):
        net.eval()
        with torch.inference_mode():
            return torch.cat([net(H[s:s + 64].float()) for s in range(0, len(H), 64)]).softmax(1).numpy()

    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        perm = torch.randperm(len(ytr))
        tot = 0.0
        for s in range(0, len(perm), 32):
            idx = perm[s:s + 32]
            flip = torch.rand(len(idx)) < 0.5
            h = torch.where(flip[:, None, None], Htr_f[idx], Htr[idx]).float()
            loss = loss_fn(net(h), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        P = predict(Hte)
        m = score(yte, np.array(classes, dtype=object)[P.argmax(1)], ok_of)
        print(f"эпоха {ep + 1:2}: loss {tot / len(ytr):.3f}  test F1 {m['macro_f1']:.3f}  acc {m['acc']:.3f}  "
              f"деф.R {m['defect_recall']:.3f}  деф.P {m['defect_prec']:.3f}  ({time.time() - t0:.0f} с)", flush=True)
    np.save(ROOT / f"runs/cls/ft_probs_{te_root.name}_k{args.k}_s{args.size}.npy", P)


if __name__ == "__main__":
    main()
