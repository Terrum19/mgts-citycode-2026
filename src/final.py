"""Финал одной командой: папки train -> выбор связки на 80/20 -> переобучение на всём -> предсказания test.

    python src/final.py data/final_train data/final_test --out out/final
    python src/final.py data/rdd_train data/rdd_test --no-qwen          # проверка без сети

Шаги:
 1. Train делится 80/20 по группам (как в перекрёстной проверке: связанные фото не разъезжаются).
 2. Один проход DINOv2-base 448 по train даёт и выход первых 10 блоков (для дообучения), и признаки
    [CLS, среднее по патчам] (для логрега). Qwen тем временем отвечает на 20% и test, по сети.
 3. На 80% учатся логрег и дообучение 2 последних блоков. На 20% сравниваются связки:
    логрег, дообучение, их среднее, каждая ± ответ Qwen с весом w. Побеждает лучшая по macro-F1.
 4. Выбранная связка переобучается на 100% train и предсказывает test.

Выход в --out: predictions.xlsx (класс, тип, дефект, P(дефект), описание от Qwen, колонки из Excel test),
report.json (таблица связок на 20%, выбор, время этапов), qwen.json (кэш ответов: повторный запуск не платит дважды).
Test может быть плоской папкой с фото или папками «Тип - Дефект» — тогда печатаются настоящие метрики.
Память: ~3 ГБ на 1000 фото train (токены 448 хранятся вместе с отражёнными); если мало — --size 336.
"""
import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import groups_of, is_ok  # noqa: E402
from embed import IMAGENET, ROOT  # noqa: E402
from folders import IMG_EXT, TABLE_EXT, load, match_table, read_table, split_label  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} мин] {msg}", flush=True)


def pick_device(want):
    if want != "auto":
        return want
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load_test(root: Path, classes, test_format="auto"):
    """Папки «Тип - Дефект» -> (items, True); плоская папка с фото (+ Excel) -> (items без label, False)."""
    imgs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    if test_format != "images":
        items = load(root)
        if test_format == "labeled":
            if not items or len(items) != len(imgs):
                raise ValueError("В режиме labeled все фото должны лежать в папках классов")
            return items, True
        if items and len(items) == len(imgs) and all(item["label"] in classes for item in items):
            return items, True
    meta = {}
    for t in sorted(p for p in root.rglob("*") if p.suffix.lower() in TABLE_EXT and not p.name.startswith("~$")):
        meta |= match_table(read_table(t), [p.relative_to(root).as_posix() for p in imgs])
    return [{"file": p.relative_to(root).as_posix(), "label": None,
             "extra": {k: v for k, v in meta.get(p.relative_to(root).as_posix(), {}).items() if v == v}} for p in imgs], False


def score(y, pred, ok_of) -> dict:
    gt_bad = np.array([not ok_of.get(v, True) for v in y])
    pr_bad = np.array([not ok_of.get(v, True) for v in pred])
    return {"macro_f1": f1_score(y, pred, average="macro"), "acc": accuracy_score(y, pred),
            "defect_recall": (gt_bad & pr_bad).sum() / max(1, gt_bad.sum()),
            "defect_prec": (gt_bad & pr_bad).sum() / max(1, pr_bad.sum())}


def fmt(m):
    return f"F1 {m['macro_f1']:.3f}  acc {m['acc']:.3f}  деф.R {m['defect_recall']:.3f}  деф.P {m['defect_prec']:.3f}"


# ---------- DINOv2: один проход даёт токены после первых cut блоков и признаки для логрега ----------

def layer_out(layer, h):
    o = layer(h)
    return o[0] if isinstance(o, (tuple, list)) else o


class Encoder:
    def __init__(self, dev, size, k):
        from transformers import AutoModel
        local = ROOT / "weights/dinov2-base"
        self.model = AutoModel.from_pretrained(str(local) if local.exists() else "facebook/dinov2-base").eval().to(dev)
        self.dev, self.size = dev, size
        self.cut = len(self.model.encoder.layer) - k
        self.mean, self.std = np.array(IMAGENET[0], np.float32), np.array(IMAGENET[1], np.float32)

    def _tensor(self, imgs):
        arr = np.stack([np.asarray(im.resize((self.size, self.size), Image.BICUBIC), np.float32) / 255 for im in imgs])
        return torch.from_numpy((arr - self.mean) / self.std).permute(0, 3, 1, 2).contiguous().to(self.dev)

    @torch.inference_mode()
    def frozen(self, imgs):
        h = self.model.embeddings(self._tensor(imgs))
        for layer in self.model.encoder.layer[:self.cut]:
            h = layer_out(layer, h)
        return h

    @torch.inference_mode()
    def feats(self, h):
        """Досчитать исходные (не дообученные) последние блоки -> [CLS, среднее], нормированный вектор."""
        for layer in self.model.encoder.layer[self.cut:]:
            h = layer_out(layer, h)
        h = self.model.layernorm(h)
        return torch.nn.functional.normalize(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1), dim=1).float().cpu().numpy()

    def batches(self, paths, batch=16, flip=False, what=""):
        for s in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
            if flip:
                imgs = [ImageOps.mirror(im) for im in imgs]
            yield imgs
            print(f"\r  {what}: {min(s + batch, len(paths))}/{len(paths)}", end="", file=sys.stderr)
        print(file=sys.stderr)


class Tail(torch.nn.Module):
    """Копия последних k блоков DINOv2 + layernorm + линейная голова на [CLS, среднее]."""

    def __init__(self, model, cut, n_cls):
        super().__init__()
        self.blocks = copy.deepcopy(model.encoder.layer[cut:])
        self.norm = copy.deepcopy(model.layernorm)
        self.head = torch.nn.Linear(2 * model.config.hidden_size, n_cls)

    def forward(self, h):
        for b in self.blocks:
            h = layer_out(b, h)
        h = self.norm(h)
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))


def train_tail(enc, H, Hf, y, idx, n_cls, epochs, lr, seed=0):
    """Дообучение на фото idx: H/Hf — токены (CPU, fp16) обычных и отражённых фото, y — номера классов."""
    torch.manual_seed(seed)
    net = Tail(enc.model, enc.cut, n_cls).to(enc.dev).train()
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": list(net.blocks.parameters()) + list(net.norm.parameters()), "lr": lr},
                             {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    cnt = np.maximum(np.bincount(y[idx], minlength=n_cls), 1)
    w = torch.tensor(len(idx) / (n_cls * cnt), dtype=torch.float32, device=enc.dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, 1e-3], total_steps=epochs * ((len(idx) + 31) // 32),
                                                pct_start=0.1)
    yt = torch.from_numpy(y)
    for ep in range(epochs):
        perm = torch.from_numpy(idx)[torch.randperm(len(idx))]
        tot = 0.0
        for s in range(0, len(perm), 32):
            b = perm[s:s + 32]
            flip = torch.rand(len(b)) < 0.5
            h = torch.where(flip[:, None, None], Hf[b], H[b]).to(enc.dev).float()
            loss = loss_fn(net(h), yt[b].to(enc.dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        print(f"\r  дообучение: эпоха {ep + 1}/{epochs}, loss {tot / len(idx):.3f}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return net.eval()


@torch.inference_mode()
def tail_probs(net, h):
    return net(h.float()).softmax(1).cpu().numpy()


def logreg(F, y, idx, n_cls, C):
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(F[idx], y[idx])
    return clf, clf.classes_


def lr_probs(clf, F, n_cls):
    P = np.zeros((len(F), n_cls))
    P[:, clf.classes_] = clf.predict_proba(F)
    return P


# ---------- Qwen: отвечает в фоне, кэш на диске ----------

class Qwen:
    def __init__(self, labels, cache: Path):
        from run import load_config
        from vlm import VLM
        cfg = load_config()
        self.vlm = VLM(cfg["variants"]["qwen235"], cfg["taxonomy"])
        self.labels, self.cache_path = labels, cache
        self.cache = json.loads(cache.read_text()) if cache.exists() else {}
        self.pool = ThreadPoolExecutor(cfg["variants"]["qwen235"].get("workers", 4))
        self.fut = {}

    def submit(self, paths):
        from vlm_classify import classify

        def one(p):
            try:
                r = classify(self.vlm, Image.open(p), self.labels)
            except Exception as e:  # noqa: BLE001 — один сбой не должен ронять финал
                r = {"label": None, "description": "", "error": str(e)[:200]}
            return {"label": r.get("label"), "description": r.get("description", ""), "error": r.get("error")}
        for p in map(str, paths):
            if p not in self.cache and p not in self.fut:
                self.fut[p] = self.pool.submit(one, p)

    def get(self, paths):
        out = []
        for p in map(str, paths):
            if p not in self.cache:
                self.cache[p] = self.fut.pop(p).result()
            out.append(self.cache[p])
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1))
        return out

    def onehot(self, answers, classes):
        Q = np.zeros((len(answers), len(classes)))
        for i, a in enumerate(answers):
            if a["label"] in classes:
                Q[i, classes.index(a["label"])] = 1
        return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train", help="папки «Тип - Дефект» с фото и Excel")
    ap.add_argument("test", help="фото для предсказания (плоская папка или такие же папки)")
    ap.add_argument("--out", default="out/final")
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--k", type=int, default=2, help="сколько последних блоков дообучать")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--C", type=float, default=32.0, help="регуляризация логрега")
    ap.add_argument("--val", type=float, default=0.2, help="доля train для выбора связки")
    ap.add_argument("--no-qwen", action="store_true")
    ap.add_argument("--training", choices=["auto", "head", "tail"], default="auto",
                    help="auto: сравнить связки; head: только логрег; tail: дообученный хвост")
    ap.add_argument("--test-format", choices=["auto", "images", "labeled"], default="auto")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_root, te_root = Path(args.train), Path(args.test)
    tr = load(tr_root)
    classes = sorted({x["label"] for x in tr})
    te, te_labeled = load_test(te_root, classes, args.test_format)
    if len(classes) < 2 or not te:
        ap.error("нужны хотя бы два класса в train и хотя бы одно фото в test")
    n_cls = len(classes)
    y = np.array([classes.index(x["label"]) for x in tr])
    ok_of = {x["label"]: is_ok(x["defect"]) for x in tr + (te if te_labeled else [])}
    tr_paths = [tr_root / x["file"] for x in tr]
    te_paths = [te_root / x["file"] for x in te]
    _, va = next(StratifiedGroupKFold(round(1 / args.val), shuffle=True, random_state=0)
                 .split(np.zeros(len(y)), y, groups_of(tr)))
    fit = np.setdiff1d(np.arange(len(y)), va)
    dev = pick_device(args.device)
    log(f"train {len(tr)} фото, {n_cls} классов (выбор связки: {len(fit)} / {len(va)}), test {len(te)} фото"
        f"{' с разметкой' if te_labeled else ''}; устройство {dev}")
    times = {}

    qwen = None
    if not args.no_qwen:
        qwen = Qwen(classes, out / "qwen.json")
        qwen.submit([tr_paths[i] for i in va] + te_paths)  # сначала 20% — они нужны раньше
        log("Qwen запущен в фоне")

    # 1. Один проход по train: токены (обычные и отражённые) + признаки
    t = time.time()
    enc = Encoder(dev, args.size, args.k)
    H, Hf, F = [], [], []
    for imgs in enc.batches(tr_paths, what="train"):
        h = enc.frozen(imgs)
        H.append(h.half().cpu())
        F.append(enc.feats(h))
    for imgs in enc.batches(tr_paths, flip=True, what="train, отражённые"):
        Hf.append(enc.frozen(imgs).half().cpu())
    H, Hf, F = torch.cat(H), torch.cat(Hf), np.concatenate(F)
    times["проход по train"] = time.time() - t
    log(f"проход по train готов ({times['проход по train'] / 60:.1f} мин)")

    # 2. Выбор связки на 20%
    t = time.time()
    clf, _ = logreg(F, y, fit, n_cls, args.C)
    P_lr = lr_probs(clf, F[va], n_cls)
    P_ft = None
    if args.training != "head":
        net = train_tail(enc, H, Hf, y, fit, n_cls, args.epochs, args.lr)
        P_ft = np.concatenate([tail_probs(net, H[va[s:s + 64]].to(dev)) for s in range(0, len(va), 64)])
        del net
    times["обучение на 80%"] = time.time() - t
    bases = {"логрег": P_lr} if args.training != "tail" else {}
    if P_ft is not None:
        bases["дообучение"] = P_ft
        if args.training == "auto":
            bases["логрег+дообучение"] = (P_lr + P_ft) / 2
    Q_va = None
    if qwen:
        log("жду ответы Qwen на 20%")
        Q_va = qwen.onehot(qwen.get([tr_paths[i] for i in va]), classes)
    C = np.array(classes, dtype=object)
    rows = []
    for name, P in bases.items():
        for w in ([0, 0.3, 0.5, 1.0] if Q_va is not None else [0]):
            pred = C[(P + w * Q_va).argmax(1)] if w else C[P.argmax(1)]
            rows.append({"связка": name + (f" + Qwen×{w}" if w else ""), "base": name, "w": w,
                         **score(C[y[va]], pred, ok_of)})
    if Q_va is not None:
        rows.append({"связка": "только Qwen", "base": None, "w": 1.0,
                     **score(C[y[va]], C[Q_va.argmax(1)], ok_of)})
    best = max((r for r in rows if r["base"]), key=lambda r: r["macro_f1"])  # при равенстве — первая, т.е. проще
    print(f"\nСвязки на {len(va)} отложенных фото (это и есть наша оценка качества):")
    for r in sorted(rows, key=lambda r: -r["macro_f1"]):
        print(f"  {'→' if r is best else ' '} {r['связка']:30} {fmt(r)}")
    log(f"выбрана: {best['связка']}")

    # 3. Переобучение выбранного на всём train
    t = time.time()
    everything = np.arange(len(y))
    use_lr, use_ft = best["base"] in ("логрег", "логрег+дообучение"), best["base"] in ("дообучение", "логрег+дообучение")
    clf = logreg(F, y, everything, n_cls, args.C)[0] if use_lr else None
    net = train_tail(enc, H, Hf, y, everything, n_cls, args.epochs, args.lr) if use_ft else None
    import joblib
    # CPU tensors keep the checkpoint portable between CPU and MPS.
    state = {
        "size": args.size, "k": args.k, "classes": classes,
        "ok": {c: ok_of[c] for c in classes}, "logreg": clf,
        "tail": {key: value.cpu() for key, value in net.state_dict().items()} if net is not None else None,
        "qwen_weight": best["w"],
    }
    joblib.dump(state, out / "photo_dino.joblib.tmp")
    (out / "photo_dino.joblib.tmp").replace(out / "photo_dino.joblib")
    del H, Hf
    times["обучение на 100%"] = time.time() - t
    log("переобучено на всём train")

    # 4. Test: один проход, обе модели сразу
    t = time.time()
    Ps = []
    for imgs in enc.batches(te_paths, what="test"):
        h = enc.frozen(imgs)
        parts = ([lr_probs(clf, enc.feats(h), n_cls)] if use_lr else []) + ([tail_probs(net, h)] if use_ft else [])
        Ps.append(sum(parts) / len(parts))
    P = np.concatenate(Ps)
    answers = qwen.get(te_paths) if qwen else [{"label": None, "description": ""}] * len(te)
    if best["w"]:
        P = P + best["w"] * qwen.onehot(answers, classes)
    P = P / P.sum(1, keepdims=True)
    times["test"] = time.time() - t
    pred = C[P.argmax(1)]
    bad = np.array([not ok_of[c] for c in classes])
    p_bad = P[:, bad].sum(1)

    rows_out = []
    for i, x in enumerate(te):
        typ, defect = split_label(pred[i], None)
        top = np.argsort(-P[i])[:3]
        rows_out.append({"Имя файла": x["file"], "Класс": pred[i], "Тип": typ, "Дефект": defect,
                         "Неисправен": bool(not ok_of[pred[i]]), "P(дефект)": round(float(p_bad[i]), 3),
                         "Уверенность": round(float(P[i].max()), 3),
                         "Топ-3": "; ".join(f"{classes[j]} {P[i, j]:.2f}" for j in top),
                         "Qwen: класс": answers[i]["label"], "Описание (Qwen)": answers[i]["description"],
                         **({"Истинный класс": x["label"]} if te_labeled else {}), **x.get("extra", {})})
    df = pd.DataFrame(rows_out)
    df.to_excel(out / "predictions.xlsx", index=False)
    df.to_csv(out / "predictions.csv", index=False)
    report = {"train": len(tr), "test": len(te), "classes": classes, "device": dev, "chosen": best["связка"],
              "val_table": rows, "minutes": {k: round(v / 60, 1) for k, v in times.items()},
              "args": vars(args)}
    if te_labeled:
        report["test_metrics"] = score(np.array([x["label"] for x in te], dtype=object), pred, ok_of)
        print(f"\nнастоящие метрики на test ({best['связка']}): {fmt(report['test_metrics'])}")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(f"\nнеисправных в test: {int(df['Неисправен'].sum())} из {len(df)}")
    log(f"готово -> {out}/predictions.xlsx; время по этапам, мин: {report['minutes']}")


if __name__ == "__main__":
    main()
