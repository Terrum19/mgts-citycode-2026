"""Финал одной командой: папки train -> выбор связки на 80/20 -> переобучение на всём -> предсказания test.

    python src/final.py data/final_train data/final_test --out out/final
    python src/final.py data/rdd_train data/rdd_test --no-qwen          # проверка без сети

Шаги:
 1. Train делится 80/20 по группам (как в перекрёстной проверке: связанные фото не разъезжаются).
 2. Один проход DINOv2-base 448 по train даёт и выход первых 10 блоков (для дообучения), и признаки
    [CLS, среднее по патчам] (для логрега). Qwen тем временем отвечает на 20% и test, по сети.
    С --augment N к признакам логрега добавляются N погодных копий train-фото ([augment] config.toml):
    хранятся только векторы, токены H/Hf не аугментируются (иначе RAM вырастает кратно).
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
import os
import sys
import time
from datetime import datetime, timezone
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
from atomic import write_json_atomic  # noqa: E402
from augment import augment, sample_seed  # noqa: E402
from classify import groups_of, is_ok  # noqa: E402
from config_utils import resolve_variant  # noqa: E402
from embed import IMAGENET, ROOT  # noqa: E402
from folders import IMG_EXT, TABLE_EXT, load, match_table, read_table, split_label  # noqa: E402
from policy import finalize, initial, semantic_policy  # noqa: E402
from selection import baselines as selection_baselines  # noqa: E402
from selection import frame_metrics, select_candidate  # noqa: E402
from taxonomy import (FORMAT_VERSION, hash_files, package_versions,  # noqa: E402
                      taxonomy_fingerprint)
from vlm_cache import Qwen  # noqa: E402

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
    def __init__(self, dev, size, k, offline: bool = False):
        from transformers import AutoModel
        local = ROOT / "weights/dinov2-base"
        if offline and not local.exists():
            raise FileNotFoundError(f"offline: нет локальных весов DINOv2 в {local}")
        source = str(local) if local.exists() else "facebook/dinov2-base"
        local_only = bool(offline or os.environ.get("HF_HUB_OFFLINE"))
        self.model = AutoModel.from_pretrained(source, local_files_only=local_only).eval().to(dev)
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


def augmented_features(enc, paths, y, indices, copies: int, cfg: dict, seed: int, batch: int = 16):
    """Погодные копии фото для признаков логрега -> (Fa, ya).

    Память: хранятся только векторы (n × 1536 float32), токены H/Hf не создаются — поэтому
    аугментация дообучаемого хвоста здесь сознательно не делается (нужна отдельная оценка RAM).
    Детерминизм: seed копии зависит только от номера фото и номера копии.
    """
    width = 2 * enc.model.config.hidden_size
    if copies <= 0 or len(indices) == 0:
        return np.empty((0, width), np.float32), np.empty(0, y.dtype)
    feats, labels = [], []
    for start in range(0, len(indices), batch):
        chunk = indices[start:start + batch]
        images = [Image.open(paths[int(i)]).convert("RGB") for i in chunk]
        for copy in range(copies):
            augmented = [augment(img, cfg, sample_seed(seed, int(i), copy)) for i, img in zip(chunk, images)]
            feats.append(enc.feats(enc.frozen(augmented)))
            labels.append(y[chunk])
        print(f"\r  погодные признаки: {min(start + batch, len(indices))}/{len(indices)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return np.concatenate(feats), np.concatenate(labels)


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



# ---------- общая policy для validation/test/PhotoDino ----------

def triples_of(classes, ok_of):
    out = []
    for label in classes:
        typ, defect = split_label(label, None)
        out.append([typ, "ok" if ok_of.get(label, False) else "bad", defect or None])
    return out


def answer_norm(answer, available=True):
    answer = answer or {}
    return {"available": bool(available), "label": answer.get("label"), "answer": None,
            "description": str(answer.get("description") or ""), "error": answer.get("error"),
            "bad_json": bool(answer.get("bad_json")), "review": bool(answer.get("review"))}


def decide_matrix(P, classes, triples, answers, Q, *, cascade, w, cfg, support, tax):
    """Прогнать production policy по матрице вероятностей. Возвращает решения по кадрам."""
    from collections import Counter
    out = []
    n = len(P)
    for i in range(n):
        scores = [float(x) for x in P[i]]
        ans = answers[i] if answers is not None else None
        if cascade:
            dec = initial(classes, triples, scores, mode="cascade", cfg={**cfg, "qwen_weight": 0.0},
                          support=support, type_hint=None, class_policy="joint")
            vlm = answer_norm(ans) if ans is not None and dec.get("needs_vlm") else None
        else:
            dec = initial(classes, triples, scores, mode="legacy",
                          cfg={**cfg, "qwen_weight": w, "min_confidence": 0.0},
                          support=None, type_hint=None, class_policy="joint")
            vlm = answer_norm(ans) if ans is not None and w > 0 else None
        merged = finalize(dec, cfg={**cfg, "qwen_weight": 0.0 if cascade else w}, tax=tax, vlm=vlm)
        if merged is None:  # целый кадр не выбрасываем: это не находка с чужой рамкой
            merged = {"status": "review", "type": None, "damage": None, "label": None,
                      "description": "", "review_reasons": ["not_target"], "review": True}
        merged["review"] = merged.get("status") == "review" or bool(merged.get("review"))
        out.append(merged)
    return out


def _status_bad_metrics(y_bad, decisions):
    tp = sum(1 for gt, d in zip(y_bad, decisions) if gt and d.get("status") == "bad")
    fp = sum(1 for gt, d in zip(y_bad, decisions) if not gt and d.get("status") == "bad")
    fn = sum(1 for gt, d in zip(y_bad, decisions) if gt and d.get("status") != "bad")
    return tp, fp, fn


def add_candidate(rows, y_bad, P, classes, triples, answers, Q, *, name, base, cascade, w,
                  cfg, support, tax, cost):
    decisions = decide_matrix(P, classes, triples, answers, Q, cascade=cascade, w=w,
                              cfg=cfg, support=support, tax=tax)
    metrics = frame_metrics(y_bad, decisions, cost)
    tp, fp, fn = _status_bad_metrics(y_bad, decisions)
    top = P.argmax(1)
    macro = f1_score(y_bad, [d.get("status") == "bad" for d in decisions], average="binary", zero_division=0)
    rows.append({
        "name": name, "base": base, "cascade": bool(cascade), "w": float(w),
        "uses_vlm": bool(answers is not None and (w > 0 or cascade)),
        "serializable": True,
        "macro_f1": float(macro),
        "defect_precision": tp / (tp + fp) if tp + fp else 0.0,
        "defect_recall": tp / (tp + fn) if tp + fn else 0.0,
        **metrics,
        "decisions": decisions,
    })


def vlm_only_candidate(rows, y_bad, answers, classes, triples, cost):
    decisions = []
    answers = list(answers) if answers else [{"label": None, "error": "нет VLM"}] * len(y_bad)
    for answer in answers:
        label = answer.get("label")
        if label in classes and not answer.get("review"):
            index = classes.index(label)
            status, typ, damage = triples[index][1], triples[index][0], triples[index][2]
            decisions.append({"status": status, "type": typ, "damage": damage, "label": label,
                              "review": status == "review", "review_reasons": [],
                              "description": str(answer.get("description") or "")})
        else:
            reason = "vlm_error" if answer.get("error") or answer.get("bad_json") else "low_confidence"
            decisions.append({"status": "review", "type": None, "damage": None, "label": None,
                              "review": True, "review_reasons": [reason],
                              "description": str(answer.get("description") or "")})
    metrics = frame_metrics(y_bad, decisions, cost)
    tp, fp, fn = _status_bad_metrics(y_bad, decisions)
    rows.append({"name": "только Qwen", "base": "vlm", "cascade": False, "w": 1.0,
                 "uses_vlm": True, "serializable": False, "macro_f1": None,
                 "defect_precision": tp / (tp + fp) if tp + fp else 0.0,
                 "defect_recall": tp / (tp + fn) if tp + fn else 0.0,
                 **metrics, "decisions": decisions})

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
    ap.add_argument("--augment", type=int, default=0,
                    help="погодных копий на каждое train-фото для логрега (0 — выключено; [augment] config.toml)")
    ap.add_argument("--augment-seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.2, help="доля train для выбора связки")
    ap.add_argument("--no-qwen", action="store_true")
    ap.add_argument("--qwen-variant", default=None, help="вариант VLM из config.toml; по умолчанию qwen235")
    ap.add_argument("--test-format", choices=["auto", "images", "labeled"], default="auto")
    args = ap.parse_args()
    if args.no_qwen and args.qwen_variant:
        ap.error("--no-qwen и --qwen-variant взаимоисключающи")
    args.qwen_variant = args.qwen_variant or "qwen235"

    from run import load_config
    cfg = load_config()
    try:
        vcfg = resolve_variant(cfg, "photo_dino")
    except ValueError:
        vcfg = dict(cfg["variants"]["photo_dino"])
    offline = bool(vcfg.get("offline"))
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
    triples = triples_of(classes, ok_of)
    C = np.array(classes, dtype=object)
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
    answers_va = answers_te = None
    Q_va = Q_te = None
    if not args.no_qwen:
        from config_utils import check_offline_vlm
        if offline:
            check_offline_vlm(cfg, {"qwen": args.qwen_variant}, True)
        try:
            qcfg = resolve_variant(cfg, args.qwen_variant)
        except ValueError as exc:
            ap.error(f"нет варианта VLM {args.qwen_variant!r}: {exc}")
        qwen = Qwen(classes, out / "qwen.json", variant=args.qwen_variant, vcfg=qcfg, tax=cfg["taxonomy"])
        if not qwen.ready:
            ap.error(f"для {args.qwen_variant} нет ключа; запустите с --no-qwen или настройте key_env")
        qwen.submit([tr_paths[i] for i in va] + te_paths)  # сначала 20% — они нужны раньше
        log(f"Qwen ({args.qwen_variant}) запущен в фоне")

    # 1. Один проход по train: токены (обычные и отражённые) + признаки
    t = time.time()
    enc = Encoder(dev, args.size, args.k, offline=offline)
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

    # 1a. Погодные копии только для признаков логрега: оригиналы остаются с индексами 0..n-1,
    # поэтому fit/va, H/Hf и y продолжают работать без переиндексации.
    aug_cfg, Fa_fit, ya_fit = {}, None, None
    F_ext, y_ext, fit_ext = F, y, fit
    if args.augment:
        from run import load_config as _load_config
        aug_cfg = _load_config().get("augment", {})
        t = time.time()
        Fa_fit, ya_fit = augmented_features(enc, tr_paths, y, fit, args.augment, aug_cfg, args.augment_seed)
        F_ext = np.concatenate([F, Fa_fit])
        y_ext = np.concatenate([y, ya_fit])
        fit_ext = np.concatenate([fit, np.arange(len(F), len(F_ext), dtype=fit.dtype)])
        times["признаки аугментаций"] = time.time() - t
        log(f"погодные признаки: +{len(Fa_fit)} векторов ({times['признаки аугментаций'] / 60:.1f} мин)")

    # 2. Выбор связки на 20% через ту же policy, что будет в production
    t = time.time()
    clf, _ = logreg(F_ext, y_ext, fit_ext, n_cls, args.C)
    P_lr = lr_probs(clf, F[va], n_cls)
    net = train_tail(enc, H, Hf, y, fit, n_cls, args.epochs, args.lr)
    P_ft = np.concatenate([tail_probs(net, H[va[s:s + 64]].to(dev)) for s in range(0, len(va), 64)])
    del net
    times["обучение на 80%"] = time.time() - t
    bases = {"logreg": P_lr, "дообучение": P_ft, "логрег+дообучение": (P_lr + P_ft) / 2}
    if qwen:
        log("жду ответы Qwen на 20%")
        answers_va = qwen.get([tr_paths[i] for i in va])
        Q_va = np.asarray(qwen.onehot(answers_va, classes), dtype=float)

    from collections import Counter
    support_fit = dict(Counter(classes[int(y[i])] for i in fit))
    cost_cfg = cfg.get("cost") or {}
    selection_cfg = cfg.get("selection") or {}
    y_bad_va = [not ok_of[str(C[i])] for i in y[va]]
    rows = []
    base_names = {"logreg": "логрег", "дообучение": "дообучение", "логрег+дообучение": "логрег+дообучение"}
    for base, P in bases.items():
        weights = [0.0, 0.3, 0.5, 1.0] if qwen else [0.0]
        for w in weights:
            add_candidate(rows, y_bad_va, P, classes, triples, answers_va, Q_va,
                          name=base_names[base] + (f" + Qwen×{w}" if w else ""),
                          base=base, cascade=False, w=w, cfg={**vcfg, "min_confidence": 0.0,
                                                            "hint_mode": "none"}, support=None,
                          tax=cfg["taxonomy"], cost=cost_cfg)
        add_candidate(rows, y_bad_va, P, classes, triples, answers_va, Q_va,
                      name=base_names[base] + " каскад" + (" + Qwen" if qwen else ""),
                      base=base, cascade=True, w=0.0,
                      cfg={**vcfg, "min_support": int(vcfg.get("min_support", 2)), "hint_mode": "none"},
                      support=support_fit, tax=cfg["taxonomy"], cost=cost_cfg)
    if qwen:
        vlm_only_candidate(rows, y_bad_va, answers_va, classes, triples, cost_cfg)

    baseline_rows = selection_baselines(y_bad_va, cost_cfg)
    selection = select_candidate(rows, selection_cfg, baseline_rows)
    print(f"\nСвязки на {len(va)} отложенных фото (кандидаты, ограничения, cost):")
    for row in sorted(rows, key=lambda r: (r.get("operational_cost") is None,
                                           r.get("operational_cost") or 0.0,
                                           -(r.get("triage_recall") or 0.0))):
        chosen = selection.get("chosen") or {}
        mark = "→" if chosen.get("name") == row.get("name") else " "
        reason = "; ".join(row.get("reject_reasons") or [])
        print(f"  {mark} {row['name']:30} cost {row.get('operational_cost')} "
              f"triage R {row.get('triage_recall')} review {row.get('review_rate')} "
              f"support {row.get('support_bad')}/{row.get('support_ok')}" + (f"  [{reason}]" if reason else ""))

    def base_report(status):
        return {"status": status, "train": len(tr), "test": len(te), "classes": classes,
                "fit": len(fit), "validation": len(va),
                "split": f"StratifiedGroupKFold({round(1 / args.val)}) по groups_of",
                "taxonomy": cfg["taxonomy"],
                "taxonomy_sha256": taxonomy_fingerprint(cfg["taxonomy"]),
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "package_versions": package_versions(), "qwen_variant": args.qwen_variant if qwen else "",
                "selection": selection, "val_table": rows, "minutes": {k: round(v / 60, 1) for k, v in times.items()},
                "args": vars(args)}

    if selection["status"] != "ok":
        report = base_report("selection_failed")
        report["data_hashes"] = {}
        write_json_atomic(out / "report.json", report)
        log(f"ОТБОР НЕ ВЫПОЛНЕН: {selection.get('reason')}. Checkpoint и predictions не созданы; "
            f"таблица кандидатов в {out}/report.json")
        sys.exit(2)

    best = selection["chosen"]
    log(f"выбрана: {best['name']} (cost {best.get('operational_cost')}, triage R {best.get('triage_recall')}, "
        f"review {best.get('review_rate')})")

    # 3. Переобучение выбранного на всём train
    t = time.time()
    everything = np.arange(len(y))
    use_lr = best["base"] in ("logreg", "логрег+дообучение")
    use_ft = best["base"] in ("дообучение", "логрег+дообучение")
    clf = None
    if use_lr:
        if args.augment:  # теперь val законно входит в 100% train: досчитываем копии и для него
            Fa_va, ya_va = augmented_features(enc, tr_paths, y, va, args.augment, aug_cfg, args.augment_seed)
            F_all = np.concatenate([F, Fa_fit, Fa_va])
            y_all = np.concatenate([y, ya_fit, ya_va])
        else:
            F_all, y_all = F, y
        clf = logreg(F_all, y_all, np.arange(len(F_all)), n_cls, args.C)[0]
    net = train_tail(enc, H, Hf, y, everything, n_cls, args.epochs, args.lr) if use_ft else None
    import joblib
    policy_cfg = {**vcfg, "cascade": bool(best["cascade"]),
                  "qwen_weight": 0.0 if best["cascade"] else float(best["w"]),
                  "hint_mode": "none", "mode": "cascade" if best["cascade"] else "legacy",
                  "min_confidence": 0.0 if not best["cascade"] else float(vcfg.get("min_confidence", 0.0))}
    chosen_policy = semantic_policy(policy_cfg)
    train_files = list(tr_paths) + sorted(p for p in tr_root.rglob("*")
                                         if p.is_file() and p.suffix.lower() in TABLE_EXT)
    data_hashes = {"train": hash_files(train_files, [tr_root])}
    state = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "taxonomy": cfg["taxonomy"],
        "taxonomy_sha256": taxonomy_fingerprint(cfg["taxonomy"]),
        "size": args.size, "k": args.k, "classes": classes,
        "ok": {c: ok_of[c] for c in classes}, "logreg": clf,
        "tail": {key: value.cpu() for key, value in net.state_dict().items()} if net is not None else None,
        "qwen_weight": 0.0 if best["cascade"] else float(best["w"]),
        "qwen_variant": args.qwen_variant if qwen else "",
        "policy": chosen_policy,
        "selection_config": selection_cfg,
        "selection": {"chosen": best["name"], "constraints": selection["constraints"],
                      "support_bad": selection["support_bad"], "support_ok": selection["support_ok"]},
        "data_hashes": data_hashes,
        "package_versions": package_versions(),
        "training_args": {**vars(args), "support_fit": support_fit, "support_full":
                          dict(Counter(classes[int(y[i])] for i in y))},
    }
    joblib.dump(state, out / "photo_dino.joblib.tmp")
    (out / "photo_dino.joblib.tmp").replace(out / "photo_dino.joblib")
    del H, Hf
    times["обучение на 100%"] = time.time() - t
    log("переобучено на всём train")

    # 4. Test: тот же initial/finalize, что validation и PhotoDino
    t = time.time()
    Ps = []
    for imgs in enc.batches(te_paths, what="test"):
        h = enc.frozen(imgs)
        parts = ([lr_probs(clf, enc.feats(h), n_cls)] if use_lr else []) + ([tail_probs(net, h)] if use_ft else [])
        Ps.append(sum(parts) / len(parts))
    P = np.concatenate(Ps)
    answers_te = qwen.get(te_paths) if qwen else [{"label": None, "description": ""} for _ in te]
    Q_te = np.asarray(qwen.onehot(answers_te, classes), dtype=float) if qwen else None
    support_full = dict(Counter(classes[int(y[i])] for i in y))
    decisions = decide_matrix(P, classes, triples, answers_te, Q_te, cascade=bool(best["cascade"]),
                              w=float(best["w"]), cfg={**vcfg, "min_confidence": 0.0, "hint_mode": "none"},
                              support=support_full, tax=cfg["taxonomy"])
    times["test"] = time.time() - t
    bad = np.array([not ok_of[c] for c in classes])
    p_bad = P[:, bad].sum(1)
    rows_out = []
    for i, x in enumerate(te):
        d = decisions[i]
        status = d.get("status", "review")
        typ, defect = d.get("type"), d.get("damage")
        top = np.argsort(-P[i])[:3]
        rows_out.append({
            "Имя файла": x["file"],
            "Класс": d.get("label") if status in ("ok", "bad") else "",
            "Тип": typ or "", "Дефект": defect or "",
            "Статус": status, "Неисправен": status == "bad",
            "Требует проверки": status == "review",
            "Причины review": ";".join(d.get("review_reasons") or []),
            "P(дефект)": round(float(p_bad[i]), 3),
            "Уверенность": round(float(P[i].max()), 3),
            "Топ-3": "; ".join(f"{classes[j]} {P[i, j]:.2f}" for j in top),
            "Qwen: класс": answers_te[i].get("label"),
            "Описание (Qwen)": answers_te[i].get("description", ""),
            **({"Истинный класс": x["label"]} if te_labeled else {}), **x.get("extra", {}),
        })
    df = pd.DataFrame(rows_out)
    try:
        df.to_excel(out / "predictions.xlsx", index=False)
    except Exception as exc:  # noqa: BLE001 — openpyxl может отсутствовать; CSV остаётся
        log(f"predictions.xlsx не записан ({str(exc)[:120]}); есть predictions.csv")
    df.to_csv(out / "predictions.csv", index=False)
    snippet = f"""[variants.photo_dino]
kind = "photo_dino"
checkpoint = "{out / 'photo_dino.joblib'}"
qwen = "{args.qwen_variant if qwen else ''}"
cascade = {str(bool(best["cascade"])).lower()}
hint_mode = "none"
p_bad_min = {chosen_policy.get("p_bad_min")}
p_bad_max_ok = {chosen_policy.get("p_bad_max_ok")}
status_conf_min = {chosen_policy.get("status_conf_min")}
margin_min = {chosen_policy.get("margin_min")}
describe_bad = {str(bool(chosen_policy.get("describe_bad"))).lower()}
not_target = "{chosen_policy.get("not_target", "review")}"
min_support = {chosen_policy.get("min_support", 2)}
"""
    (out / "photo_dino_variant.toml").write_text(snippet, encoding="utf-8")
    report = base_report("ok")
    report.update({
        "chosen": best["name"], "chosen_policy": chosen_policy,
        "chosen_variant_snippet": snippet, "data_hashes": data_hashes,
    })
    if te_labeled:
        y_te = np.array([classes.index(x["label"]) for x in te])
        y_bad_te = [not ok_of[str(C[i])] for i in y_te]
        report["test_metrics"] = frame_metrics(y_bad_te, decisions, cost_cfg)
        report["classification_reference"] = score(C[y_te], C[P.argmax(1)], ok_of)
    write_json_atomic(out / "report.json", report)
    if qwen:
        qwen.close()
    print(f"\nнеисправных в test: {int(df['Неисправен'].sum())} из {len(df)}; "
          f"review: {int(df['Требует проверки'].sum())}; ok: {int((df['Статус'] == 'ok').sum())}")
    log(f"готово -> {out}/predictions.xlsx; время по этапам, мин: {report['minutes']}")


if __name__ == "__main__":
    main()
