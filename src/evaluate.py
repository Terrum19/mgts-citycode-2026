"""Метрики: предсказания против ручной разметки.

    python src/evaluate.py data/labels.json runs/pano_dev__qwen235/preds.json [--split dev]

Правила (те же, что в docs/labeling.md):
- кадры с флагом bad_frame не оцениваются;
- объекты типа "other" — «не важно»: их не обязательно находить, и предсказание на их месте не ошибка;
- статус "unclear" — объект ищем, но в метриках статуса не участвует;
- объекты сопоставляются жадно по IoU внутри одного типа. Основной порог 0.5; на панорамах объекты
  мелкие и рамки неточные, поэтому показываем и порог 0.3.
"""
import argparse
import json
from collections import Counter

IGNORE_TYPES = {"other"}


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt: list, pr: list, thr: float, same_type: bool = True) -> list[tuple[int, int]]:
    """Жадное сопоставление по убыванию IoU -> пары (индекс gt, индекс pred)."""
    pairs = sorted(((iou(g["bbox"], p["bbox"]), i, j) for i, g in enumerate(gt) for j, p in enumerate(pr)
                    if not same_type or g["type"] == p["type"]), reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def prf(tp, fp, fn) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(labels: list[dict], preds: dict[str, dict], thr: float = 0.5) -> dict:
    det = {t: Counter() for t in (thr, 0.3)}   # tp / gt / pr по порогам
    per_type = {}                              # тип -> Counter(tp, gt, pr) при основном пороге
    repair_tp = repair_pred = review = 0
    any_type_tp = 0                            # нашли место объекта, но, возможно, с другим типом
    st, img, confusion, type_conf = Counter(), Counter(), Counter(), Counter()
    dmg_ok = dmg_n = n_bad_gt = 0
    bad_json = errors = skipped = 0
    secs, cost = [], 0.0
    per_image = []
    for lab in labels:
        if "bad_frame" in lab.get("frame_flags", []):
            skipped += 1
            continue
        p = preds.get(lab["file"], {"objects": [], "error": "нет предсказания"})
        bad_json += bool(p.get("bad_json"))
        errors += bool(p.get("error"))
        cost += p.get("cost", 0.0)
        if "seconds" in p:
            secs.append(p["seconds"])
        gt = [o for o in lab["objects"] if o["type"] not in IGNORE_TYPES]
        ign = [o for o in lab["objects"] if o["type"] in IGNORE_TYPES]
        n_bad_gt += sum(o["status"] == "bad" for o in gt)
        # предсказание на месте «не важного» объекта не считаем, если рядом нет настоящего
        pr = [q for q in p["objects"]
              if not any(iou(q["bbox"], o["bbox"]) >= 0.3 for o in ign)
              or any(iou(q["bbox"], g["bbox"]) >= 0.3 for g in gt)]

        for t in det:
            det[t]["tp"] += len(match(gt, pr, t))
            det[t]["gt"] += len(gt)
            det[t]["pr"] += len(pr)
        m = match(gt, pr, thr)
        for o in gt:
            per_type.setdefault(o["type"], Counter())["gt"] += 1
        for q in pr:
            per_type.setdefault(q["type"], Counter())["pr"] += 1
        for i, _ in m:
            per_type[gt[i]["type"]]["tp"] += 1
        ma = match(gt, pr, thr, same_type=False)
        any_type_tp += len(ma)
        for i, j in ma:
            type_conf[(gt[i]["type"], pr[j]["type"])] += 1

        bad_gt = [o for o in gt if o["status"] == "bad"]
        bad_pr = [o for o in pr if o["status"] == "bad"]
        confirmed = match(bad_gt, bad_pr, thr)
        repair_tp += len(confirmed)
        repair_pred += len(bad_pr)
        review += sum(o["status"] == "review" for o in pr)
        status_err = 0
        for i, j in m:
            g, q = gt[i], pr[j]
            if g["status"] == "unclear":
                continue
            st[(g["status"], q["status"])] += 1
            status_err += g["status"] != q["status"]
            confusion[(g["damage"] or "ok", q["damage"] or "ok")] += 1
            if g["status"] == "bad" and q["status"] == "bad":
                dmg_n += 1
                dmg_ok += g["damage"] == q["damage"]
        g_bad = any(o["status"] == "bad" for o in gt)
        p_bad = any(o["status"] == "bad" for o in pr)
        img[(g_bad, p_bad)] += 1
        per_image.append({"file": lab["file"], "fn": len(gt) - len(m), "fp": len(pr) - len(m),
                          "status_err": status_err, "missed_bad": len(confirmed) < len(bad_gt)})

    n = len(labels) - skipped
    out = {"images": n, "skipped_bad_frames": skipped, "gt_objects": det[thr]["gt"],
           "gt_bad_objects": n_bad_gt, "gt_bad_images": img[(True, True)] + img[(True, False)]}
    for t, c in det.items():
        sfx = "" if t == thr else f"@{t}"
        out[f"det_precision{sfx}"] = c["tp"] / c["pr"] if c["pr"] else 0.0
        out[f"det_recall{sfx}"] = c["tp"] / c["gt"] if c["gt"] else 0.0
    out["det_recall_anytype"] = any_type_tp / det[thr]["gt"] if det[thr]["gt"] else 0.0
    ip, ir, if1 = prf(img[(True, True)], img[(False, True)], img[(True, False)])
    sp, sr, sf1 = prf(st[("bad", "bad")], st[("ok", "bad")], st[("bad", "ok")] + st[("bad", "review")])
    rp, rr, rf = prf(repair_tp, repair_pred - repair_tp, n_bad_gt - repair_tp)
    out |= {
        "repair_tp": repair_tp, "repair_fp": repair_pred - repair_tp, "repair_fn": n_bad_gt - repair_tp,
        "repair_precision": rp, "repair_recall": rr, "repair_f1": rf, "review_objects": review,
        "img_bad_recall": ir, "img_bad_precision": ip, "img_bad_f1": if1,
        "img_accuracy": (img[(True, True)] + img[(False, False)]) / n if n else 0.0,
        "obj_bad_recall": sr, "obj_bad_precision": sp, "obj_bad_f1": sf1,
        "damage_acc": dmg_ok / dmg_n if dmg_n else 0.0,
        "bad_json_rate": bad_json / n if n else 0.0, "error_rate": errors / n if n else 0.0,
        "sec_per_image": sum(secs) / len(secs) if secs else 0.0,
        "cost_usd": cost,
        "per_type": {t: {"gt": c["gt"], "pred": c["pr"],
                         "precision": round(c["tp"] / c["pr"], 3) if c["pr"] else 0.0,
                         "recall": round(c["tp"] / c["gt"], 3) if c["gt"] else 0.0}
                     for t, c in sorted(per_type.items())},
        "type_confusion": {f"{a} -> {b}": c for (a, b), c in sorted(type_conf.items()) if a != b},
        "damage_confusion": {f"{a} -> {b}": c for (a, b), c in sorted(confusion.items())},
        # худшие кадры для разбора глазами: сначала пропущенный bad, потом по сумме ошибок
        "worst": sorted((x for x in per_image if x["fn"] + x["fp"] + x["status_err"] or x["missed_bad"]),
                        key=lambda x: (x["missed_bad"], x["fn"] + x["fp"] + x["status_err"]), reverse=True)[:20],
    }
    return out


def load_labels(path, split: str | None = None) -> list[dict]:
    labels = json.load(open(path))
    return [x for x in labels if split is None or x.get("split") == split]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("preds")
    ap.add_argument("--split")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    m = evaluate(load_labels(args.labels, args.split), json.load(open(args.preds)), args.iou)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
