"""Метрики: предсказания против ручной разметки (оба файла в нашем формате).

    python src/evaluate.py data/smoke/manholes/labels.json runs/<имя>/preds.json
"""
import json
import sys
from collections import Counter


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt: list, pr: list, thr: float = 0.5) -> list[tuple[int, int]]:
    """Жадное сопоставление по убыванию IoU."""
    pairs = sorted(((iou(g["bbox"], p["bbox"]), i, j) for i, g in enumerate(gt) for j, p in enumerate(pr)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j); out.append((i, j))
    return out


def evaluate(labels: list[dict], preds: dict[str, dict], thr: float = 0.5) -> dict:
    tp = n_gt = n_pr = 0
    st = Counter()          # статус на сопоставленных объектах
    img = Counter()         # статус на уровне фото: есть ли хоть один bad
    dmg_ok = dmg_n = 0
    confusion = Counter()
    bad_json = errors = 0
    secs = []
    for lab in labels:
        p = preds.get(lab["file"], {"objects": []})
        bad_json += bool(p.get("bad_json")); errors += bool(p.get("error"))
        if "seconds" in p:
            secs.append(p["seconds"])
        gt, pr = lab["objects"], p["objects"]
        n_gt += len(gt); n_pr += len(pr)
        m = match(gt, pr, thr)
        tp += len(m)
        for i, j in m:
            g, q = gt[i], pr[j]
            st[(g["status"], q["status"])] += 1
            confusion[(g["damage"] or "ok", q["damage"] or "ok")] += 1
            if g["status"] == "bad" and q["status"] == "bad":
                dmg_n += 1; dmg_ok += g["damage"] == q["damage"]
        g_bad = any(o["status"] == "bad" for o in gt)
        p_bad = any(o["status"] == "bad" for o in pr)
        img[(g_bad, p_bad)] += 1

    def prf(tp_, fp_, fn_):
        p_ = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r_ = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return p_, r_, (2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)

    ip, ir, if1 = prf(img[(True, True)], img[(False, True)], img[(True, False)])
    sp, sr, sf1 = prf(st[("bad", "bad")], st[("ok", "bad")], st[("bad", "ok")])
    n = len(labels)
    return {
        "images": n,
        "det_precision": tp / n_pr if n_pr else 0.0,
        "det_recall": tp / n_gt if n_gt else 0.0,
        "img_bad_recall": ir, "img_bad_precision": ip, "img_bad_f1": if1,
        "img_accuracy": (img[(True, True)] + img[(False, False)]) / n if n else 0.0,
        "obj_bad_recall": sr, "obj_bad_f1": sf1,
        "damage_acc": dmg_ok / dmg_n if dmg_n else 0.0,
        "bad_json_rate": bad_json / n if n else 0.0, "error_rate": errors / n if n else 0.0,
        "sec_per_image": sum(secs) / len(secs) if secs else 0.0,
        "confusion": {f"{a} -> {b}": c for (a, b), c in sorted(confusion.items())},
    }


def main() -> None:
    labels = json.load(open(sys.argv[1]))
    preds = json.load(open(sys.argv[2]))
    print(json.dumps(evaluate(labels, preds), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
