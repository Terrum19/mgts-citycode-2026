"""Разбить папку формата финала на train/test по группам (симлинки на фото + свой Excel в каждой папке).

    python src/poc/split_folders.py data/sim_final --test 0.5     # -> data/sim_final_train, data/sim_final_test
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classify import groups_of  # noqa: E402
from folders import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--test", type=float, default=0.5)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    items = load(root)
    y = [x["label"] for x in items]
    k = round(1 / args.test)
    _, te = next(StratifiedGroupKFold(k, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups_of(items)))
    te = set(te)
    for part in ("train", "test"):
        out = root.parent / f"{root.name}_{part}"
        rows = defaultdict(list)
        for i, x in enumerate(items):
            if (i in te) != (part == "test"):
                continue
            dst = out / x["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(root / x["file"])
            rows[dst.parent].append({"Имя файла": dst.name, "Описание дефекта": x["description"],
                                     **{k: v for k, v in x["extra"].items() if k not in ("Имя файла", "Описание дефекта")}})
        for d, r in rows.items():
            pd.DataFrame(r).to_excel(d / "описание.xlsx", index=False)
        print(f"{out}: {sum(len(r) for r in rows.values())} фото")


if __name__ == "__main__":
    main()
