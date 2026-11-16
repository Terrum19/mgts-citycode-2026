"""Координатор: сливает архивы от команды в data/raw/panoramas и обновляет общий список снятых точек.

    python3 src/collect/pano/merge.py panoramas_petya_*.tar.gz panoramas_vasya_*.tar.gz
    python3 src/collect/pano/merge.py            # только пересобрать configs/panoramas_done.txt из своего meta.jsonl

После слияния закоммить configs/panoramas_done.txt — друзья сделают git pull и не будут снимать то же самое.
"""
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "data/raw/panoramas"
DONE = ROOT / "configs/panoramas_done.txt"


def read_meta(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DIR / "meta.jsonl"
    known = {r["id"]: r for r in read_meta(meta_path)}
    added = 0
    for archive in sys.argv[1:]:
        who = Path(archive).name.split("_")[1] if "_" in Path(archive).name else "?"
        with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive) as tar:
            # только ожидаемые файлы: meta.jsonl и *.jpg без путей (защита от ../ в архиве)
            members = [m for m in tar.getmembers()
                       if m.isfile() and "/" not in m.name and (m.name == "meta.jsonl" or m.name.endswith(".jpg"))]
            tar.extractall(tmp, members=members)
            for r in read_meta(Path(tmp) / "meta.jsonl"):
                if r["id"] in known:
                    continue
                if r.get("file"):
                    src = Path(tmp) / r["file"]
                    if not src.exists():
                        print(f"  ! {who}: нет файла {r['file']}, пропускаю")
                        continue
                    (DIR / r["file"]).write_bytes(src.read_bytes())
                r["collected_by"] = who
                known[r["id"]] = r
                added += 1
        print(f"{archive}: ок")

    meta_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in known.values()))
    ids = sorted(known)
    DONE.write_text("# точки, уже обработанные командой (снятые и без панорамы); генерирует merge.py\n" + "\n".join(ids) + "\n")
    shot = sum(1 for r in known.values() if r.get("file"))
    by_type: dict[str, int] = {}
    for r in known.values():
        if r.get("file"):
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"добавлено {added}; всего обработано точек {len(ids)}, с кадрами {shot}: {by_type}")
    print(f"-> {DONE.relative_to(ROOT)} (закоммить его)")


if __name__ == "__main__":
    main()
