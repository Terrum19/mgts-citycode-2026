"""Скачать модель с huggingface в weights/<имя> в несколько потоков (одно соединение у нас ~0.2 МБ/с).

    python src/fetch_weights.py facebook/dinov2-base
    python src/fetch_weights.py facebook/dinov2-base --threads 16

Берёт config.json, preprocessor_config.json и model.safetensors — этого хватает для from_pretrained(папка).
Докачивает: уже скачанные куски не качает заново.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["config.json", "preprocessor_config.json"]
CHUNK = 2 ** 20


def fetch_big(url: str, dest: Path, threads: int) -> None:
    head = requests.head(url, allow_redirects=False, timeout=60)  # HEAD на сам CDN у нас зависает
    head.raise_for_status()
    size, real = int(head.headers["x-linked-size"]), head.headers["location"]
    parts = dest.with_suffix(".parts")
    parts.mkdir(exist_ok=True)
    ranges = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK))]
    done = [0]
    t0 = time.time()

    def get(r):
        i, a, b = r
        f = parts / f"{i:05d}"
        if f.exists() and f.stat().st_size == b - a + 1:
            done[0] += b - a + 1
            return
        for attempt in range(8):
            try:
                resp = requests.get(real, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if resp.status_code == 206 and len(resp.content) == b - a + 1:
                    f.write_bytes(resp.content)
                    done[0] += b - a + 1
                    mb = done[0] / 2 ** 20
                    print(f"\r  {dest.name}: {mb:.0f}/{size / 2 ** 20:.0f} МБ, {mb / (time.time() - t0):.1f} МБ/с",
                          end="", file=sys.stderr)
                    return
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"кусок {i} не скачался")

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(get, ranges))
    with dest.open("wb") as out:
        for i, _, _ in ranges:
            out.write((parts / f"{i:05d}").read_bytes())
    for f in parts.iterdir():
        f.unlink()
    parts.rmdir()
    print(file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    out = ROOT / "weights" / args.repo.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{args.repo}/resolve/main"
    for name in SMALL:
        r = requests.get(f"{base}/{name}", timeout=60)
        if r.ok:
            (out / name).write_bytes(r.content)
    if not (out / "model.safetensors").exists():
        fetch_big(f"{base}/model.safetensors", out / "model.safetensors", args.threads)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
