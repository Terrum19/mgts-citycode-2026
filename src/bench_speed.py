"""Замер скорости нашей связки на этой машине: признаки DINOv2, логрег, дообучение хвоста, (опц.) Qwen.

Ничего, кроме этого файла, не нужно — данные не нужны (фото генерируются: на скорость содержимое не влияет).

Установка (один раз):
    python3 -m venv .venv && source .venv/bin/activate
    pip install torch transformers scikit-learn pillow numpy requests

Запуск:
    python bench_speed.py                    # всё по умолчанию (~5 мин)
    python bench_speed.py --quick            # быстрая проверка, что всё работает (~2 мин)
    python bench_speed.py --qwen 8           # + замер Qwen (нужен OPENROUTER_API_KEY в окружении)

Модель: если есть папка weights/dinov2-base — берётся она, иначе скачивается facebook/dinov2-base (~350 МБ).
В конце печатается таблица и оценка времени на весь финал. Её и надо прислать.
"""
import argparse
import base64
import io
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MEAN, STD = np.array((0.485, 0.456, 0.406), np.float32), np.array((0.229, 0.224, 0.225), np.float32)


def pick_device(want: str) -> str:
    if want != "auto":
        return want
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def fake_photos(n, w=720, h=720, seed=0):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def to_tensor(imgs, size, dev):
    arr = np.stack([np.asarray(im.resize((size, size), Image.BICUBIC), np.float32) / 255 for im in imgs])
    return torch.from_numpy((arr - MEAN) / STD).permute(0, 3, 1, 2).contiguous().to(dev)


def block(layer, h):
    o = layer(h)
    return o[0] if isinstance(o, (tuple, list)) else o


def timed(fn, dev):
    sync(dev)
    t = time.perf_counter()
    out = fn()
    sync(dev)
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--n", type=int, default=64, help="сколько фото гонять для замера признаков")
    ap.add_argument("--ft-n", type=int, default=128, help="сколько фото в замере дообучения")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--qwen", type=int, default=0, help="сколько запросов к Qwen (0 = не мерить)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train-photos", type=int, default=500, help="для итоговой оценки: фото в train на финале")
    ap.add_argument("--test-photos", type=int, default=650, help="для итоговой оценки: фото в test на финале")
    args = ap.parse_args()
    if args.quick:
        args.n, args.ft_n = 16, 32

    from transformers import AutoModel

    dev = pick_device(args.device)
    local = Path(__file__).resolve().parent.parent / "weights/dinov2-base"
    src = str(local) if local.exists() else "facebook/dinov2-base"
    print(f"машина: {platform.platform()} | {platform.processor() or platform.machine()} | torch {torch.__version__} | устройство {dev}")
    print(f"модель: {src}")
    model = AutoModel.from_pretrained(src).eval().to(dev)
    photos = fake_photos(max(args.n, args.ft_n))
    res = {}

    # 1. Признаки (то, что нужно для логрега)
    for size in (224, 448):
        with torch.inference_mode():
            model(pixel_values=to_tensor(photos[:2], size, dev))  # прогрев
            def run():
                for s in range(0, args.n, args.batch):
                    model(pixel_values=to_tensor(photos[s:s + args.batch], size, dev))
            _, t = timed(run, dev)
        res[f"признаки {size}"] = t / args.n
        print(f"  признаки DINOv2 {size}: {t / args.n * 1000:.0f} мс/фото")

    # 2. Логрег
    from sklearn.linear_model import LogisticRegression
    X = np.random.rand(args.train_photos, 1536).astype(np.float32)
    y = np.random.randint(0, 7, args.train_photos)
    t = time.perf_counter()
    LogisticRegression(C=32, max_iter=5000, class_weight="balanced").fit(X, y)
    res["логрег обучение"] = time.perf_counter() - t
    print(f"  обучение логрега на {args.train_photos} фото: {res['логрег обучение']:.1f} с")

    # 3. Дообучение хвоста (2 последних блока) на 448
    size, k = 448, 2
    cut = len(model.encoder.layer) - k
    with torch.inference_mode():
        def tokens():
            out = []
            for s in range(0, args.ft_n, args.batch):
                h = model.embeddings(to_tensor(photos[s:s + args.batch], size, dev))
                for layer in model.encoder.layer[:cut]:
                    h = block(layer, h)
                out.append(h)
            return torch.cat(out)
        H, t = timed(tokens, dev)
    res["токены 448"] = t / args.ft_n
    print(f"  подготовка к дообучению (448): {t / args.ft_n * 1000:.0f} мс/фото")

    import copy
    blocks = copy.deepcopy(model.encoder.layer[cut:]).train()
    norm = copy.deepcopy(model.layernorm)
    head = torch.nn.Linear(2 * model.config.hidden_size, 7).to(dev)
    params = list(blocks.parameters()) + list(norm.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=2e-5)
    yy = torch.randint(0, 7, (args.ft_n,), device=dev)
    H = H.detach().clone()

    def epoch():
        for s in range(0, args.ft_n, 32):
            h = H[s:s + 32]
            for b in blocks:
                h = block(b, h)
            h = norm(h)
            loss = torch.nn.functional.cross_entropy(head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)), yy[s:s + 32])
            opt.zero_grad()
            loss.backward()
            opt.step()
    epoch()  # прогрев
    _, t = timed(epoch, dev)
    res["эпоха на фото"] = t / args.ft_n
    print(f"  дообучение: {t / args.ft_n * 1000:.0f} мс/фото за эпоху")

    # 4. Qwen (опционально)
    if args.qwen:
        import requests
        from concurrent.futures import ThreadPoolExecutor
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            print("  Qwen: нет OPENROUTER_API_KEY — пропускаю")
        else:
            buf = io.BytesIO()
            photos[0].resize((512, 512)).save(buf, "JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            def ask(_):
                t0 = time.perf_counter()
                requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": "qwen/qwen3-vl-235b-a22b-instruct", "max_tokens": 50, "messages": [{"role": "user", "content": [
                                  {"type": "text", "text": "Что на фото? Ответь одним словом."},
                                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]})
                return time.perf_counter() - t0
            t0 = time.perf_counter()
            with ThreadPoolExecutor(4) as ex:
                lat = list(ex.map(ask, range(args.qwen)))
            res["qwen поток"] = (time.perf_counter() - t0) / args.qwen
            print(f"  Qwen: задержка {np.median(lat):.1f} с, в 4 потока {res['qwen поток']:.1f} с/фото")

    # Итог: оценка на финал
    tr, te, ep = args.train_photos, args.test_photos, 12
    feats = (tr + te) * res["признаки 448"]
    ft_tok = (2 * tr + te) * res["токены 448"]  # train + отражённые train + test
    ft_train = ep * tr * res["эпоха на фото"]
    q = te * res.get("qwen поток", 1.5)
    print(f"\nОценка на финал (train {tr}, test {te}, {ep} эпох):")
    print(f"  признаки 448 для логрега:  {feats / 60:5.1f} мин")
    print(f"  логрег:                    {res['логрег обучение'] / 60:5.1f} мин")
    print(f"  дообучение, подготовка:    {ft_tok / 60:5.1f} мин")
    print(f"  дообучение, {ep} эпох:       {ft_train / 60:5.1f} мин   (с проверкой 80/20 — примерно вдвое больше)")
    print(f"  Qwen на test:              {q / 60:5.1f} мин{'' if 'qwen поток' in res else '   (оценка 1.5 с/фото, не мерили)'}")
    print(f"  ВСЕГО: {(feats + res['логрег обучение'] + ft_tok + ft_train) / 60:.0f} мин без Qwen "
          f"(Qwen идёт параллельно, по сети)")


if __name__ == "__main__":
    main()
