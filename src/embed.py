"""Признаки фото из готовой сети (DINOv2): вектор на фото, кэш на диске — считаем один раз.

Вектор = [CLS-токен, среднее по патчам], нормированный. На таких векторах линейный классификатор
обучается за секунды и на десятках примеров на класс.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"


class Embedder:
    def __init__(self, model: str = "facebook/dinov2-base", device: str = "cpu", batch: int = 16):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch, self.name, self.device, self.batch = torch, model, device, batch
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval().to(device)

    def _key(self, p: Path) -> str:
        st = p.stat()
        return hashlib.sha1(f"{self.name}|{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()

    def __call__(self, paths: list[Path], verbose: bool = True) -> np.ndarray:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE / f"emb_{Path(self.name).name}.npz"
        cache = dict(np.load(cache_file)) if cache_file.exists() else {}
        keys = [self._key(Path(p)) for p in paths]
        todo = [i for i, k in enumerate(keys) if k not in cache]
        t0 = time.time()
        for s in range(0, len(todo), self.batch):
            idx = todo[s:s + self.batch]
            imgs = [Image.open(paths[i]).convert("RGB") for i in idx]
            with self.torch.inference_mode():
                out = self.model(**self.proc(images=imgs, return_tensors="pt").to(self.device)).last_hidden_state
            feat = self.torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=1)
            feat = self.torch.nn.functional.normalize(feat, dim=1).cpu().numpy()
            for i, f in zip(idx, feat):
                cache[keys[i]] = f.astype(np.float32)
            if verbose:
                print(f"\r  признаки: {min(s + self.batch, len(todo))}/{len(todo)}", end="", file=sys.stderr)
        if todo:
            np.savez(cache_file, **cache)
            if verbose:
                print(f"  ({(time.time() - t0) / len(todo) * 1000:.0f} мс/фото)", file=sys.stderr)
        return np.stack([cache[k] for k in keys])
