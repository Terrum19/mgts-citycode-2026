"""Признаки фото из готовой сети: вектор на фото, кэш на диске — считаем один раз.

Поддерживаются папки весов из src/fetch_weights.py:
  - DINOv2 (transformers): вектор = [CLS-токен, среднее по патчам];
  - модели timm (config.json с полем "architecture"): вектор = выход головы без классификатора.
size — сторона входа (фото сжимается в квадрат целиком, без обрезки краёв; у DINOv2 кратно 14).
tta — усреднить с отражённым по горизонтали фото.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class Embedder:
    def __init__(self, model: str, device: str = "cpu", batch: int = 16, size: int | None = None, tta: bool = False):
        import torch

        self.torch, self.path, self.device, self.batch, self.tta = torch, Path(model), device, batch, tta
        cfg = json.loads((self.path / "config.json").read_text())
        if "architecture" in cfg:  # timm
            import timm
            import torch
            self.kind = "timm"
            # pretrained=True у timm означает сетевой fallback; создаём архитектуру и
            # загружаем локальный state dict явно.
            self.model = timm.create_model(cfg["architecture"], pretrained=False, num_classes=0)
            weights = self.path / "model.safetensors"
            if weights.exists():
                from safetensors.torch import load_file
                state = load_file(str(weights))
            elif (self.path / "pytorch_model.bin").exists():
                state = torch.load(self.path / "pytorch_model.bin", map_location="cpu", weights_only=True)
            else:
                raise FileNotFoundError(f"нет локальных весов timm в {self.path}")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(f"timm state dict неполный: missing={missing[:5]}")
            dc = timm.data.resolve_data_config(model=self.model)
            self.size = size or dc["input_size"][-1]
            self.mean, self.std = dc["mean"], dc["std"]
        else:  # transformers (DINOv2)
            from transformers import AutoModel
            self.kind = "hf"
            self.model = AutoModel.from_pretrained(self.path, local_files_only=True)
            self.size = size or 224
            self.mean, self.std = IMAGENET
        self.model = self.model.eval().to(device)
        self.tag = f"{self.path.name}_s{self.size}{'_tta' if tta else ''}"

    def _key(self, p: Path) -> str:
        st = p.stat()
        return hashlib.sha1(f"{self.tag}|{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()

    def _tensor(self, imgs: list[Image.Image]):
        t = self.torch
        arr = np.stack([np.asarray(im.convert("RGB").resize((self.size, self.size), Image.BICUBIC), dtype=np.float32) / 255
                        for im in imgs])
        x = (arr - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        return t.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(self.device)

    def _forward(self, imgs: list[Image.Image]):
        x = self._tensor(imgs)
        if self.kind == "timm":
            return self.model(x)
        out = self.model(pixel_values=x).last_hidden_state
        return self.torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=1)

    def __call__(self, paths: list[Path], verbose: bool = True) -> np.ndarray:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE / f"emb_{self.tag}.npz"
        cache = dict(np.load(cache_file)) if cache_file.exists() else {}
        keys = [self._key(Path(p)) for p in paths]
        todo = [i for i, k in enumerate(keys) if k not in cache]
        t0 = time.time()
        for s in range(0, len(todo), self.batch):
            idx = todo[s:s + self.batch]
            imgs = [Image.open(paths[i]) for i in idx]
            with self.torch.inference_mode():
                feat = self._forward(imgs)
                if self.tta:
                    feat = feat + self._forward([ImageOps.mirror(im.convert("RGB")) for im in imgs])
            feat = self.torch.nn.functional.normalize(feat, dim=1).cpu().numpy()
            for i, f in zip(idx, feat):
                cache[keys[i]] = f.astype(np.float32)
            if verbose:
                print(f"\r  признаки {self.tag}: {min(s + self.batch, len(todo))}/{len(todo)}", end="", file=sys.stderr)
        if todo:
            np.savez(cache_file, **cache)
            if verbose:
                print(f"  ({(time.time() - t0) / len(todo) * 1000:.0f} мс/фото)", file=sys.stderr)
        return np.stack([cache[k] for k in keys])
