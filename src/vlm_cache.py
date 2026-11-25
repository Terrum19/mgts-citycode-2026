"""Версионированный кэш ответов VLM вместо ключа «путь к файлу».

Ключ — SHA-256 канонического payload: variant, VLM fingerprint, SHA-256 фактических
JPEG-байтов (после RGB+resize), SHA-256 фактического промпта, labels/hint/examples и
версия схемы. Поэтому замена изображения по тому же пути, смена модели/промпта/таксономии
или variant не переиспользуют старый ответ.

Ошибки API, битый JSON и невалидный class не кэшируются. Explicit review (class=null,
review=true) — валидный кэшируемый ответ. Запись атомарная.
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from atomic import read_json, write_json_atomic

CACHE_SCHEMA = 2


def cache_key(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cacheable(answer: Mapping[str, Any] | None) -> bool:
    if not isinstance(answer, Mapping):
        return False
    if answer.get("error") or answer.get("bad_json"):
        return False
    label = answer.get("label")
    if label is not None and str(label).strip():
        return True
    return bool(answer.get("review"))


class Qwen:
    """Фоновые ответы VLM с дисковым кэшем. Сеть не трогается, пока не вызван submit/get."""

    def __init__(self, labels: list[str], cache: Path, *, variant: str, vcfg: dict, tax: dict,
                 vlm: Any | None = None):
        if vlm is None:
            from vlm import VLM
            vlm = VLM(vcfg, tax)
        self.labels, self.cache_path, self.variant, self.vcfg, self.tax = labels, Path(cache), variant, vcfg, tax
        self.vlm = vlm
        self.pool = ThreadPoolExecutor(vcfg.get("workers", 4))
        self.fut: dict[str, Any] = {}
        self.lock = threading.Lock()
        raw = read_json(self.cache_path, default={})
        entries = raw.get("entries") if isinstance(raw, dict) and raw.get("schema_version") == CACHE_SCHEMA else {}
        self.entries: dict[str, dict] = dict(entries) if isinstance(entries, dict) else {}
        self.dirty = False

    @property
    def ready(self) -> bool:
        key_env = str(self.vcfg.get("key_env") or "")
        return not (key_env and not getattr(self.vlm, "key", ""))

    def _prompt(self, hint=None, examples=None) -> str:
        from vlm_classify import build_prompt
        return build_prompt(self.labels, hint, examples,
                            prompt_mode=getattr(self.vlm, "prompt_mode", "default"))

    def _key_payload(self, path: Path, hint=None, examples=None) -> tuple[str, dict]:
        from PIL import Image
        from vlm import to_jpeg
        with Image.open(path) as img:
            jpeg = to_jpeg(img.convert("RGB"), self.vlm.max_side)
        prompt = self._prompt(hint, examples)
        payload = {
            "schema_version": CACHE_SCHEMA, "variant": self.variant,
            "vlm": self.vlm.fingerprint(),
            "image_sha256": hashlib.sha256(jpeg).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "labels": list(self.labels), "hint": hint, "examples": examples,
        }
        return cache_key(payload), payload

    def key_for(self, path: Path, hint=None, examples=None) -> str:
        return self._key_payload(path, hint, examples)[0]

    def _one(self, path: Path, key: str) -> dict:
        from PIL import Image
        from vlm_classify import classify
        try:
            with Image.open(path) as img:
                return classify(self.vlm, img, self.labels)
        except Exception as exc:  # noqa: BLE001 — один сбой не должен ронять прогон
            return {"label": None, "description": "", "error": str(exc)[:300], "bad_json": False,
                    "review": False, "review_reason": None, "cost": 0.0, "seconds": 0.0}

    def submit(self, paths: Sequence[Path]) -> None:
        for path in map(Path, paths):
            key, _ = self._key_payload(path)
            with self.lock:
                if key in self.entries and cacheable(self.entries[key].get("response")):
                    continue
                if key in self.fut:
                    continue
                self.fut[key] = self.pool.submit(self._one, path, key)

    def get(self, paths: Sequence[Path]) -> list[dict]:
        out: list[dict] = []
        new = False
        for path in map(Path, paths):
            key, meta = self._key_payload(path)
            with self.lock:
                entry = self.entries.get(key)
                if entry and cacheable(entry.get("response")):
                    out.append(dict(entry["response"]))
                    continue
                fut = self.fut.pop(key, None)
            if fut is None:
                fut = self.pool.submit(self._one, path, key)
            response = fut.result()
            if cacheable(response):
                with self.lock:
                    self.entries[key] = {
                        "response": response, "variant": self.variant,
                        "image_sha256": meta["image_sha256"],
                        "prompt_sha256": meta["prompt_sha256"],
                    }
                    self.dirty = True
                    new = True
            out.append(dict(response))
        if new:
            self.save()
        return out

    def save(self) -> None:
        with self.lock:
            payload = {"schema_version": CACHE_SCHEMA, "entries": dict(self.entries)}
            self.dirty = False
        write_json_atomic(self.cache_path, payload)

    def onehot(self, answers: Sequence[Mapping[str, Any]], classes: Sequence[str]) -> list[list[float]]:
        index = {str(name): i for i, name in enumerate(classes)}
        out = [[0.0] * len(classes) for _ in answers]
        for row, answer in zip(out, answers):
            label = answer.get("label") if isinstance(answer, Mapping) else None
            if label in index:
                row[index[str(label)]] = 1.0
        return out

    def close(self) -> None:
        self.pool.shutdown(wait=False)
