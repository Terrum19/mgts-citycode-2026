"""Разрешение вариантов config.toml: extends без дублирования и offline-ограничения."""
from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlparse

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def resolve_variant(cfg: Mapping[str, Any], name: str) -> dict:
    """Вернуть вариант с раскрытым одним или несколькими extends."""
    variants = cfg.get("variants") or {}
    if name not in variants:
        raise ValueError(f"неизвестный вариант {name!r}")
    if not isinstance(variants[name], Mapping):
        raise ValueError(f"[variants.{name}] должен быть таблицей")

    def resolve(key: str, seen: tuple[str, ...]) -> dict:
        if key in seen:
            raise ValueError(f"цикл extends: {' -> '.join(seen + (key,))}")
        raw = variants.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"[variants.{key}] отсутствует или не таблица")
        parent = raw.get("extends")
        if not parent:
            return dict(raw)
        if parent not in variants:
            raise ValueError(f"[variants.{key}] extends неизвестного {parent!r}")
        base = resolve(str(parent), seen + (key,))
        merged = {**base, **raw}
        merged.pop("extends", None)
        return merged

    return resolve(name, ())


def is_offline(vcfg: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> bool:
    return bool(vcfg.get("offline") or (cfg or {}).get("offline"))


def is_local_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url))
    except ValueError:
        return False
    if parsed.scheme in ("", "file"):
        return True
    return (parsed.hostname or "") in LOCAL_HOSTS


def check_offline_vlm(cfg: Mapping[str, Any], vcfg: Mapping[str, Any], offline: bool) -> None:
    """Offline-вариант не должен молча ходить в удалённую VLM."""
    if not offline:
        return
    name = vcfg.get("vlm") or vcfg.get("qwen")
    target = resolve_variant(cfg, str(name)) if name else dict(vcfg)
    url = str(target.get("base_url") or "")
    if url and not is_local_url(url):
        raise ValueError(f"offline-вариант использует удалённую VLM {name or 'self'!r} ({url}); "
                         "разрешены только отсутствие VLM и localhost")


def detector_config(cfg: Mapping[str, Any], vcfg: Mapping[str, Any]) -> dict:
    base = dict(cfg.get("detector") or {})
    if is_offline(vcfg, cfg):
        base["offline"] = True
    return base
