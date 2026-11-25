"""Версия таксономии и воспроизводимость checkpoint.

Fingerprint считается по каноническому JSON только смысловых полей (object_types и
damage). [names] — presentation, порядок ключей TOML не влияет, тексты описаний влияют:
если организаторы меняют формулировку дефекта, старый classifier нельзя молча
использовать для нового списка.

Старые checkpoint без fingerprint поддерживаются в режиме ``warn`` (совместимость),
но final/preflight должны требовать ``error``.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMAT_VERSION = 2
PACKAGES = ("torch", "transformers", "scikit-learn", "numpy", "joblib", "Pillow", "timm")


def canonical_taxonomy(tax: Mapping[str, Any] | None) -> str:
    tax = tax or {}
    payload = {
        "object_types": {str(k): str(v) for k, v in sorted((tax.get("object_types") or {}).items())},
        "damage": {str(k): str(v) for k, v in sorted((tax.get("damage") or {}).items())},
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def taxonomy_fingerprint(tax: Mapping[str, Any] | None) -> str:
    return hashlib.sha256(canonical_taxonomy(tax).encode("utf-8")).hexdigest()


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(files: Sequence[Path], roots: Sequence[Path]) -> dict:
    """SHA-256 каждого файла относительно одного из roots + хэш канонического manifest.

    В manifest не попадают абсолютные пути машины: checkpoint должен быть воспроизводим
    на другой машине с той же раскладкой данных.
    """
    roots = [Path(r).resolve() for r in roots]
    entries: dict[str, str] = {}
    for raw in files:
        path = Path(raw).resolve()
        rel = path.name
        for root in roots:
            try:
                rel = path.relative_to(root).as_posix()
                break
            except ValueError:
                continue
        entries[rel] = sha256_file(path)
    manifest = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"files": entries, "count": len(entries), "manifest_sha256": hashlib.sha256(manifest).hexdigest()}


def validate_checkpoint(
    state: Mapping[str, Any] | None,
    tax: Mapping[str, Any] | None,
    *,
    legacy: str = "warn",
) -> dict:
    """Проверить формат и taxonomy fingerprint.

    legacy:
      * warn — старый checkpoint без fingerprint пропускается с предупреждением;
      * allow — пропускается молча (только для явного временного режима);
      * error — любой несовместимый/старый checkpoint останавливает загрузку.
    Несовпадение сохранённого fingerprint всегда фатально.
    """
    if legacy not in ("warn", "allow", "error"):
        raise ValueError("legacy должен быть warn, allow или error")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint не является словарём")
    current = taxonomy_fingerprint(tax)
    version = state.get("format_version")
    saved = state.get("taxonomy_sha256")
    if saved:
        if saved != current:
            raise ValueError(
                "Checkpoint обучен для другой таксономии.\n"
                "Переобучите classifier или выберите явный VLM-вариант.")
        return {"legacy": False, "format_version": version, "taxonomy_sha256": current, "matches": True}
    if version is not None and int(version) >= FORMAT_VERSION:
        raise ValueError("Checkpoint формата 2 не содержит taxonomy_sha256: файл повреждён или собран вручную")
    message = ("Checkpoint без taxonomy fingerprint (legacy). Совместимость не доказана; "
               "для финала переобучите classifier.")
    if legacy == "error":
        raise ValueError(message)
    if legacy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return {"legacy": True, "format_version": version, "taxonomy_sha256": current, "matches": False}
