"""Атомарная запись файлов и JSON.

Промежуточные preds.json, qwen.json и checkpoint не должны оставаться битыми после
прерывания. Пишем во временный файл в той же директории, сбрасываем на диск и заменяем
целевой файл через os.replace. Внутрипроцессная блокировка защищает общий кэш от гонок
между потоками Qwen.
"""
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))


def write_json_atomic(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=1, default=float)
    write_bytes_atomic(path, text.encode("utf-8"))


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
