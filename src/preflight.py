"""Preflight: проверить ссылки, локальные веса, checkpoint и offline-ограничения.

    python3 src/preflight.py --variant det_dino_offline --offline
    python3 src/preflight.py --variant photo_dino --load
    python3 src/preflight.py --variant det_dino --images data/raw/panoramas --limit 1

Без --load проверяются только файлы/импорты/ссылки. --load действительно загружает
модель, --images выполняет один inference. Отсутствие весов — честный failure, а не
повод подменить модель сетью.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import is_offline, is_local_url, resolve_variant  # noqa: E402
from taxonomy import taxonomy_fingerprint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "transformers", "sklearn", "joblib", "PIL")


def _check_modules(required: tuple[str, ...]) -> list[dict]:
    out = []
    for name in required:
        found = importlib.util.find_spec(name) is not None
        out.append({"check": f"import:{name}", "ok": found,
                    "detail": "найден" if found else "не установлен"})
    return out


def _check_model_dir(path: Path, backend: str) -> list[dict]:
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("config.json", (path / "config.json").is_file())
    weight = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).is_file():
            weight = path / name
            break
    index = next((p for p in (path / "model.safetensors.index.json", path / "pytorch_model.bin.index.json")
                  if p.is_file()), None)
    add("weights", bool(weight or index), str(weight or index or "нет model.safetensors/pytorch_model"))
    if index:
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            shards = sorted(set(data.get("weight_map", {}).values()))
            missing = [s for s in shards if not (path / s).is_file()]
            add("weight_shards", not missing, f"{len(shards)} shard(s), missing={missing[:3]}")
        except Exception as exc:  # noqa: BLE001
            add("weight_shards", False, str(exc)[:120])
    add("preprocessor", any((path / n).is_file() for n in
                            ("preprocessor_config.json", "processor_config.json")),
        "preprocessor/processor config")
    if backend == "grounding_dino":
        add("tokenizer", any((path / n).is_file() for n in
                             ("tokenizer.json", "tokenizer_config.json", "vocab.txt")), "tokenizer files")
    bad = [p.name for p in path.rglob("*") if p.name.endswith((".parts", ".incomplete", ".lock")) or
           (p.is_file() and p.stat().st_size == 0)]
    add("no_partial_files", not bad, ", ".join(bad[:5]))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="photo_dino")
    ap.add_argument("--offline", action="store_true", help="запретить сетевые fallback")
    ap.add_argument("--load", action="store_true", help="действительно загрузить модель")
    ap.add_argument("--images", type=Path, help="папка/файл для одного inference")
    ap.add_argument("--limit", type=int, default=1)
    args = ap.parse_args()
    if args.offline or os.environ.get("HF_HUB_OFFLINE"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import tomllib
    cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    checks: list[dict] = []
    try:
        vcfg = resolve_variant(cfg, args.variant)
    except ValueError as exc:
        checks.append({"check": "variant", "ok": False, "detail": str(exc)})
        print(json.dumps({"status": "failed", "checks": checks}, ensure_ascii=False, indent=1))
        sys.exit(1)
    kind = vcfg.get("kind", "vlm")
    offline = bool(args.offline or is_offline(vcfg, cfg))
    checks.append({"check": "variant", "ok": True, "detail": f"{args.variant} ({kind}), offline={offline}"})
    checks.append({"check": "taxonomy", "ok": bool(cfg.get("taxonomy")),
                   "detail": taxonomy_fingerprint(cfg.get("taxonomy")) if cfg.get("taxonomy") else "нет [taxonomy]"})
    required = ("PIL",) if kind in ("vlm", "boxes_vlm") else HEAVY
    checks += _check_modules(required)

    # VLM
    vlm_name = vcfg.get("vlm") or vcfg.get("qwen")
    if vlm_name or kind == "vlm":
        try:
            target = resolve_variant(cfg, str(vlm_name)) if vlm_name else vcfg
            url = str(target.get("base_url") or "")
            checks.append({"check": "vlm_url", "ok": bool(url), "detail": url or "нет base_url"})
            if offline and not is_local_url(url):
                checks.append({"check": "vlm_offline", "ok": False,
                               "detail": f"offline запрещает удалённую VLM {url}"})
            key_env = str(target.get("key_env") or "")
            from vlm import env as env_value
            checks.append({"check": "vlm_key", "ok": not (key_env and not env_value(key_env)),
                           "detail": f"key_env={key_env or 'нет'}"})
        except ValueError as exc:
            checks.append({"check": "vlm_ref", "ok": False, "detail": str(exc)})

    # Checkpoint / encoder / detector
    ckpt = vcfg.get("checkpoint")
    if ckpt:
        path = ROOT / str(ckpt)
        checks.append({"check": "checkpoint", "ok": path.is_file(), "detail": str(path)})
    encoder = vcfg.get("encoder") or (cfg.get("detector") or {}).get("encoder")
    if encoder and kind in ("det_dino", "boxes_dino"):
        enc = ROOT / str(encoder)
        checks += _check_model_dir(enc, "encoder") if enc.exists() else [
            {"check": "encoder", "ok": False, "detail": f"нет папки {enc}"}]
    if kind in ("det_dino", "det_vlm", "boxes_dino"):
        model = (cfg.get("detector") or {}).get("model", "")
        backend = (cfg.get("detector") or {}).get("backend", "mask2former")
        path = ROOT / str(model)
        if path.exists():
            checks += _check_model_dir(path, backend)
        elif offline:
            checks.append({"check": "detector_weights", "ok": False,
                           "detail": f"offline требует локальную папку, а model={model!r}"})
        else:
            checks.append({"check": "detector_weights", "ok": True,
                           "detail": f"HF id {model!r}: сетевая загрузка разрешена"})

    status = "ok" if all(c["ok"] for c in checks) else "failed"
    result = {"status": status, "variant": args.variant, "offline": offline, "checks": checks}

    if args.load and status == "ok":
        try:
            import variants
            model = variants.make(args.variant, cfg)
            result["loaded"] = type(model).__name__
            if args.images:
                from PIL import Image
                src = args.images
                paths = sorted(p for p in src.iterdir() if p.suffix.lower() in
                               {".jpg", ".jpeg", ".png", ".webp", ".bmp"}) if src.is_dir() else [src]
                paths = paths[:int(args.limit or 1)]
                result["images"] = []
                for path in paths:
                    pred = model.predict(Image.open(path).convert("RGB"))
                    result["images"].append({"file": str(path), "scope": pred.get("scope"),
                                             "status": (pred.get("frame") or {}).get("status"),
                                             "objects": len(pred.get("objects") or []),
                                             "seconds": pred.get("seconds")})
        except Exception as exc:  # noqa: BLE001 — preflight должен показать реальную ошибку
            result["status"] = "failed"
            result["load_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    elif args.load and status != "ok":
        result["load_error"] = "не загружаю: сначала исправьте failed-проверки"
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if result["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
