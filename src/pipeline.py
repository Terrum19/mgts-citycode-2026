"""One entry point for training, prediction and evaluation. Run --help or --list."""
import argparse
import copy
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description="Обучение и запуск маршрутов МГТС")
    p.add_argument("--variant", default="det_dino", help="вариант config.toml или final")
    p.add_argument("--list", action="store_true")
    p.add_argument("--config", type=Path, default=ROOT / "config.toml")
    p.add_argument("--train", choices=["never", "if-missing", "always"], default="never")
    p.add_argument("--training", choices=["auto", "head", "tail"], default="auto")
    p.add_argument("--train-images", type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--split", help="часть разметки для обучения, например dev")
    p.add_argument("--images", type=Path, help="фото для предсказаний / test для final")
    p.add_argument("--boxes", type=Path, help="рамки организатора или preds.json предыдущего запуска det_dino")
    p.add_argument("--meta", type=Path)
    p.add_argument("--dataset", help="оценка набора из datasets в конфиге вместо predict")
    p.add_argument("--checkpoint", type=Path, help="путь к модели; при обучении будет записан сюда")
    p.add_argument("--out", type=Path, default=Path("out/pipeline"))
    p.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--qwen", action="store_true", help="включить Qwen при обучении целых фото")
    p.add_argument("--test-format", choices=["auto", "images", "labeled"], default="images")
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true", help="проверить параметры и показать этапы без запуска")
    return p


def prepare(a, cfg):
    cfg = copy.deepcopy(cfg)
    final = a.variant == "final"
    name = "photo_dino" if final else a.variant
    if name not in cfg["variants"]:
        raise ValueError(f"Нет варианта {name}; используйте --list")
    v = cfg["variants"][name]
    kind = v.get("kind", "vlm")
    local = kind in ("det_dino", "boxes_dino", "photo_dino")
    if a.dataset and a.images:
        raise ValueError("Выберите --dataset или --images")
    if not a.dataset and not a.images:
        raise ValueError("Нужен --images или --dataset")
    if a.dataset and (a.dataset not in cfg.get("datasets", {}) or kind in ("boxes_dino", "photo_dino")):
        raise ValueError("--dataset требует существующий набор и маршрут с самостоятельной детекцией")
    if bool(a.boxes) != (kind == "boxes_dino"):
        raise ValueError("--boxes обязателен только для boxes_dino")
    if a.train != "never" and not local:
        raise ValueError("Этот VLM-маршрут не поддерживает обучение")
    if a.training != "auto" and kind != "photo_dino":
        raise ValueError("--training head/tail относится только к photo_dino/final")
    if a.qwen and kind != "photo_dino":
        raise ValueError("--qwen относится к photo_dino/final; VLM-маршруты используют свой API")
    if a.checkpoint and not local:
        raise ValueError("У этого маршрута нет локального классификатора")
    checkpoint = (a.checkpoint or (a.out / "photo_dino.joblib" if final else Path(v.get("checkpoint", ".")))).resolve()
    train = final or a.train == "always" or (a.train == "if-missing" and not checkpoint.exists())
    if final and a.dataset:
        raise ValueError("final требует --images с тестовыми фото")
    if local and not train and not checkpoint.is_file():
        raise ValueError(f"Нет модели {checkpoint}. Укажите --train if-missing и обучающие данные")
    if train:
        if not a.train_images or not a.train_images.is_dir():
            raise ValueError("Для обучения нужен --train-images с существующей папкой")
        if kind in ("det_dino", "boxes_dino"):
            if not a.annotations or not a.annotations.is_file():
                raise ValueError("Для обучения кропов нужен --annotations")
            if a.dataset:
                ds = cfg["datasets"][a.dataset]
                if (ROOT / ds["labels"]).resolve() == a.annotations.resolve() and (not a.split or a.split == ds.get("split")):
                    raise ValueError("Обучение и оценка используют ту же разметку без разделения split")
        elif a.annotations or a.split:
            raise ValueError("photo_dino/final ожидает папки классов, а не JSON; --annotations/--split не применяются")
    if a.epochs < 1 or not 1 <= a.k <= 12 or a.size < 14 or a.size % 14:
        raise ValueError("epochs >= 1, k от 1 до 12, size положительный и кратен 14")
    for path in (a.images, a.boxes, a.meta):
        if path and not path.exists():
            raise ValueError(f"Не найден путь {path}")
    if train and kind == "photo_dino" and checkpoint.name != "photo_dino.joblib":
        raise ValueError("Для обучения photo_dino имя --checkpoint должно быть photo_dino.joblib")
    v["checkpoint"] = str(checkpoint)
    return cfg, name, kind, checkpoint, train


def invoke(module, args):
    import importlib
    previous = sys.argv
    try:
        sys.argv = [module + ".py", *map(str, args)]
        importlib.import_module(module).main()
    finally:
        sys.argv = previous


def main():
    p = parser()
    a = p.parse_args()
    cfg = tomllib.loads(a.config.read_text())
    if a.list:
        for name, v in cfg["variants"].items():
            print(f"{name:20} {v.get('kind', 'vlm')}")
        print("final                обучение и тест целых фото")
        return
    try:
        cfg, name, kind, checkpoint, train = prepare(a, cfg)
    except ValueError as e:
        p.error(str(e))
    print(f"Маршрут: {a.variant}; обучение: {train}; режим: {a.training}", flush=True)
    if kind in ("det_dino", "boxes_dino", "photo_dino"):
        print(f"Чекпоинт: {checkpoint}", flush=True)
    if a.dry_run:
        print("Проверка параметров завершена; модели и API не запускались.")
        return
    import run
    run.load_config = lambda: cfg
    if a.device != "auto":
        cfg["detector"]["device"] = a.device
        cfg["variants"][name]["device"] = a.device
    if train and kind in ("det_dino", "boxes_dino"):
        from crop_dino import train as train_crops
        # boxes_dino shares the crop encoder settings with det_dino.
        settings = {**cfg["variants"]["det_dino"], **cfg["variants"][name]}
        train_crops(settings, a.annotations, a.train_images, a.split, checkpoint)
    elif train:
        args = [a.train_images, a.images, "--out", checkpoint.parent,
                "--training", a.training, "--device", a.device, "--size", a.size,
                "--epochs", a.epochs, "--k", a.k, "--test-format", a.test_format]
        if not a.qwen:
            args.append("--no-qwen")
        if checkpoint.name != "photo_dino.joblib":
            p.error("Для обучения photo_dino имя --checkpoint должно быть photo_dino.joblib")
        invoke("final", args)
        if a.variant == "final":
            return
        if a.qwen:
            cfg["variants"][name]["qwen"] = "qwen235"
    if a.dataset:
        invoke("run", ["--variant", name, "--data", a.dataset])
    else:
        args = ["--variant", name, "--images", a.images, "--out", a.out]
        for flag, value in (("--boxes", a.boxes), ("--meta", a.meta), ("--limit", a.limit)):
            if value is not None:
                args += [flag, value]
        invoke("predict", args)


if __name__ == "__main__":
    main()
