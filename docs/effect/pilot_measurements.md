# Измерения на учебных панорамах

Сводка локальной проверки от 26 ноября 2026 года. В таблицах сохранены агрегаты разметки и результаты каждого прогона. Исходные фотографии, разметку и журналы команда хранит отдельно от Git. Для сверки с оригиналами ниже приведены команды и контрольные суммы.

## Что есть в разметке

В двух столбцах с пометкой `bad` разные единицы учёта. Один кадр может содержать несколько объектов. Пустыми считаем кадры, для которых в экспорте нет размеченных объектов.

| Часть выборки | Кадры | Пустые кадры | Кадры хотя бы с одним `bad` | Объекты | `ok` | `bad` | `unclear` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Для разработки (`dev`) | 47 | 8 | 3 | 127 | 120 | 3 | 4 |
| Отложенная (`test`) | 164 | 58 | 4 | 227 | 218 | 4 | 5 |
| Всего | 211 | 66 | 7 | 354 | 338 | 7 | 9 |

Каждая строка разметки соответствует отдельному JPEG из 287 файлов в каталоге панорам. Части выборки согласуются с назначениями по именам файлов, общих идентификаторов мест между ними нет. Журнал съёмки содержит строки с координатами камеры для 76 изображений. Среди 211 размеченных кадров совпадений с этими строками по имени файла нет. Частоту дефектов в потоке МГТС по учебной разметке определить нельзя. [Назначения кадров по частям выборки](../../configs/split.json) · [Порядок работы с разметкой](../labeling.md)

## Что попало в список проверки

Прогоны выполнены на ноутбуке Intel Core i3-1115G4 с CPU-сборкой PyTorch. В каждом журнале одно изображение. Буквами обозначены два разных кадра. На кадре А отдельно проверили поиск объектов моделью и классификацию по рамкам из разметки.

| Каталог в `out/` | Кадр | Вход | Время модели, с | В том числе поиск, с | Строк в review.csv | Из них с координатами камеры | Строк в repair.csv |
|---|---|---|---:|---:|---:|---:|---:|
| train-demo | А | Панорама | 85,715 | 84,836 | 1 (опора) | 0 | 0 |
| train-gps | Б | Панорама | 72,344 | 70,897 | 2 (люки) | 2 | 0 |
| train-boxes | А | Три готовые рамки | 2,295 | 0 | 3 (опоры) | 0 | 0 |

`review` означает направление оператору. Координаты камеры в двух строках унаследованы от одной панорамы. По ним нельзя установить точные позиции обнаруженных люков. Журналы не содержат решений оператора по этим объектам.

Таймер `seconds` охватывает поиск рамок, если он включён, и классификацию внутри `predict`. Загрузка весов, открытие и преобразование исходного изображения, запись файлов и работа оператора лежат за границами замера. Версию Git и характеристики машины сами журналы предсказаний не записывают. Границы таймера сверены по коду, в том числе в ревизии `abf67a5`. Эти единичные запуски позволяют проверить формат выдачи. Для прогноза скорости потока нужен замер всей партии на целевом оборудовании. [Таймер модели](../../src/crop_dino.py) · [Чтение изображения и выгрузка CSV](../../src/predict.py)

## Как сверить с исходными данными

Запускать из корня репозитория с локальным комплектом команды. Скрипт читает только готовые журналы и выводит агрегаты.

```python
import csv
import json
from pathlib import Path

def read_text(path):
    return Path(path).read_text(encoding="utf-8")

labels = json.loads(read_text("data/labels.json"))
splits = json.loads(read_text("configs/split.json"))
files = {row["file"] for row in labels}
images = {path.name for path in Path("data/raw/panoramas").glob("*.jpg")}
assert len(files) == len(labels) and files <= images
assert all(splits[row["file"]] == row["split"] for row in labels)
locations = [{row["location_id"] for row in labels if row["split"] == split}
             for split in ("dev", "test")]
assert locations[0].isdisjoint(locations[1])
for split in ("dev", "test"):
    rows = [row for row in labels if row["split"] == split]
    statuses = [obj["status"] for row in rows for obj in row["objects"]]
    bad_frames = sum(any(obj["status"] == "bad" for obj in row["objects"]) for row in rows)
    print(split, len(rows), sum(not row["objects"] for row in rows), bad_frames,
          len(statuses), [statuses.count(status) for status in ("ok", "bad", "unclear")])

meta = [json.loads(line) for line in read_text("data/raw/panoramas/meta.jsonl").splitlines()]
geotagged = {row["file"] for row in meta if row.get("ok") and row.get("file") in images
             and all(isinstance(row.get(key), (int, float)) for key in ("pano_lat", "pano_lon"))}
print("JPEG, с координатами, совпадений с разметкой", len(images), len(geotagged), len(files & geotagged))

def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))

for run in ("train-demo", "train-gps", "train-boxes"):
    folder = Path("out") / run
    record, = json.loads(read_text(folder / "preds.json")).values()
    review = csv_rows(folder / "review.csv")
    with_gps = sum(bool(row.get("pano_lat") and row.get("pano_lon")) for row in review)
    print(run, round(record["seconds"], 3), round(record["det_seconds"], 3),
          [obj["type"] for obj in record["objects"]], len(review), with_gps,
          len(csv_rows(folder / "repair.csv")))
```

Контрольные суммы SHA-256. Блок можно передать команде `sha256sum -c` через стандартный ввод.

```text
e4aa7b0d980966ff38d16bbc27a1b7c74f9f648125cdc58fa1a839906e46f83a  data/labels.json
48ea8262f53304f547f0887f97b788fdc09a15357b05bd80619f49b341e5a7ba  configs/split.json
d663ef48f3a194147dcf9d1fce764eacc0824b6b94e90907e83b902bd28b38ec  data/raw/panoramas/meta.jsonl
8c05f4bb881d7df35b983e3480036dfa5cddcde927c7732476c771903eab2437  out/train-demo/preds.json
76cc2e4ea26154653fc4dfc13cb4b36874329019885dd017df800fb2161d7da7  out/train-gps/preds.json
709d5944ad4c2a97644dc7f070311dbee24539ccecd98ccfc7c0d0e59f4779b6  out/train-boxes/preds.json
039e5bb094180fd537e59ab01fc88a74c283e9668a44b940b12192a51a94f1b3  out/train-demo/review.csv
76db46574526ed777f768f529cc2c91d13137b83a882e920348f8c41929df8ab  out/train-gps/review.csv
b8b62cd88ef7eb5f058b3d6b33f62876feb0f0f7f1bfd7c21e7120aba2cd1dbf  out/train-boxes/review.csv
8d7802a819fffa835b8dfeed95997a7396aded4e0c09b6b5bbaadb3595f5d505  out/train-demo/repair.csv
8d7802a819fffa835b8dfeed95997a7396aded4e0c09b6b5bbaadb3595f5d505  out/train-gps/repair.csv
8d7802a819fffa835b8dfeed95997a7396aded4e0c09b6b5bbaadb3595f5d505  out/train-boxes/repair.csv
```
