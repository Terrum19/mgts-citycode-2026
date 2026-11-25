# МГТС «Код города» 2026 — Сетевая инфраструктура и сервис

Кейс: прототип компьютерного зрения, который по фото с камер служебных автомобилей
находит объекты городской инфраструктуры (кабельные колодцы, опоры двойного назначения),
определяет признаки неудовлетворительного состояния и формирует перечень объектов
на проверку/ремонт с привязкой к координатам.

## Сроки

| Дата | Что |
|---|---|
| 15.11 | Конец регистрации (команда 3–5 человек, все зарегистрированы, модерация пройдена) |
| **19.11** | **Загрузить квалификационное задание (эссе) на платформу** |
| **26.11** | **Очный финал в МТУСИ, 09:00–18:00.** Данные и полная постановка выдаются только в этот день; ~3 часа на решение, затем защита |

Призы: 200 000 ₽ победителю направления, fast track на стажировку за 2–3 место.

## Документы

- [docs/case.md](docs/case.md) — постановка кейса и всё, что сказали на вебинаре (включая ответы на вопросы)
- [docs/plan.md](docs/plan.md) — **финальный план**: валидационный стенд, три варианта, метрики, график
- [docs/speed.md](docs/speed.md) — скорость связки на M4/CPU и сколько времени займёт финал
- [docs/models.md](docs/models.md) — справка по моделям: детекторы, VLM, OpenRouter
- [docs/essay.md](docs/essay.md) — квалификационное эссе
- [docs/webinar_transcript.txt](docs/webinar_transcript.txt) — расшифровка вебинара 12.11
- [docs/webinar_slides/](docs/webinar_slides/) — ключевые слайды

## Как запустить

```bash
uv venv .venv && uv pip install -r requirements.txt          # + requirements-detector.txt для варианта 1
echo "OPENROUTER_API_KEY=..." >> .env
```

| Что | Команда |
|---|---|
| **Финал: папки train → выбор связки → предсказания test** | `python src/final.py data/final_train data/final_test --out out/final` |
| Система: фото → перечень на ремонт | `python src/predict.py --variant qwen235 --images photos/ [--meta photos.csv] --out out/run1` |
| Интерфейс для защиты | `streamlit run src/app.py` |
| Сравнение вариантов на разметке | `python src/run.py --variant qwen235 det_qwen235 --data pano_dev damage_manholes` |
| Выгрузить разметку из Label Studio | `python src/label/export.py` |

### Режимы финала

Для локальных вариантов установите также `requirements-detector.txt`. Заранее положите
`dinov2-base` в `weights/` (`python src/fetch_weights.py facebook/dinov2-base`) и скачайте Mask2Former
из `[detector].model` в локальный кэш Transformers. На месте проверьте с `HF_HUB_OFFLINE=1`.

| Формат задания | Запуск |
|---|---|
| Один класс на фото | `python src/final.py TRAIN TEST --no-qwen --out out/final`, затем `photo_dino` в интерфейсе для новых фото |
| Объекты с готовыми рамками | `python src/predict.py --variant boxes_dino --images PHOTO_DIR --boxes boxes.json --meta photos.xlsx --out out/boxes` |
| Рамки нужно найти локально | `python src/predict.py --variant det_dino --images PHOTO_DIR --meta photos.xlsx --out out/det` |
| Рамки нужно найти, API доступен | `python src/predict.py --variant qwen235 --images PHOTO_DIR --meta photos.xlsx --out out/qwen` |

`boxes.json` — массив записей `{"file":"foto.jpg","objects":[{"bbox":[10,20,130,180],"type":"manhole","object_id":"42"}]}`.
Координаты в пикселях, `[x1,y1,x2,y2]`. Пустой кадр задаётся через `"objects":[]`.
Тип и ID объекта можно опустить. Для папок тестовых фото, которые являются просто
каталогами, укажите `--test-format images`. Для размеченного теста используйте `--test-format labeled`.

### Обучение и оценка кропов

```bash
python src/poc/make_crop_train.py > crop_train.json
python src/crop_dino.py crop_train.json data
python src/run.py --variant det_dino --data pano_dev
python src/poc/compare_crops.py pano_dev runs/pano_dev__det_dino/preds.json
```

Для адаптации к рамкам организатора используйте `src/crop_dino.py LABELS PHOTO_DIR`.
В `LABELS` применяется тот же JSON, дополненный `status` (`ok` или `bad`)
и `damage` для каждого объекта. При нехватке примеров состояния или низкой уверенности
найденный объект попадает в `review.csv`. `repair.csv` содержит подтверждённые заявки.
В отчёте оценки `repair_recall` учитывает и пропущенные рамки. Сначала подбирайте параметры на dev.

### Каскад, погодные аугментации и бэкенды

`det_dino`, `boxes_dino` и `photo_dino` работают в режиме `cascade = true`: быстрый DINOv2 решает
по суммарной `P(bad)`, уверенности статуса `max(P(ok), P(bad))` и отрыву между статусами
`|P(bad) − P(ok)|`, а VLM вызывается только на спорные объекты/кадры (и для описания подтверждённых `bad`). Нет ключа или
сети — спорное уходит в `review.csv`, а подтверждённый `bad` не теряется из-за необязательного
описания. Пороги (`p_bad_min`, `p_bad_max_ok`, `status_conf_min`, `margin_min`) — стартовые:
подбирать на dev, не на test.

Погодные аугментации (свет, цвет, смаз, шум, JPEG; секция `[augment]` в config.toml) применяются
только к train после сплита: `python src/crop_dino.py crop_train.json data` и
`python src/final.py TRAIN TEST --augment 1` (дополнительные копии только к признакам логрега).
Валидация и тест всегда считаются на оригиналах.

Детектор подключается через `[detector].backend`: `mask2former` (по умолчанию, проверенный),
`grounding_dino` (open-vocabulary: видит `aerial_equipment`, но чаще путает люк и ливнёвку) или
`rtdetr` (свои веса). Новые бэкенды локально не проверялись — переключать только после прогона
на dev; молчаливой подмены модели нет.

Резерв при дефиците редких примеров — `src/poc/copy_paste.py`: вставка дефектов (по маске) на
реальные фоны `data/raw/panoramas` с перспективой, feathering и подгонкой яркости; на выходе
`images/` + `labels.json` в формате `crop_dino.py`. Синтетика в train автоматически не
подмешивается, качество проверять только на настоящих фото.

Всё настраивается в [config.toml](config.toml): `[taxonomy]` — типы объектов и виды повреждений (на финале
меняем только его и `[names]`), `[variants.*]` — модели, `[datasets.*]` — наборы для оценки, `[detector]` — вариант 1.

### Проверки без тяжёлых зависимостей

```bash
python3 -m unittest discover -s tests
```

Тесты работают на stdlib + PIL/numpy: логика каскада и порогов, погодные аугментации, разбор
конфига, геометрия рамок детектора, copy-paste. Пути, которым нужны torch/sklearn/requests
(обучение, чекпойнты, VLM, Streamlit), проверяются только `python3 -m py_compile`.

| Файл | Что делает |
|---|---|
| `src/vlm.py` | клиент OpenAI-совместимого API, промпты, разбор ответа |
| `src/detector.py` | детектор объектов: бэкенды Mask2Former / Grounding DINO / RT-DETR → рамки |
| `src/cascade.py` | каскад: суммарные P(ok)/P(bad) внутри типа, margin между статусами, `review`, слияние с VLM |
| `src/augment.py` | погодные аугментации (только train): свет, цвет, смаз, шум, JPEG |
| `src/variants.py` | `make(имя).predict(фото)` — один интерфейс для всех вариантов |
| `src/evaluate.py` | метрики: поиск объектов по типам, полнота по `bad`, вид повреждения, $/сек |
| `src/run.py` | прогон вариантов на наборах, кэш ответов, картинки худших ошибок в `runs/*/review/` |
| `src/predict.py`, `src/app.py` | система для финала: CLI и Streamlit |

## Открытые вопросы к организаторам (задать в чате кейса)

1. Будет ли интернет на площадке и можно ли пользоваться внешними API (OpenRouter и т.п.)?
2. Можно ли отправлять выданные фото во внешние сервисы?
3. В каком виде будут фото: крупный план одного объекта или кадр с регистратора?
