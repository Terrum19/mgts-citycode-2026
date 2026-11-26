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

### ТКШ и РЩ: Grounding DINO и Qwen по каждой рамке

```bash
python src/pipeline.py --variant det_grounding_qwen235 \
  --images PHOTO_DIR --meta photos.xlsx --out out/grounding-qwen
```

Требуются `requirements-detector.txt`, локальный снимок `IDEA-Research/grounding-dino-tiny`
в `weights/grounding-dino-tiny/` и `OPENROUTER_API_KEY`. Grounding DINO ищет оба типа
оборудования. После подавления пересекающихся дублей каждый оставшийся вырез с запасом
проверяется отдельным запросом Qwen. Тип определяется по вырезу. В `preds.json` запись
`crop_results` хранит вердикт или ошибку для каждой рамки, включая отклонённый фон.
Подтверждённые дефекты идут в `repair.csv`, сомнения и ошибки API — в `review.csv`.

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

Для подсчёта заявок и основных метрик поиска рамки сопоставляются один к одному внутри одного типа.
Пара засчитывается при IoU ≥ 0,10 и расстоянии между центрами не больше 25% диагонали эталонной рамки.
Отчёт указывает правило в поле `matching` и отдельно показывает метрики поиска при IoU 0,50 и 0,30.
Чтобы пересчитать сохранённые предсказания по прежнему протоколу, добавьте `--iou 0.5` к `src/evaluate.py`.

Всё настраивается в [config.toml](config.toml): `[taxonomy]` — типы объектов и виды повреждений (на финале
меняем только его и `[names]`), `[variants.*]` — модели, `[datasets.*]` — наборы для оценки, `[detector]` — вариант 1.

| Файл | Что делает |
|---|---|
| `src/vlm.py` | клиент OpenAI-совместимого API, промпты, разбор ответа |
| `src/detector.py` | Mask2Former (Mapillary Vistas): люки, опоры, шкафы → рамки |
| `src/variants.py` | `make(имя).predict(фото)` — один интерфейс для всех вариантов |
| `src/evaluate.py` | метрики: поиск объектов по типам, полнота по `bad`, вид повреждения, $/сек |
| `src/run.py` | прогон вариантов на наборах, кэш ответов, картинки худших ошибок в `runs/*/review/` |
| `src/predict.py`, `src/app.py` | система для финала: CLI и Streamlit |

## Открытые вопросы к организаторам (задать в чате кейса)

1. Будет ли интернет на площадке и можно ли пользоваться внешними API (OpenRouter и т.п.)?
2. Можно ли отправлять выданные фото во внешние сервисы?
3. В каком виде будут фото: крупный план одного объекта или кадр с регистратора?
