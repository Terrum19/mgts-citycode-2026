# Справка по моделям

## Детекторы

### Mask2Former на Mapillary Vistas (основной для варианта 1)
- HF: `facebook/mask2former-swin-large-mapillary-vistas-panoptic` (нужна panoptic: каждый объект отдельно).
  Semantic-версия склеивает соседние объекты в одно пятно.
- Обучена на 25 тыс. уличных кадров со всего мира, 65 классов. Нужные нам (проверено в конфиге):
  `Manhole`, `Utility Pole`, `Pole`, `Street Light`, `Junction Box`, `Catch Basin`, `CCTV Camera`.
- `Catch Basin` (ливнёвка) — отдельный класс, т.е. люк с ливнёвкой модель не путает.
- Состояние не оценивает, только находит. Тяжёлая (Swin-L), на CPU — секунды на кадр.

```python
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
name = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"
proc = AutoImageProcessor.from_pretrained(name)
model = Mask2FormerForUniversalSegmentation.from_pretrained(name)
out = model(**proc(images=img, return_tensors="pt"))
res = proc.post_process_panoptic_segmentation(out, target_sizes=[img.size[::-1]])[0]
```

### Подключаемые бэкенды (`[detector].backend`)

Код детектора — один интерфейс `detect(img)`, выбор модели в config.toml. По умолчанию стоит
проверенный `mask2former`; смена бэкенда меняет `fingerprint()` и инвалидирует кэш `runs/*/preds.json`.

| backend | Модель | Плюсы | Минусы |
|---|---|---|---|
| `mask2former` | Swin-L на Mapillary Vistas (panoptic) | не путает люк и ливнёвку; проверен на проекте | секунды на CPU, нет `aerial_equipment`, веса `CC BY-NC-SA 4.0` |
| `grounding_dino` | `IDEA-Research/grounding-dino-*`, Apache-2.0 | open-vocabulary: можно искать `aerial_equipment`; лёгкий | хуже различает люк и ливнёвку: нецелевое отсекается на 2-м уровне каскада (`is_object = false`) |
| `rtdetr` | свои веса (Apache-2.0/`RT-DETR`) | скорость, лицензионная чистота | нужно обучение на псевдоразметке; весов в репозитории нет |

Для `grounding_dino` в `[detector]` задаются `prompts = { наш_тип = "текст запроса." }`, отдельный
запрос на каждый тип (однозначный маппинг prompt → type) и `text_threshold`. Для `rtdetr` `classes`
маппит `model.config.id2label` в наши типы. Оба новых бэкенда в этом репозитории **локально не
запускались**: перед финалом переключать только после прогона на dev.

### Grounding DINO (запасной)
- HF: `IDEA-Research/grounding-dino-base` / `-tiny`, Apache-2.0. Детекция по тексту, только рамки.
- Лёгкая, работает на CPU/`mps`. Хуже различает люк и ливнёвку.
- Реализована как `backend = "grounding_dino"` (см. таблицу выше): отдельный текстовый запрос на
  каждый наш тип, дедупликация пересекающихся рамок по IoU 0.5.
- Закрытые 1.6 Pro / DINO-X точнее, но только через платный API IDEA Research.

```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
name = "IDEA-Research/grounding-dino-base"
proc = AutoProcessor.from_pretrained(name)
model = AutoModelForZeroShotObjectDetection.from_pretrained(name)
inputs = proc(images=img, text="manhole cover. utility pole. street cabinet.", return_tensors="pt")
res = proc.post_process_grounded_object_detection(
    model(**inputs), inputs.input_ids, threshold=0.3, text_threshold=0.25,
    target_sizes=[img.size[::-1]])
```

### SAM 3 (не в основном плане)
- HF: `facebook/sam3` (Meta, ноябрь 2025), **доступ по ручной заявке**. Маски всех объектов по
  короткой фразе или по образцу; presence head снижает ложные срабатывания; есть трекинг на видео.
- ~850M параметров, без NVIDIA медленно.

## VLM

### Qwen3-VL (основной кандидат)
- Сам выдаёт рамки в JSON по запросу → может заменить детектор.
- Локально: Qwen3-VL-8B 4-бит на M4 через MLX (`mlx-vlm`); на Ultra 9 — llama.cpp GGUF (CPU, медленно).
- Облако (OpenRouter, цены на 16.11, $ за 1M токенов вход/выход):
  - `qwen/qwen3-vl-235b-a22b-instruct` — 0.21 / 1.90
  - `qwen/qwen3-vl-30b-a3b-instruct` — 0.15 / 0.60
  - `qwen/qwen3-vl-8b-instruct` — 0.25 / 0.75
- Датасет финала (~200 фото × ~1.5k токенов) через 235B ≈ несколько центов.

### Gemini (для сравнения, OpenRouter)
- Сильный grounding (рамки). `google/gemini-3.5-flash` — 1.50 / 9.00, `google/gemini-3.1-pro-preview` — 2 / 12.

### Бесплатные VLM на OpenRouter (на 16.11)
- `google/gemma-4-31b-it:free` — лучший бесплатный кандидат
- `google/gemma-4-26b-a4b-it:free`
- `nvidia/nemotron-3.5-super-vl:free`
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`
- прочие (`thinkingmachines/inkling-2-small:free`, `inclusionai/ling-3.2-flash:free`, …) — не проверены на картинках

Подвохи бесплатных: лимит ~50 запросов/день без покупки кредитов (≈1000/день после покупки
≥ $10 — проверить через `GET /api/v1/key`), 429 под нагрузкой, модели пропадают из списка,
запросы могут логироваться провайдером. Российские карты не принимаются — пополнение криптой.

## Обучающие данные и Россия
- Vistas: 6 континентов, разбивки по странам нет — считаем, что российских кадров почти нет.
- SA-1B (база SAM): по памяти, Россия среди лидеров по числу фото, но это общие фото, не улица с машины.
- Внешний вид люка/опоры почти универсален — это не главный риск.
- **Главный риск — ноябрьская Москва**: снег, слякоть, грязь, блики. Люк под снегом не найдёт никто.
- Российская специфика (опоры двойного назначения с пучками кабелей, бетонные опоры, шкафы МГТС)
  детектором находится как «опора/шкаф», уточнение — за VLM.
