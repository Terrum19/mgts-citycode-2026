# Отчёт о реализации: что фактически сделано и что нет

**База:** `57f317a41669a2d394c77b624a01baf9fd18d0d0`, ветка `matthew-branch`.
**Дата:** 25.11.2026.
**Главный принцип:** отчёт фиксирует только реально выполненные команды. Синтетический smoke не
является метрикой качества; отсутствующие веса и данные не подменяются сетью или «примерными»
числами.

---

## 1. Итог

Реализовано ядро, которое можно проверить без целевых данных:

- единый контракт `frame/objects/mixed` и сквозной `review`;
- объектные и кадровые метрики, triage, F2, review rate, cost, baselines;
- общая `initial/finalize` policy для `CropDino`, `PhotoDino`, validation и test `final.py`;
- image-level отбор связки по operational cost, а не по `macro_f1`;
- `triage.csv`/`frames.csv` и UI-очередь;
- версионированные checkpoint/taxonomy, data hashes;
- VLM cache по содержимому/промпту/variant;
- `boxes_vlm`, offline/local варианты, preflight, Quality Gate (выключен);
- аудит данных и исправленный протокол оценки;
- 210 stdlib-тестов (было 110).

Не реализовано и не заявляется выполненным: обучение на целевых данных, реальный Mask2Former/VLM
inference, crop-tiling, OOD, temperature calibration, сравнение Grounding DINO, проверка на M4/Ultra 9,
чистый holdout.

---

## 2. Фактические проверки

### 2.1. Unit-тесты

До изменений:

```text
Ran 110 tests in 0.091s
OK
```

После изменений:

```text
python3 -m unittest discover -s tests
Ran 210 tests
OK
```

Тесты остаются stdlib-only (`python3` 3.14 без torch/sklearn); тяжёлые пути проверяются отдельно.
Добавлены файлы:

- `tests/test_results.py`
- `tests/test_evaluate_frames.py`
- `tests/test_triage.py`
- `tests/test_policy.py`
- `tests/test_selection.py`
- `tests/test_taxonomy.py`
- `tests/test_vlm_cache.py`
- `tests/test_config_utils.py`
- `tests/test_export_rows.py`
- `tests/test_quality.py`
- `tests/test_audit_data.py`

Обновлены `tests/test_config.py`, `tests/test_vlm.py`, `tests/test_ok_words.py`.

### 2.2. Компиляция

```text
python3 -m py_compile src/*.py src/poc/*.py src/collect/*.py src/label/*.py
OK
.venv/bin/python -m py_compile src/*.py
OK
```

### 2.3. Аудит текущей разметки

```text
python3 src/audit_data.py --labels data/labels.json --images data/raw/panoramas
exit 0
status=ok, errors=0, warnings=40
frames=211 (dev 47, test 164), objects=354
statuses: ok 338, bad 7, unclear 9
bad_frame=2, target_not_visible=62
labels SHA-256: e4aa7b0d980966ff38d16bbc27a1b7c74f9f648125cdc58fa1a839906e46f83a
```

Предупреждения, которые нельзя скрывать:

- у 37+ кадров `width/height` в labels `1280x600`, у фото `1280x657` — рамки/размер стоит перепроверить;
- один `bad_frame` содержит объект (статус кадра всё равно не оценивается);
- `damage=other_damage` встречается в разметке; добавлен в таксономию как catch-all;
- `runs/pano_test__gemma26` и `runs/pano_test__qwen235` зафиксированы как `contaminated`.

### 2.4. Синтетический end-to-end `final.py`

Созданы 60 синтетических train-фото (30 ok / 30 broken) и 3 test-фото в `runs/smoke_data`. Запуск:

```text
.venv/bin/python src/final.py runs/smoke_data/train runs/smoke_data/test \
  --no-qwen --epochs 1 --size 224 --val 0.3 --out runs/final_smoke
exit 0
```

Результат: `predictions.xlsx`, `predictions.csv`, `report.json`, `photo_dino.joblib`;
выбрана `логрег`; запись `photo_dino_variant.toml` создаётся для синхронизации конфига с checkpoint.
Это проверка проводки, а не качества: DINOv2 работала на синтетике, классов два, один epoch.

Дополнительно загружен этот checkpoint через `PhotoDino` и выполнен один inference: возвращён
`scope="frame"`, корректный `frame.status`, `objects=[]`, без `bbox=None`.

`predict.run` + `export_rows` на двух синтетических фото дали `frames=2` и пустые
`objects/repair/review` — frame-only путь не теряет кадровый результат.

### 2.5. Preflight

```text
.venv/bin/python src/preflight.py --variant det_dino_offline --offline
exit 1
```

Причины отказа — ожидаемые:

- нет `weights/crop_dino.joblib`;
- `[detector].model` остаётся HF repository id, а offline требует локальную папку;
- локальные веса detector не скачаны.

`python3 src/preflight.py ...` без venv дополнительно честно сообщает, что torch/transformers
не установлены. Это не «offline проверен», а корректный отказ.

### 2.6. Найденный и исправленный баг

`classify.is_ok("cover_broken")` возвращал `True`: подстрока `"ok"` входит в `"broken"`.
Это искажало `ok_of` в `final.py` (все классы `cover_broken` считались исправными) и могло
выбрать неверную связку. Исправлено сравнением коротких `ok/good/normal` целым токеном;
добавлен регрессионный тест `cover_broken`/`broken`.

---

## 3. Что изменено по файлам

| Файл | Изменение |
|---|---|
| `src/results.py` | новый контракт frame/object/mixed, review reasons, bbox sanitation, triage route |
| `src/evaluate.py` | frame/object/mixed, target_not_visible, triage, F2, review rate, cost, baselines |
| `src/policy.py` | общая initial/finalize policy, legacy/cascade, effective_policy |
| `src/selection.py` | image-level operational cost, ограничения, tie-break, baselines |
| `src/cascade.py` | `ANSWER_STATUSES`, review в clean_state/answer_error/merge_vlm |
| `src/vlm.py` | prompt_mode во всех builders, review normalize, fingerprint, context/crop max_side |
| `src/vlm_classify.py` | review/class=null, prompt mode |
| `src/vlm_cache.py` | кэш по JPEG/prompt/variant, atomic, ошибки не кэшируются |
| `src/photo_dino.py` | frame-only, общая policy, checkpoint/qwen validation, offline |
| `src/crop_dino.py` | policy, checkpoint metadata, support augmented, atomic save |
| `src/final.py` | image-level selection, Qwen variant, review в Excel, report/checkpoint v2 |
| `src/predict.py` | objects/frames/repair/review/triage CSV, boxes kinds, quality gate |
| `src/app.py` | очередь triage, frames, причины review, offline scatter |
| `src/export_rows.py` | общие CSV-строки для CLI/UI |
| `src/taxonomy.py` | canonical fingerprint, checkpoint validation, hashes, versions |
| `src/config_utils.py` | extends, offline restrictions |
| `src/preflight.py` | проверки импортов/файлов/весов/URL, честный `--load` |
| `src/audit_data.py` | аудит разметки, split/leakage, contaminated runs |
| `src/quality.py` | Quality Gate по яркости/контрасту/резкости/пересвету, выключен |
| `src/classify.py` | исправлен `is_ok` (ok внутри broken) |
| `src/embed.py`, `src/detector.py` | offline `local_files_only`, timm без сетевого pretrained |
| `src/atomic.py` | атомарная запись JSON/checkpoint |
| `config.toml` | `[selection]`, `[cost]`, `[app]`, `[quality_gate]`, offline/local/boxes variants |
| `configs/evaluation_protocol.json` | baseline, contaminated `pano_test`, holdout protocol |
| `docs/implementation_plan.md` | исправленный план и аудит слабых мест |
| `README.md` | режимы, triage, offline, проверки |

---

## 4. Отклонения от исходного плана (осознанные)

1. **`DetVLM.not_target` по умолчанию `drop`**, а не `review`: это поведение детектора; для рамок
   организатора `BoxesVLM` и `boxes_dino` — `review`. В `[variants.det_qwen235]` явно задан `drop`
   (находки детектора можно отбросить, рамки организатора — нельзя).
2. **`not_target="review"` причина — `not_target`**, а не `taxonomy_mismatch`; `taxonomy_mismatch`
   остаётся для неизвестного типа/таксономии. Обе причины нормализованы.
3. **`PhotoDino` не выбирает VLM-only candidate**: он не сериализуется в checkpoint; candidate
   показывается в таблице с причиной `no_serializable_production_path`.
4. **`final.py` использует image-level cost**, а не object-level triage: формат «папка = класс»
   не содержит рамок. Object-level triage реализован в `evaluate.py`.
5. **`[taxonomy].other_damage` добавлен**: иначе размеченный `other_damage` не проходит аудит и
   не может быть предсказан.
6. **`pano_test` не считается чистым**; новый holdout требует новых изображений.

---

## 5. Блокировано и требует данных/весов

- `data/roboflow/jie-yu_manhole-cover-4` и `data/smoke/` отсутствуют: восстановление crop-набора,
  обучение `crop_dino`, сравнение с VLM и реальные метрики не выполнены.
- Нет detector weights (`weights/mask2former...`, `weights/grounding-dino-tiny`): реальный
  Mask2Former/Grounding DINO inference не выполнялся.
- Нет `weights/crop_dino.joblib` и боевого `out/final/photo_dino.joblib`: реальный `det_dino`/
  `photo_dino` на целевых данных не проверен.
- Нет calibration/dev с достаточной поддержкой `bad`: temperature/C/threshold/OOD/Quality Gate
  не калибровались; Quality Gate выключен.
- Нет второй машины (M4 / Ultra 9): кросс-платформенная проверка не выполнялась.
- `pano_test` уже использован; чистый `holdout_v1` не создан.
- Grounding DINO comparison, crop-tiling, TTA selection и OOD не запускались; код/критерии
  зафиксированы, но результатов нет.

---

## 6. Как воспроизвести текущее состояние

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile src/*.py src/poc/*.py src/collect/*.py src/label/*.py
python3 src/audit_data.py --labels data/labels.json --images data/raw/panoramas
python3 src/preflight.py --variant det_dino_offline --offline   # ожидаемо failed
```

Тяжёлый synthetic smoke (не качество, только проводка):

```bash
.venv/bin/python src/final.py runs/smoke_data/train runs/smoke_data/test \
  --no-qwen --epochs 1 --size 224 --val 0.3 --out runs/final_smoke
```

Перед финалом:

1. Восстановить данные/веса и выполнить `preflight --load`.
2. Обучить/проверить checkpoint, синхронизировать `photo_dino_variant.toml` с `config.toml`.
3. Собрать новый holdout и заморозить SHA-256 до первого прогона.
4. Выполнить один финальный test с run manifest; после просмотра test новые изменения требуют
   нового независимого holdout.
