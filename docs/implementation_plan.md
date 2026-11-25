# План реализации v2: аудит исходного плана и исправленная версия

**Базовая точка:** `matthew-branch`, commit `57f317a41669a2d394c77b624a01baf9fd18d0d0`.
**Дата аудита:** 25.11.2026.
**Что это:** исправленная версия «Подробного плана реализации» после сверки с фактическим кодом,
данными и окружением. Исходный план сохранён по смыслу, но его ошибочные и невыполнимые места
исправлены явно.

Статусы:

| Статус | Значение |
|---|---|
| **DONE** | реализовано и проверено stdlib-тестами или end-to-end smoke на синтетике |
| **CODE-READY** | код есть, но реальный прогон требует данных/весов/API и не выполнялся |
| **BLOCKED** | нет данных, весов, второй машины или чистого test; имитировать нельзя |
| **DOC** | зафиксировано как протокол/чеклист, кода не требует |

---

## 1. Что было не так в исходном плане

### 1.1. Baseline и релиз

| Проблема | Исправление |
|---|---|
| План считает базой `00b70a7..a9f3d98`, но HEAD уже `57f317a`; push иначе потеряет последний коммит. | Базой считается фактический HEAD `57f317a`; новые изменения — поверх него. |
| «Не добавлять в push архив» бессмысленно: архив был untracked и в push не попадал. | В `.gitignore` добавлено `/*.tar.gz`; коммиты — только явными путями, без `git add -A`. |
| Проверка «на M4 и Ultra 9» записана как критерий готовности коммита. | Это внешний чеклист `NOT RUN`; локальная готовность подтверждается тестами и отчётом, а не имитацией второй машины. |
| 110 тестов на system Python не доказывают работу torch-путей. | Разделены два уровня: stdlib-тесты (`python3`) и тяжёлый smoke (`.venv/bin/python`), в отчёте указано, что фактически запускалось. |

### 1.2. Загрязнение test

`runs/pano_test__qwen235` и `runs/pano_test__gemma26` уже существуют: старый `pano_test` использован
для прогонов и не может быть «чистым» финальным test. План требовал «test не использовался ни в
одном sweep/report/cache», что уже невыполнимо.

**Исправление:** `configs/evaluation_protocol.json` фиксирует `pano_test` как `contaminated` с
`selection_eligible=false`. `pano_dev` — development, а не holdout. Честный `holdout_v1` может быть
только из **новых** изображений; перетасовка старых 211 кадров не устраняет уже состоявшийся просмотр.
До появления новых данных статус — `holdout_unavailable`, а не «test пройден».

### 1.3. Frame/object-контракт

Исходный план говорил «frame вместо bbox=None», но не определял:

- что делать с `target_not_visible` (62 кадра в текущей разметке);
- как считать image-level метрики, если объектный вариант не отдаёт `frame`;
- что означает `status="bad"` вместе с `review=true`;
- где хранятся `review_reasons` и некорректные рамки;
- как валидировать GT-рамки, а не только предсказания.

**Исправление:** введён `src/results.py` и схема `schema_version=2`. `review` отделён от `status`;
маршрут один: `review=true` или `status=review` → review, иначе `status=bad` → repair, иначе none.
`target_not_visible` — валидный отрицательный кадр (FP/review учитываются), `bad_frame` исключается
полностью. Некорректная рамка не роняет IoU, а попадает в `invalid_bbox_objects` и требует review.
Для объектов без `frame` кадровый агрегат выводится из объектов и помечается `source="objects"`.

### 1.4. Triage-метрики и cost

Исходная формула `triage_fp = triage_pred - triage_tp` смешивала FP-детекции и review-FP;
`C = C_FN*N_FN + C_FP*N_FP + C_review*N_review` при `N_FP=triage_fp` и `N_review=review_objects`
дважды списывал review-FP. Не было единого сопоставления, а `all-review`/`all-bad` не проверялись.

**Исправление:** один greedy matching по IoU и типу; счётчики `bad_to_bad`, `bad_to_review`,
`bad_to_ok`, `bad_to_missed`, `ok_to_*`, `spurious_*`, `unclear_to_*`; `triage_fp_auto` и
`triage_fp_review` разделены; cost списывает review-FP только как review, а не как FP.
Baselines `all_review`, `all_bad`, `all_ok` считаются на фактическом распределении validation.
Для image-level `final.py` cost считается по кадрам, а не по объектам: там нет bbox.

### 1.5. `final.py`: категориальная ошибка отбора

Валидация `final.py` работает с классом-папкой на фотографию, а `score()` требовал object-level
`repair_tp/fp/fn`. Это разные единицы: объектных рамок в формате «папка = класс» нет.

**Исправление:** `src/selection.py` считает image-level `operational_cost` (`FN`, `FP`, `review`
frames). Object-level triage остаётся в `evaluate.py` для общих кадров. `score()` больше не влияет
на выбор; `macro_f1` остаётся справочным. Test-инференс использует тот же `initial/finalize`, что
validation и `PhotoDino`, а `predictions.xlsx` получает `Статус`, `Требует проверки`, `Причины review`.
VLM-only candidate показывается, но не выбирается (`serializable=false`), пока нет production-пути.

**Дополнительно найдено:** validation вызывала VLM без hint, а `PhotoDino._cascade` — с top-3 hint.
Это разные промпты и разные решения. Hint убран из production/validation; `hint_mode="none"`.

### 1.6. Общая decision policy

Сигнатура в исходном плане не включала `support`, `min_support`, `type_hint`, `class_policy`,
не различала legacy-additive и cascade-resolver, не задавала приоритет checkpoint policy vs config.

**Исправление:** `src/policy.py` с `initial()`, `finalize()`, `effective_policy()`,
`decision_reasons()`. `cascade.STATUSES` остаётся `("ok","bad")` для готовности классификатора;
ответы VLM проверяются отдельным `ANSWER_STATUSES`. `effective_policy` использует сохранённую
policy checkpoint; конфликтующий semantic-ключ в config — ошибка, а не молчаливая подмена.
Deployment-ключи (device, endpoint, key, checkpoint) в policy не входят.

### 1.7. VLM: review, `is_object`, пустой ответ

`clean_state` превращал неизвестный статус в `ok`; `answer_error` считал `review` ошибкой;
`DetVLM` всегда выбрасывал `is_object=false`; `VLMOnly` при ошибке не имел кадрового результата.
Успешный пустой ответ и ошибка API смешивались.

**Исправление:** `clean_state` неизвестное → `review`; explicit `review` имеет `damage=None`;
`answer_error` принимает `ok/bad/review`; `DetVLM.not_target` явный (`drop` по умолчанию для
детектора, `review` для рамок организатора); `BoxesVLM` сохраняет bbox/type/object_id и не удаляет
рамки; успешный пустой ответ → `frame.ok`; ошибка/битый JSON → `frame.review`. Prompt mode
проброшен во все builders, fingerprint включает resolution, context/crop prompts, `not_target`.

### 1.8. Кэш VLM

Ключ по пути не различал замену файла и смену промпта; ошибки кэшировались как успех; запись не была
атомарной.

**Исправление:** `src/vlm_cache.py`: SHA-256 канонического payload, в котором variant, VLM
fingerprint, SHA-256 **фактических JPEG-байтов** после resize, SHA-256 фактического промпта,
labels/hint/examples, schema version. Ошибки, битый JSON и невалидный class не кэшируются;
explicit review кэшируется. Запись — `atomic.write_json_atomic` (temp + fsync + replace).

### 1.9. Checkpoint и таксономия

Не было версии, fingerprint, data hashes, support augmented; старый classifier мог молча работать
с новой таксономией.

**Исправление:** `src/taxonomy.py`: канонический JSON только `object_types` + `damage` (не `names`),
SHA-256; `format_version=2`; `validate_checkpoint` с режимами `warn/allow/error` (legacy по умолчанию
warn, final/preflight — error). `crop_dino` и `photo_dino` сохраняют версию, taxonomy, classes,
policy, support, data hashes, package versions, training args. Атомарная запись.

### 1.10. Offline, зависимости, UI

- `Encoder` молча скачивал `facebook/dinov2-base`; detector делал сетевой `from_pretrained`;
  `timm` использовал `pretrained=True`.
- В `requirements.txt` не было прямых `joblib/numpy/scikit-learn`; в detector — `timm`.
- `st.map` мог требовать внешнюю подложку.
- Вариантов `*_offline/*_local` и `boxes_vlm` не было.

**Исправление:** offline-флаг проброшен в `Encoder`, detector backends, `Embedder`/timm; добавлены
`extends`-варианты, `BoxesVLM`, `preflight.py`, `[app].offline_map`, `st.scatter_chart` с подписью
«координатная схема». Offline-вариант с удалённой VLM/HF id падает явно, а не идёт в сеть.

### 1.11. Данные и P1-эксперименты

Нет `data/roboflow/`, `data/smoke/`, detector-весов, обученного crop/photo checkpoint, второй машины.
Разделы 6.1, 6.5, 10, 12–16, 18 нельзя выполнить как реальные эксперименты. Нельзя выдавать
`preflight` без `--load` за проверенный offline и синтетический checkpoint за боевой.

**Исправление:** реализованы проверяемые части — `audit_data.py`, `quality.py` (по умолчанию
выключен), `preflight.py`, manifests/протокол; OOD, temperature fitting, tiling, TTA selection,
Grounding DINO comparison оставлены **BLOCKED** с критериями запуска. `bench_speed.py` не выдаёт
оценки вместо измерений.

### 1.12. Найденный баг вне плана

`classify.is_ok("cover_broken")` возвращал `True`: подстрока `"ok"` находится внутри `"broken"`.
Это ломало `ok_of` в `final.py` и весь отбор. Исправлено: короткие `ok/good/normal` сравниваются
только целым токеном; добавлен регрессионный тест.

---

## 2. Исправленные инварианты

1. **Repair ≠ Triage.** Repair — только автоматически подтверждённые `bad`. Triage — `bad + review`,
   обязательная очередь реагирования.
2. **Неизвестное не становится `ok`.** Ошибка VLM, неизвестный статус, OOD, quality, invalid bbox
   дают `review`, если нет более сильного подтверждённого `bad`.
3. **Уверенный `bad` не понижается.** Ошибка необязательного описания, quality gate и OOD не
   заменяют `bad` на `review`/`ok`/удаление.
4. **Объект и кадр разделены.** Объект имеет корректный `bbox`; кадровый результат — `frame`;
   `bbox=None` допустим только как legacy-вход и читается как frame, а не как объект.
5. **Один decision policy.** Validation, test и production используют `policy.initial/finalize`;
   выбор связки не может быть argmax при production-каскаде.
6. **Никакой имитации.** Отсутствующие веса/данные дают failure с именем файла, а не сетевой
   fallback, подмену модели или «примерные» метрики.

---

## 3. Контракты

### 3.1. Результат (`src/results.py`)

```json
{
  "schema_version": 2,
  "scope": "frame | objects | mixed",
  "frame": {"status": "ok|bad|review", "type": null, "damage": null, "description": "",
             "review": false, "review_reasons": [], "source": "native|objects|quality_gate|legacy|missing"},
  "objects": [{"type": "...", "bbox": [1,2,3,4], "status": "ok|bad|review", "damage": null,
                "description": "", "review_reasons": [], "object_id": "..."}],
  "invalid_bbox_objects": [{"index": 0, "bbox": null, "reason": "missing"}],
  "seconds": 0.0, "cost": 0.0, "bad_json": false, "error": null
}
```

Маршрут кадра: `review=true`/`review` → **review**; иначе `bad` → **repair**; иначе **none**.
Один кадр с `bad` и спорным объектом имеет `status="bad"`, `review=true` и маршрут review; он не
считается одновременно FP и review.

### 3.2. Метрики (`src/evaluate.py`)

- Object-level: detection, repair, triage, review rate по объектам, cost по triage-счётчикам.
- Frame-level: `frame_bad_to_*`, `frame_ok_to_*`, `review_frames`, `review_rate`,
  `operational_cost`.
- `bad_frame` исключается; `target_not_visible` — валидный отрицательный кадр; `unclear` вне
  бинарных статусных метрик, но остаётся в review-rate.
- Новые rate при нулевом знаменателе — `None`; старые поля сохраняют прежний смысл.
- Baselines: `all_review`, `all_bad`, `all_ok`.

### 3.3. Policy (`src/policy.py`)

- `initial(labels, classes, scores, mode, cfg, support, type_hint, class_policy)`.
- `finalize(decision, cfg, tax, vlm)`; `None` — только явный `not_target=drop`.
- `effective_policy(saved, runtime_cfg)`; checkpoint policy приоритетна, конфликт — `ValueError`.
- Legacy: argmax/`min_confidence`/additive VLM. Cascade: `summarize/decide/pick` + VLM resolver.

### 3.4. Отбор (`src/selection.py`)

Image-level candidates: legacy `w ∈ {0,0.3,0.5,1.0}`, cascade, VLM-only (не выбирается).
Ограничения: `min_bad`, `min_ok`, `min_triage_recall`, `max_review_rate`. Выбор — минимальный
`operational_cost`; tie-break: `-triage_recall`, `review_rate`, `-auto_bad_recall`, `uses_vlm`,
`base_complexity`, имя. Нет eligible → `selection_failed`, report есть, checkpoint/predictions нет,
exit code 2.

### 3.5. VLM (`src/vlm.py`, `src/vlm_classify.py`, `src/vlm_cache.py`)

- `prompt_mode` во всех builders; `recall` разрешает `review` и `class=null, review=true`.
- `normalize` сохраняет review, собирает invalid bbox.
- Fingerprint — SHA-256 смыслового запроса, включая размеры и оба context-промпта.
- Кэш — по фактическому JPEG/prompt/variant; ошибки не кэшируются; запись атомарная.

### 3.6. Checkpoint (`src/taxonomy.py`)

- `format_version=2`, `taxonomy`, `taxonomy_sha256`, `classes`, `support_original`,
  `support_augmented`, `policy`, `data_hashes`, `package_versions`, `training_args`, `created_at_utc`.
- `validate_checkpoint`: совпадение fingerprint обязательно; legacy без fingerprint — warn/allow/error;
  v2 без fingerprint — ошибка.

### 3.7. CSV/UI (`src/export_rows.py`, `src/predict.py`, `src/app.py`)

`objects.csv`, `frames.csv`, `repair.csv`, `review.csv`, `triage.csv`, `preds.json`.
Frame-строки имеют пустые `bbox/object_id/object_n`; метаданные организатора переносятся в оба scope
с префиксом `meta_` при коллизии. UI показывает очередь triage, подтверждённый repair, review и
причины; offline — `st.scatter_chart`, а не `st.map`.

---

## 4. Что реализовано и что заблокировано

| Раздел исходного плана | Статус | Комментарий |
|---|---|---|
| 0. Инварианты | DONE | `results.py`, `cascade.py`, тесты |
| 1. Baseline | DONE/DOC | HEAD зафиксирован, `.gitignore`, протокол; вторая машина — NOT RUN |
| 2. Frame/object-контракт | DONE | `results.py`, все варианты |
| 3. Repair/triage/F2/cost | DONE | `evaluate.py`, `tests/test_triage.py` |
| 4. Отбор и policy | DONE (код) | `policy.py`, `selection.py`, synthetic end-to-end; реальные данные — BLOCKED |
| 5. Triage CSV/UI | DONE | `export_rows.py`, `predict.py`, `app.py` |
| 6. Восстановление crop-данных | BLOCKED | нет `data/roboflow`, `data/smoke`; аудит и metadata готовы |
| 7. Зависимости/локальные веса | CODE-READY/BLOCKED | offline propagation, preflight; весов детектора нет |
| 8. VLM cache/variants/offline | DONE (код) | cache, variants, boxes_vlm; API-прогон не выполнялся |
| 9. Review/prompt modes | DONE | prompts, normalize, merge, fingerprint |
| 10. Mask2Former baseline | BLOCKED | нет benchmark-манифеста и detector-весов |
| 11. Расширение разметки | BLOCKED | нет новых данных |
| 12. Калибровка C/T/порогов | BLOCKED | нет calibration split; `C` вынесен в параметр |
| 13. OOD | BLOCKED | нет calibration; политика описана |
| 14. Quality Gate | CODE-READY | `quality.py`, по умолчанию выключен; калибровки нет |
| 15. Crop-tiling | BLOCKED | не реализовано в inference; критерии в отчёте |
| 16. Grounding DINO | BLOCKED | backend сохранён; сравнения нет |
| 17. Таблица режимов | DONE/DOC | README + отчёт |
| 18. Репетиция 20 минут | DOC | чеклист, фактические замеры не выдаются |
| 19. Заморозка/test | DOC | `pano_test` contaminated; нужен новый holdout |

---

## 5. Порядок работ (фактический)

1. `chore: freeze evaluation protocol and data audit`
2. `feat: add frame results and operational evaluation`
3. `refactor: unify decision and VLM review policy`
4. `feat: version checkpoints and selection in final pipeline`
5. `feat: add offline variants, boxes VLM and preflight`
6. `feat: align exports, quality gate and implementation report`

15 коммитов исходного плана были бы фиктивными при отсутствии данных; оставлены проверяемые этапы.

---

## 6. Критерии готовности

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile src/*.py src/poc/*.py src/collect/*.py src/label/*.py
python3 src/audit_data.py --labels data/labels.json --images data/raw/panoramas
python3 src/preflight.py --variant det_dino_offline --offline   # честный failure без весов
```

Тяжёлый smoke — только `.venv/bin/python`; он не заменяет проверку на целевых данных.
Ни один отчёт не должен утверждать «обучено», «offline проверено», «Grounding DINO протестирован»
или «test чист», если соответствующая команда не запускалась.

---

## 7. Что делать после появления данных

1. Восстановить `data/roboflow/...`, `data/smoke/...`, detector-веса; выполнить `audit_data.py`.
2. Обучить `crop_dino` на train, проверить taxonomy fingerprint и `preflight --load`.
3. Собрать новый `holdout_v1` из новых изображений; заморозить labels/images SHA-256 до первого
   прогона; `pano_test` оставить contaminated.
4. Запустить benchmark Mask2Former, затем calibration C/T/thresholds на calibration, подтвердить
   на dev; не открывать holdout до финальной заморозки.
5. OOD/quality/tiling/TTA включать только после сравнения cost/recall/review rate на dev.
6. Финальный test — один раз, с run manifest (dataset hash, code commit, model fingerprint).
