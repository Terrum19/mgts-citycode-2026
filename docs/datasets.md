# Открытые датасеты (Roboflow Universe)

Поиск сделан скриптом [src/collect/rf_search.py](../src/collect/rf_search.py) через API: 176 датасетов,
у 44 есть классы состояния. Полный список лежит в `data/rf_search.json` (локально, в git не попадает).
Числа ниже — объекты по классам, данные на 16.11. Все датасеты под лицензией CC BY 4.0.

## Колодцы

| Датасет | Фото | Классы (объектов) | Комментарий |
|---|---|---|---|
| **[jie-yu/manhole-cover-4](https://universe.roboflow.com/jie-yu/manhole-cover-4)** | 3830 | good 1071, broke 824, lose 890, uncovered 565, circle 356 | **Основной.** Классический китайский набор дефектов люков, 5 классов, хороший баланс |
| [manhole-mlr2j/manhole-detection-hyr17](https://universe.roboflow.com/manhole-mlr2j/manhole-detection-hyr17) | 2660 | good 1270, lose 604, uncovered 594, broke 569, circle 164 | Копия того же набора |
| [test-iqa6e/manhole-e0p0b](https://universe.roboflow.com/test-iqa6e/manhole-e0p0b) | 1283 | Good 773, Lose 295, Tilted 284, Displaced 230 | Те же китайские фото (`well*`, `net1`, `synth`), но размечены по-другому |
| [qassim-university-rgdar/open-manhole-image-dataset6](https://universe.roboflow.com/qassim-university-rgdar/open-manhole-image-dataset6) | 2922 | Open Manhole 3003, Manhole 907, Pothole 346 | Другой регион, много открытых люков |

Что значат классы китайского набора (по превью и по тому, как их обычно трактуют; уточнить на картинках):
- `good`: крышка на месте, исправна;
- `broke`: крышка сломана или треснула;
- `lose`: крышки нет, колодец открыт;
- `uncovered`: крышка есть, но лежит криво или неплотно, колодец частично открыт;
- `circle`: повреждено кольцо или покрытие вокруг люка, крышка просела или выступает.

⚠️ **Многие датасеты — копии одного китайского источника.** Нельзя брать dev из одной копии,
а test из другой: одинаковые фото окажутся в обеих частях. Выбираем **одну** копию.

## Опоры

| Датасет | Фото | Классы | Комментарий |
|---|---|---|---|
| [ghost-gsj7h/ghost05-utility-pole](https://universe.roboflow.com/ghost-gsj7h/ghost05-utility-pole) | 1200 | pole_broken 1192, pole 1126, creeper 421 | Сломанные опоры + заросшие растительностью. Проверить, какие там опоры |
| [infosys-uyzkr/electric-pole-damages](https://universe.roboflow.com/infosys-uyzkr/electric-pole-damages) | 604 | Undamaged 291, Foreign object 211, Sparked Transformer 132, Damaged 119 | Опоры ЛЭП, похоже на Индию |
| [pea-n1-g7/pole-vuxpf](https://universe.roboflow.com/pea-n1-g7/pole-vuxpf) | 208 | Normal 151, Object 48, Vines 28, Damage 13 | Мало повреждений |

Опоры двойного назначения МГТС — это металлические фонарные столбы с оборудованием связи
(см. слайд). Подходящего датасета с их повреждениями нет. Опоры ЛЭП и деревянные столбы
годятся только как приблизительные примеры.

## Не брать
- `shreekhant/dash-object-detection`: ямы и ливнёвки, используем только как трудные отрицательные примеры.
- `old building damage detectionv2`: трещины в бетоне без объекта.
- Опоры с одним классом `pole`: состояния нет.

## План использования
- **Шаг 0 (#1):** 30 фото из `jie-yu/manhole-cover-4` (по 6 на класс) + 5–10 из `ghost05-utility-pole`.
- **dev:** можно добавить ещё из `jie-yu`. **test** — только наши московские кадры.
- Перед использованием разметку на выбранных фото проверяем глазами.
