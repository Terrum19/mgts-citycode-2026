"""Паспорт актива: идентификатор объекта, функциональные последствия, рекомендация и срочность.

Модуль намеренно без torch, PIL и сети: он превращает уже полученный результат классификации
в рабочую запись для диспетчера — «какой объект, какая функция нарушена, что делать и насколько
срочно», а не в набор вероятностей.

    python src/asset.py            # самопроверка правил на фиксированных примерах

Границы ответственности:
  * `assess_frame` описывает последствия и маршрут, но НИКОГДА не понижает `bad` до `ok`;
  * `enrich_result` добавляет поля `asset` и `assessment`, не меняя `objects` и `status`;
  * координаты берутся только из подтверждённого реестра или из EXIF-метаданных;
    ничего не вычисляется и не подставляется «на глаз».
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# Идентификатор нанесён на шкаф вручную: «ПАО МТС БС-12235», встречаются BS/BC и пробелы.
ASSET_RE = re.compile(r"^\s*(?:БС|BS|BC)\s*[-–—]?\s*(\d{4,8})\s*$", re.IGNORECASE)
TAG_RE = re.compile(r"(?:БС|BS|BC)\s*[-–—]?\s*(\d{4,8})", re.IGNORECASE)

# Признаки того, что нарушено крепление: объект держится на кабеле или уже упал.
MOUNT_PHRASES = ("удержива", "отрыв", "падени", "сорван", "упал", "висит на", "узлов крепления",
                 "узлы крепления", "узла крепления")
# Признаки нарушения защитной оболочки: герметичность, степень защиты, отсутствие дверцы.
ENVELOPE_PHRASES = ("герметичн", "степень защиты", "степени защиты", "оболочк", "дверц",
                    "отсутствует дверь", "отсутствие дверцы")

FUNCTION_EFFECT = {
    "coating": "Нарушено защитное покрытие корпуса",
    "envelope": "Нарушена целостность защитной оболочки",
    "mounting": "Нарушено крепление объекта на опоре",
    "envelope_or_mounting": "Возможна потеря целостности оболочки или крепления",
    "not_assessed": "Функциональных нарушений не выявлено",
    "unknown": "Состояние не подтверждено",
}

RECOMMENDATION = {
    "coating": "Плановое восстановление ЛКП с проверкой коррозии металла",
    "envelope": "Внеплановый осмотр: восстановить защитную оболочку",
    "mounting": "Ограничить доступ и направить аварийную бригаду",
    "envelope_or_mounting": "Внеплановый осмотр: проверить оболочку и узлы крепления",
    "not_assessed": "Объект в норме; следующий плановый осмотр",
    "unknown": "Обязательная проверка специалистом",
}

URGENCY = {
    "coating": "planned",
    "envelope": "unscheduled",
    "mounting": "emergency",
    "envelope_or_mounting": "unscheduled",
    "not_assessed": "routine",
    "unknown": "manual_review",
}


def normalize_asset_id(value: object) -> str | None:
    """«БС-12235», «BS 12235», «BC-12235» -> «БС-12235». Произвольные числа не принимаются."""
    if value is None or isinstance(value, bool):
        return None
    match = ASSET_RE.match(str(value))
    if not match:
        return None
    return f"БС-{match.group(1)}"


def find_tag_id(text: str) -> str | None:
    """Первый идентификатор, найденный в распознанном тексте бирки. Несколько разных — неоднозначно."""
    found = {f"БС-{m.group(1)}" for m in TAG_RE.finditer(text or "")}
    return found.pop() if len(found) == 1 else None


def _status_and_damage(frame: Mapping[str, Any]) -> tuple[str, str | None]:
    """Свести кадр к одному наблюдению: любой повреждённый объект делает кадр повреждённым."""
    objects = frame.get("objects")
    if isinstance(objects, list) and objects:
        statuses = [str(o.get("status", "")) for o in objects]
        bad = [o for o in objects if str(o.get("status")) == "bad"]
        if bad:
            return "bad", bad[0].get("damage")
        if "review" in statuses:
            return "review", None
        return "ok", None
    return str(frame.get("status", "")), frame.get("damage")


def assess_frame(frame: Mapping[str, Any], *, evidence_text: str = "") -> dict:
    """Функциональные последствия, рекомендация и срочность для одного наблюдения."""
    status, damage = _status_and_damage(frame)
    damage_upper = str(damage or "").upper()
    text = (evidence_text or "").casefold()

    if status == "ok":
        function, basis = "not_assessed", "status=ok"
    elif status == "review":
        function, basis = "unknown", "status=review"
    elif status != "bad":
        function, basis = "unknown", f"status={status or 'нет'}"
    elif "ЛКП" in damage_upper:
        function, basis = "coating", "status=bad; damage=ЛКП"
    elif "МП" in damage_upper:
        # Текст описания может уточнить функцию и повысить срочность, но не отменить дефект.
        if any(w in text for w in MOUNT_PHRASES):
            function, basis = "mounting", "status=bad; damage=МП; текст: крепление"
        elif any(w in text for w in ENVELOPE_PHRASES):
            function, basis = "envelope", "status=bad; damage=МП; текст: оболочка"
        else:
            function, basis = "envelope_or_mounting", "status=bad; damage=МП"
    else:
        function, basis = "unknown", f"status=bad; damage={damage or 'нет'}"

    return {"function": function, "effect": FUNCTION_EFFECT[function],
            "recommendation": RECOMMENDATION[function], "urgency": URGENCY[function],
            "basis": basis}


def load_registry(rows: list[dict]) -> dict[str, dict]:
    """Индекс реестра по нормализованному идентификатору. Дубликаты — ошибка, а не «первая строка»."""
    registry: dict[str, dict] = {}
    duplicates = set()
    for row in rows or []:
        asset_id = normalize_asset_id(row.get("asset_id") or row.get("ID объекта"))
        if asset_id is None:
            continue
        if asset_id in registry:
            duplicates.add(asset_id)
            continue
        registry[asset_id] = {
            "asset_id": asset_id,
            "lat": _coord(row.get("lat")),
            "lon": _coord(row.get("lon")),
            "address": (row.get("address") or row.get("адрес") or "").strip() or None,
        }
    if duplicates:
        raise ValueError(f"Повторяющиеся идентификаторы в реестре: {', '.join(sorted(duplicates))}")
    return registry


def _coord(value: object) -> float | None:
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number


def enrich_result(pred: Mapping[str, Any], *, meta: Mapping[str, Any] | None = None,
                  registry: Mapping[str, dict] | None = None) -> dict:
    """Добавить паспорт актива к результату. Возвращает новый dict, вход не мутируется."""
    meta = meta or {}
    objects = pred.get("objects") or []
    frame_scope = not objects or all(o.get("bbox") is None for o in objects)

    explicit = meta.get("asset_id")
    asset_id = normalize_asset_id(explicit)
    if explicit not in (None, "") and asset_id is None:
        identity_status, identity_source = "invalid", "meta"
    elif asset_id is not None:
        identity_status, identity_source = "confirmed", "meta"
    else:
        tag = find_tag_id(str(meta.get("tag_text") or ""))
        if tag is not None:
            asset_id, identity_status, identity_source = tag, "candidate", "tag_text"
        else:
            identity_status, identity_source = "missing", "none"

    lat = lon = None
    address = None
    registry_status = "not_provided"
    coords_source = "none"
    if registry is not None and identity_status == "confirmed":
        record = registry.get(asset_id)
        if record is not None:
            registry_status, coords_source = "matched", "registry"
            lat, lon, address = record["lat"], record["lon"], record["address"]
        else:
            registry_status = "not_found"
    else:
        # Сверять нечего: реестр не передан либо идентификатор не подтверждён.
        registry_status = "not_provided"

    # EXIF — сырые метаданные фотографии, а НЕ координата актива. В asset.lat/lon попадает
    # только точная запись реестра; иначе вызывающий код принял бы координату снимка за объект.
    photo_lat, photo_lon = _coord(meta.get("lat")), _coord(meta.get("lon"))
    if not (photo_lat is not None and photo_lon is not None and -90 <= photo_lat <= 90 and -180 <= photo_lon <= 180):
        photo_lat = photo_lon = None

    result = dict(pred)
    result["scope"] = "frame" if frame_scope else "objects"
    result["asset"] = {"asset_id": asset_id, "identity_status": identity_status,
                       "identity_source": identity_source, "registry_status": registry_status,
                       "lat": lat, "lon": lon, "address": address, "coords_source": coords_source,
                       "photo_lat": photo_lat, "photo_lon": photo_lon}
    result["assessment"] = assess_frame(pred, evidence_text=str(meta.get("tag_text") or pred.get("description") or ""))
    return result


def main() -> None:
    cases = [
        ({"status": "ok"}, "", "not_assessed", "routine"),
        ({"status": "bad", "damage": "Неудовлетворительное состояние ЛКП"}, "", "coating", "planned"),
        ({"status": "bad", "damage": "МП"}, "", "envelope_or_mounting", "unscheduled"),
        ({"status": "bad", "damage": "МП"}, "ТКШ удерживается на питающем кабеле", "mounting", "emergency"),
        ({"status": "bad", "damage": "МП"}, "Нарушение степени защиты оболочки", "envelope", "unscheduled"),
        ({"status": "review"}, "герметичность", "unknown", "manual_review"),
    ]
    for frame, text, function, urgency in cases:
        got = assess_frame(frame, evidence_text=text)
        ok = got["function"] == function and got["urgency"] == urgency
        print(f"{'OK ' if ok else 'FAIL'} {frame.get('status'):6} {function:20} {got['urgency']:12} {got['recommendation']}")
    for value in ("БС-12235", "BS 12235", "bc-12235", "ПАО МТС БС-12235", "23349371", "", None):
        print(f"  {value!r:24} -> {normalize_asset_id(value)}")


if __name__ == "__main__":
    main()
