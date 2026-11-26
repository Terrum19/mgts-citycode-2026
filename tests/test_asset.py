"""Паспорт актива: идентификатор, функциональные последствия, рекомендация, срочность."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asset import (assess_frame, enrich_result, find_tag_id, load_registry,  # noqa: E402
                   normalize_asset_id)

FRAME_BAD = {"objects": [{"type": "РЩ", "bbox": None, "status": "bad", "damage": "МП"}]}
FRAME_OK = {"objects": [{"type": "ТКШ", "bbox": None, "status": "ok", "damage": None}]}


class AssetIdTest(unittest.TestCase):
    def test_normalizes_variants(self):
        for value in ("БС-12235", "BS 12235", "bc-12235", "БС 12235", "бс-12235", "БС12235"):
            with self.subTest(value=value):
                self.assertEqual(normalize_asset_id(value), "БС-12235")

    def test_rejects_non_ids(self):
        for value in ("23349371", "", None, "ПАО МТС БС-12235", "12235", True, 42):
            with self.subTest(value=value):
                self.assertIsNone(normalize_asset_id(value))

    def test_tag_requires_single_candidate(self):
        self.assertEqual(find_tag_id("ПАО МТС БС-12235"), "БС-12235")
        self.assertIsNone(find_tag_id("БС-12235 и БС-99999"), "неоднозначная бирка не должна выбирать первый ID")
        self.assertIsNone(find_tag_id("без номера"))


class AssessFrameTest(unittest.TestCase):
    def test_decision_table(self):
        cases = [
            ({"status": "ok"}, "", "not_assessed", "routine"),
            ({"status": "bad", "damage": "Неудовлетворительное состояние ЛКП"}, "", "coating", "planned"),
            ({"status": "bad", "damage": "МП"}, "", "envelope_or_mounting", "unscheduled"),
            ({"status": "bad", "damage": "coating_corrosion"}, "", "coating", "planned"),
            ({"status": "bad", "damage": "mechanical_damage"}, "", "envelope_or_mounting", "unscheduled"),
            ({"status": "bad", "damage": "МП"}, "ТКШ удерживается на питающем кабеле", "mounting", "emergency"),
            ({"status": "bad", "damage": "МП"}, "Нарушение степени защиты оболочки РЩ", "envelope", "unscheduled"),
            ({"status": "review"}, "герметичность", "unknown", "manual_review"),
        ]
        for frame, text, function, urgency in cases:
            with self.subTest(frame=frame, text=text):
                got = assess_frame(frame, evidence_text=text)
                self.assertEqual((got["function"], got["urgency"]), (function, urgency))
                self.assertTrue(got["recommendation"] and got["effect"] and got["basis"])

    def test_text_never_downgrades_bad(self):
        got = assess_frame({"status": "bad", "damage": "МП"}, evidence_text="объект в норме, дефектов нет")
        self.assertEqual(got["function"], "envelope_or_mounting")
        self.assertEqual(got["urgency"], "unscheduled")

    def test_review_is_never_ok(self):
        got = assess_frame({"status": "review"})
        self.assertNotEqual(got["urgency"], "routine")
        self.assertEqual(got["function"], "unknown")

    def test_frame_with_any_bad_object_is_bad(self):
        frame = {"objects": [{"status": "ok", "damage": None}, {"status": "bad", "damage": "МП"}]}
        self.assertEqual(assess_frame(frame)["function"], "envelope_or_mounting")


class EnrichResultTest(unittest.TestCase):
    def test_no_registry_no_coordinates(self):
        result = enrich_result(FRAME_BAD)
        self.assertEqual(result["scope"], "frame")
        self.assertIsNone(result["asset"]["lat"])
        self.assertIsNone(result["asset"]["lon"])
        self.assertEqual(result["asset"]["registry_status"], "not_provided")
        self.assertEqual(result["asset"]["identity_status"], "missing")

    def test_registry_match_copies_coordinates(self):
        registry = load_registry([{"asset_id": "БС-12235", "lat": 55.7363, "lon": 37.6743, "address": "Москва"}])
        result = enrich_result(FRAME_BAD, meta={"asset_id": "БС-12235"}, registry=registry)
        self.assertEqual(result["asset"]["registry_status"], "matched")
        self.assertEqual(result["asset"]["lat"], 55.7363)
        self.assertEqual(result["asset"]["coords_source"], "registry")

    def test_duplicate_registry_rejected(self):
        with self.assertRaises(ValueError):
            load_registry([{"asset_id": "БС-12235"}, {"asset_id": "БС-12235"}])

    def test_tag_is_candidate_and_not_matched(self):
        registry = load_registry([{"asset_id": "БС-12235", "lat": 1.0, "lon": 2.0}])
        result = enrich_result(FRAME_BAD, meta={"tag_text": "ПАО МТС БС-12235"}, registry=registry)
        self.assertEqual(result["asset"]["identity_status"], "candidate")
        # Неподтверждённый ID не сверяется с реестром: сверять нечего.
        self.assertEqual(result["asset"]["registry_status"], "not_provided")
        self.assertIsNone(result["asset"]["lat"], "неподтверждённый ID не должен давать координаты")

    def test_invalid_explicit_id(self):
        result = enrich_result(FRAME_BAD, meta={"asset_id": "23349371"})
        self.assertEqual(result["asset"]["identity_status"], "invalid")
        self.assertIsNone(result["asset"]["asset_id"])

    def test_exif_does_not_become_asset_coordinates(self):
        """EXIF — метаданные снимка, а не координата актива: без записи реестра lat/lon пусты."""
        result = enrich_result(FRAME_BAD, meta={"lat": 55.7, "lon": 37.6})
        self.assertIsNone(result["asset"]["lat"])
        self.assertIsNone(result["asset"]["lon"])
        self.assertEqual(result["asset"]["coords_source"], "none")
        self.assertEqual((result["asset"]["photo_lat"], result["asset"]["photo_lon"]), (55.7, 37.6))

    def test_bad_exif_coordinates_dropped(self):
        result = enrich_result(FRAME_BAD, meta={"lat": 999, "lon": 999})
        self.assertIsNone(result["asset"]["photo_lat"])
        self.assertIsNone(result["asset"]["lat"])

    def test_objects_scope_when_bbox_present(self):
        pred = {"objects": [{"type": "РЩ", "bbox": [1, 2, 30, 40], "status": "bad", "damage": "МП"}]}
        self.assertEqual(enrich_result(pred)["scope"], "objects")

    def test_input_not_mutated(self):
        before = dict(FRAME_OK)
        enrich_result(FRAME_OK)
        self.assertEqual(FRAME_OK, before)


class ExportRowTest(unittest.TestCase):
    def test_passport_columns_present(self):
        from predict import object_rows
        pred = {"scope": "frame", **FRAME_BAD,
                "asset": {"asset_id": "БС-12235", "identity_status": "confirmed", "identity_source": "meta",
                          "registry_status": "matched", "lat": 55.7, "lon": 37.6, "address": "Москва",
                          "coords_source": "registry"},
                "assessment": {"effect": "e", "recommendation": "r", "urgency": "emergency", "basis": "b"}}
        row = object_rows("a.jpg", pred, {}, {"РЩ": "РЩ"})[0]
        for key in ("scope", "asset_id", "urgency", "recommendation", "registry_lat", "coords_source"):
            self.assertIn(key, row)
        self.assertEqual((row["asset_id"], row["urgency"], row["scope"]), ("БС-12235", "emergency", "frame"))

    def test_photo_keys_do_not_collapse_same_names(self):
        from predict import photo_keys
        paths = [Path("data/dss_test/ТКШ/МП/20211214.jpg"), Path("data/dss_test/РЩ/ЛКП/20211214.jpg")]
        keys = photo_keys(paths)
        self.assertEqual(len(set(keys.values())), 2, "одноимённые фото из разных папок должны различаться")


if __name__ == "__main__":
    unittest.main()
