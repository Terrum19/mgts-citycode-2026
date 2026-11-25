"""Единый контракт результата: frame/objects/mixed, review, invalid bbox."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import results  # noqa: E402


def obj(box=(0, 0, 10, 10), status="ok", typ="manhole", damage=None, reasons=None):
    return {"type": typ, "bbox": list(box) if box is not None else None, "status": status,
            "damage": damage, "description": "", "review_reasons": reasons or []}


class BboxTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(results.valid_bbox([0, 1, 2, 3]), (0.0, 1.0, 2.0, 3.0))
        self.assertEqual(results.valid_bbox(["0", "1", "2.5", "3"]), (0.0, 1.0, 2.5, 3.0))

    def test_reject(self):
        for box in (None, [0, 0, 0, 1], [0, 0, 10], [0, 0, 10, float("nan")],
                    [True, 0, 10, 10], ["a", 0, 1, 1], [10, 10, 0, 0]):
            with self.subTest(box=box):
                self.assertIsNone(results.valid_bbox(box))

    def test_sanitize_records_reason(self):
        good, bad = results.sanitize_objects([obj(), obj(box=None), obj(box=[0, 0, 0, 1])])
        self.assertEqual(len(good), 1)
        self.assertEqual([x["reason"] for x in bad], ["missing", "degenerate"])


class FrameRulesTest(unittest.TestCase):
    def test_bad_and_review_both_kept(self):
        r = results.make_result(objects=[obj(status="bad", damage="cover_broken"),
                                         obj(box=(20, 20, 30, 30), status="review",
                                             reasons=["low_confidence"])])
        self.assertEqual(r["scope"], "objects")
        self.assertEqual(r["frame"]["status"], "bad")
        self.assertTrue(r["frame"]["review"])
        self.assertIn("low_confidence", r["frame"]["review_reasons"])
        self.assertEqual(results.triage_route(r["frame"]), "review")

    def test_native_frame_has_no_objects(self):
        frame = results.make_frame("bad", type="manhole", damage="cover_broken")
        r = results.make_result(frame=frame, objects=[], scope="frame")
        self.assertEqual(r["scope"], "frame")
        self.assertEqual(r["objects"], [])
        self.assertEqual(r["frame"]["source"], "native")
        self.assertEqual(results.triage_route(r["frame"]), "repair")

    def test_successful_empty_is_ok_not_review(self):
        r = results.make_result(objects=[], scope="mixed")
        self.assertEqual(r["frame"]["status"], "ok")
        self.assertFalse(r["frame"]["review"])

    def test_error_with_bad_is_not_review(self):
        r = results.make_result(objects=[obj(status="bad")], error="timeout")
        self.assertEqual(r["frame"]["status"], "bad")
        self.assertFalse(r["frame"]["review"])

    def test_error_without_bad_is_review(self):
        r = results.make_result(objects=[obj(status="ok")], error="timeout")
        self.assertEqual(r["frame"]["status"], "review")
        self.assertTrue(r["frame"]["review"])
        self.assertIn("vlm_error", r["frame"]["review_reasons"])

    def test_invalid_bbox_requires_review_but_does_not_crash(self):
        r = results.make_result(objects=[{"type": "manhole", "bbox": None, "status": "bad"}])
        self.assertEqual(r["objects"], [])
        self.assertEqual(r["invalid_bbox_objects"][0]["reason"], "missing")
        self.assertTrue(r["frame"]["review"])
        self.assertEqual(results.triage_route(r["frame"]), "review")

    def test_schema_version_and_scope_validation(self):
        r = results.make_result(objects=[], scope="nonsense")
        self.assertEqual(r["schema_version"], results.SCHEMA_VERSION)
        self.assertIn(r["scope"], results.SCOPES)

    def test_review_reason_normalization(self):
        self.assertEqual(results.normalize_reasons(["vlm_error", "bad_json"]), ["bad_json", "vlm_error"])
        self.assertEqual(results.normalize_reasons(["review"]), ["low_confidence"])


if __name__ == "__main__":
    unittest.main()
