"""evaluate: frame-only, mixed, legacy bbox=None, target_not_visible и invalid bbox."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate  # noqa: E402
from results import make_frame, make_result  # noqa: E402

COST = {"fn": 20, "fp": 2, "review": 1}


def object(box=(0, 0, 10, 10), status="ok", typ="manhole"):
    return {"type": typ, "bbox": list(box), "status": status, "damage": None, "description": ""}


def label(file, objects, flags=(), split="dev"):
    return {"file": file, "width": 100, "height": 100, "split": split,
            "frame_flags": list(flags), "location_id": file, "objects": objects}


class FrameEvaluationTest(unittest.TestCase):
    def test_frame_only_has_no_detection_metrics(self):
        labels = [label("a.jpg", [object()], flags=["bad_frame"]),
                  label("b.jpg", [object(status="bad")])]
        preds = {"b.jpg": make_result(frame=make_frame("bad", type="manhole", damage="cover_broken"),
                                      objects=[], scope="frame")}
        m = evaluate.evaluate(labels, preds, cost=COST)
        self.assertEqual(m["evaluation_mode"], "frame")
        self.assertEqual(m["images"], 1)
        self.assertEqual(m["skipped_bad_frames"], 1)
        self.assertIsNone(m["det_precision"])
        self.assertIsNone(m["det_recall"])
        self.assertEqual(m["gt_bad_frames"], 1)
        self.assertEqual(m["frame_bad_to_bad"], 1)

    def test_mixed_native_frame_and_objects(self):
        labels = [label("b.jpg", [object(status="bad")])]
        pred = make_result(objects=[object(status="bad")], scope="mixed")
        m = evaluate.evaluate(labels, {"b.jpg": pred}, cost=COST)
        self.assertEqual(m["evaluation_mode"], "mixed")
        self.assertEqual(m["repair_tp"], 1)
        self.assertEqual(m["frame_bad_to_bad"], 1)

    def test_legacy_photodino_bbox_none_is_frame(self):
        labels = [label("b.jpg", [object(status="bad")])]
        pred = {"objects": [{"type": "manhole", "bbox": None, "status": "bad",
                             "damage": "cover_broken", "description": ""}]}
        m = evaluate.evaluate(labels, {"b.jpg": pred}, cost=COST)
        self.assertEqual(m["evaluation_mode"], "frame")
        self.assertEqual(m["legacy_photo_dino"], 1)
        self.assertEqual(m["frame_bad_to_bad"], 1)
        self.assertIsNone(m["det_recall"])

    def test_invalid_pred_bbox_is_counted_not_fatal(self):
        labels = [label("a.jpg", [])]
        pred = {"objects": [{"type": "manhole", "bbox": None, "status": "bad"},
                             {"type": "manhole", "bbox": [0, 0, 0, 1], "status": "ok"}], "error": None}
        m = evaluate.evaluate(labels, {"a.jpg": pred}, cost=COST)
        self.assertEqual(m["invalid_bbox_objects"], 2)
        self.assertEqual(m["review_frames"], 1)

    def test_missing_prediction_is_review(self):
        labels = [label("a.jpg", [])]
        m = evaluate.evaluate(labels, {}, cost=COST)
        self.assertEqual(m["missing_predictions"], 1)
        self.assertEqual(m["review_frames"], 1)
        self.assertEqual(m["frame_ok_to_review"], 1)
        self.assertEqual(m["review_rate"], 1.0)

    def test_target_not_visible_is_a_valid_negative(self):
        labels = [label("a.jpg", [], flags=["target_not_visible"])]
        m = evaluate.evaluate(labels, {}, cost=COST)
        self.assertEqual(m["images"], 1)
        self.assertEqual(m["gt_bad_frames"], 0)
        self.assertEqual(m["review_frames"], 1)  # нет предсказания -> review, не ok

    def test_target_not_visible_with_objects_is_invalid_label(self):
        labels = [label("a.jpg", [object()], flags=["target_not_visible"])]
        m = evaluate.evaluate(labels, {}, cost=COST)
        self.assertTrue(any(x["reason"] == "target_not_visible_with_objects"
                            for x in m["invalid_labels"]))
        self.assertEqual(m["gt_frames"], 0)


if __name__ == "__main__":
    unittest.main()
