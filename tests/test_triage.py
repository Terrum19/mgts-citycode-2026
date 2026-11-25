"""Object-level triage: matched/unmatched FP, bad_to_*, review rate и cost без двойного счёта."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate  # noqa: E402


def obj(box, status, typ="manhole"):
    return {"type": typ, "bbox": list(box), "status": status, "damage": None,
            "description": "", "review_reasons": ["low_confidence"] if status == "review" else []}


def label(objects):
    return {"file": "a.jpg", "width": 100, "height": 100, "split": "dev",
            "frame_flags": [], "location_id": "a", "objects": objects}


def run(gt, pr):
    pred = {"objects": [obj(*p) for p in pr]}
    return evaluate.evaluate([label([obj(*g) for g in gt])], {"a.jpg": pred},
                             cost={"fn": 20, "fp": 2, "review": 1})


GT_BAD = ((0, 0, 10, 10), "bad")
GT_OK = ((20, 20, 30, 30), "ok")
GT_UNCLEAR = ((40, 40, 50, 50), "unclear")


class TriageTest(unittest.TestCase):
    def test_bad_review_and_ok_bad(self):
        m = run([GT_BAD, GT_OK, GT_UNCLEAR],
                [((0, 0, 10, 10), "review"), ((20, 20, 30, 30), "bad"), ((40, 40, 50, 50), "ok")])
        self.assertEqual(m["bad_to_review"], 1)
        self.assertEqual(m["ok_to_bad"], 1)
        self.assertEqual(m["triage_tp"], 1)
        self.assertEqual(m["triage_fn"], 0)
        self.assertEqual(m["triage_fp"], 1)          # ok_to_bad
        self.assertEqual(m["triage_fp_auto"], 1)
        self.assertEqual(m["triage_fp_review"], 0)
        self.assertEqual(m["review_objects"], 1)
        # FN=0, FP(автоматический)=1, review=1 -> 2 + 1
        self.assertEqual(m["triage_cost"], 3)

    def test_bad_to_ok_is_missed_but_bad_to_review_is_not(self):
        m = run([GT_BAD], [((0, 0, 10, 10), "ok")])
        self.assertEqual(m["bad_to_ok"], 1)
        self.assertEqual(m["triage_fn"], 1)
        self.assertEqual(m["missed_bad_after_triage"], 1)
        self.assertEqual(m["review_objects"], 0)
        m2 = run([GT_BAD], [((0, 0, 10, 10), "review")])
        self.assertEqual(m2["bad_to_review"], 1)
        self.assertEqual(m2["triage_fn"], 0)
        self.assertEqual(m2["missed_bad_after_triage"], 0)

    def test_review_of_ok_is_charged_once(self):
        m = run([GT_OK], [((20, 20, 30, 30), "review")])
        self.assertEqual(m["ok_to_review"], 1)
        self.assertEqual(m["triage_fp"], 1)
        self.assertEqual(m["triage_fp_auto"], 0)
        self.assertEqual(m["triage_fp_review"], 1)
        self.assertEqual(m["triage_cost"], 1)   # только review, не FP+review

    def test_spurious_predictions(self):
        m = run([], [((0, 0, 10, 10), "bad"), ((20, 20, 30, 30), "review")])
        self.assertEqual(m["spurious_bad"], 1)
        self.assertEqual(m["spurious_review"], 1)
        self.assertEqual(m["triage_fp_unmatched"], 2)
        self.assertEqual(m["review_objects"], 1)

    def test_review_rate_denominator(self):
        m = run([GT_OK, GT_OK], [((20, 20, 30, 30), "review"), ((60, 60, 70, 70), "ok")])
        self.assertEqual(m["review_objects"], 1)
        self.assertAlmostEqual(m["review_rate_objects"], 1 / 2)
        self.assertEqual(m["review_frames"], 1)


if __name__ == "__main__":
    unittest.main()
