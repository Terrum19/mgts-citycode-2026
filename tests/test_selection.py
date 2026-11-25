"""Image-level отбор: ограничения, cost, baselines, tie-break и отсутствие eligible."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import selection  # noqa: E402

COST = {"fn": 20, "fp": 2, "review": 1}


def row(name, base, decisions, y_bad, *, uses_vlm=False, serializable=True):
    m = selection.frame_metrics(y_bad, decisions, COST)
    return {"name": name, "base": base, "uses_vlm": uses_vlm, "serializable": serializable, **m}


class FrameMetricsTest(unittest.TestCase):
    def test_all_review_has_full_triage_recall_but_full_review_rate(self):
        m = selection.frame_metrics([True, False], [{"status": "review", "review": True}] * 2, COST)
        self.assertEqual(m["triage_recall"], 1.0)
        self.assertEqual(m["review_rate"], 1.0)
        self.assertEqual(m["N_FN"], 0)
        self.assertEqual(m["N_FP"], 0)
        self.assertEqual(m["operational_cost"], 2)

    def test_all_bad_has_fp_cost(self):
        m = selection.frame_metrics([True, False], [{"status": "bad"}] * 2, COST)
        self.assertEqual(m["N_FN"], 0)
        self.assertEqual(m["N_FP"], 1)
        self.assertEqual(m["operational_cost"], 2)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            selection.frame_metrics([True], [], COST)


class SelectTest(unittest.TestCase):
    def test_rejects_all_review(self):
        y = [True, True, False, False, False]
        all_review = row("review", "logreg", [{"status": "review", "review": True}] * 5, y)
        good = row("good", "logreg",
                   [{"status": "bad"}, {"status": "bad"}, {"status": "ok"},
                    {"status": "ok"}, {"status": "review", "review": True}], y)
        out = selection.select_candidate([all_review, good],
                                         {"min_bad": 1, "min_ok": 1, "min_triage_recall": 0.7,
                                          "max_review_rate": 0.35})
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["chosen"]["name"], "good")

    def test_no_eligible_is_selection_failed(self):
        y = [True, False, False]
        all_review = row("review", "logreg", [{"status": "review", "review": True}] * 3, y)
        out = selection.select_candidate([all_review], {"min_bad": 1, "min_ok": 1,
                                                        "max_review_rate": 0.2})
        self.assertEqual(out["status"], "selection_failed")
        self.assertIsNone(out["chosen"])
        self.assertEqual(out["reason"], "constraints_failed")

    def test_support_failure(self):
        y = [True, False]
        r = row("x", "logreg", [{"status": "bad"}, {"status": "ok"}], y)
        out = selection.select_candidate([r], {"min_bad": 3, "min_ok": 5})
        self.assertEqual(out["status"], "selection_failed")
        self.assertEqual(out["reason"], "support_failed")

    def test_tie_break_is_deterministic_and_prefers_simple(self):
        y = [True, False, False]
        d = [{"status": "bad"}, {"status": "ok"}, {"status": "review", "review": True}]
        a = row("b", "tail", d, y)
        b = row("a", "logreg", d, y)
        out = selection.select_candidate([a, b], {"min_bad": 1, "min_ok": 1, "max_review_rate": 1})
        self.assertEqual(out["chosen"]["base"], "logreg")

    def test_vlm_only_not_serializable(self):
        y = [True, False]
        r = row("vlm", "vlm", [{"status": "bad"}, {"status": "ok"}], y, uses_vlm=True, serializable=False)
        out = selection.select_candidate([r], {"min_bad": 1, "min_ok": 1, "max_review_rate": 1})
        self.assertEqual(out["status"], "selection_failed")
        self.assertIn("no_serializable_production_path", out["rejected"][0]["reject_reasons"])

    def test_baselines_present(self):
        b = selection.baselines([True, False], COST)
        self.assertIn("all_review", b)
        self.assertIn("all_bad", b)
        self.assertIn("all_ok", b)


if __name__ == "__main__":
    unittest.main()
