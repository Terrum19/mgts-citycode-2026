"""Логика каскада: суммарные P(ok)/P(bad), margin, поддержка классов, решение review.

Синтетические вероятности — без torch/sklearn/сети.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cascade  # noqa: E402

OK_MANHOLE = ["manhole", "ok", None]
BROKEN = ["manhole", "bad", "cover_broken"]
OPEN = ["manhole", "bad", "cover_open_or_shifted"]
POLE_OK = ["pole", "ok", None]
POLE_BAD = ["pole", "bad", "pole_tilted"]
CLASSES = [BROKEN, OPEN, OK_MANHOLE, POLE_BAD, POLE_OK]
LABELS = [str(c) for c in CLASSES]
CFG = {"p_bad_min": 0.60, "p_bad_max_ok": 0.40, "status_conf_min": 0.65, "margin_min": 0.10}


def summary(scores, support=None, type_hint="manhole", policy="detector", min_support=2):
    return cascade.summarize(CLASSES, scores, labels=LABELS, support=support, type_hint=type_hint,
                             policy=policy, min_support=min_support)


class SummarizeTest(unittest.TestCase):
    def test_two_bad_damages_sum(self):
        # старый порог 0.55 по одному кортежу не проходил: 0.42 < 0.55 -> review
        s = summary([0.42, 0.42, 0.16, 0.0, 0.0])
        self.assertAlmostEqual(s["p_bad"], 0.84)
        self.assertAlmostEqual(s["p_ok"], 0.16)
        self.assertAlmostEqual(s["status_confidence"], 0.84)
        self.assertTrue(s["ready"])
        self.assertEqual(cascade.decide(s, CFG), "bad")

    def test_ok_summed(self):
        s = summary([0.20, 0.05, 0.75, 0.0, 0.0])
        self.assertAlmostEqual(s["p_ok"], 0.75)
        self.assertEqual(cascade.decide(s, CFG), "ok")

    def test_detector_policy_ignores_other_types(self):
        # опора получила 0.9, но детектор сказал «люк»: её вероятность не должна решать судьбу люка
        s = summary([0.0, 0.0, 0.20, 0.80, 0.0])
        self.assertAlmostEqual(s["p_bad"], 0.0)
        self.assertAlmostEqual(s["p_ok"], 0.20)

    def test_joint_policy_uses_top_type_only(self):
        # тип выбирает лучший поддержанный кортеж: классы другого типа не разбавляют P(bad)
        s = summary([0.0, 0.0, 0.20, 0.70, 0.10], policy="joint")
        self.assertEqual(s["eligible"], [3, 4])
        self.assertAlmostEqual(s["p_bad"], 0.70)
        self.assertAlmostEqual(s["p_ok"], 0.10)
        self.assertAlmostEqual(s["margin"], 0.60)
        self.assertEqual(cascade.pick(s, CLASSES, "bad"), ("pole", "pole_tilted"))

    def test_joint_does_not_sum_bad_across_types(self):
        s = summary([0.45, 0.0, 0.15, 0.40, 0.0], policy="joint")
        self.assertAlmostEqual(s["p_bad"], 0.45)     # не 0.85: 0.40 опоры — другой тип
        self.assertEqual(cascade.decide(s, CFG), "review")

    def test_detector_without_hint_uses_top_type_only(self):
        s = summary([0.45, 0.0, 0.15, 0.40, 0.0], type_hint=None)
        self.assertAlmostEqual(s["p_bad"], 0.45)
        self.assertEqual(s["eligible"], [0, 1, 2])  # все классы выбранного типа «люк»

    def test_joint_ready_within_selected_type(self):
        # у люка нет поддержанного ok: наличие ok у опоры готовность не даёт
        classes = [BROKEN, POLE_BAD, POLE_OK]
        labels = [str(c) for c in classes]
        s = cascade.summarize(classes, [0.90, 0.05, 0.05], labels=labels, policy="joint")
        self.assertEqual(s["eligible"], [0])
        self.assertFalse(s["ready"])
        self.assertEqual(cascade.decide(s, CFG), "review")

    def test_no_renormalization(self):
        # 0.30 массы не поддержано: P(bad)+P(ok) не обязаны давать 1, уверенность падает
        s = summary([0.35, 0.35, 0.0, 0.0, 0.0], support={LABELS[0]: 5, LABELS[1]: 1, LABELS[2]: 5,
                                                          LABELS[3]: 5, LABELS[4]: 5})
        self.assertAlmostEqual(s["p_bad"], 0.35)   # OPEN с support=1 отсеян, но не перенормирован
        self.assertEqual(cascade.decide(s, CFG), "review")

    def test_margin_is_status_gap(self):
        # margin — отрыв между статусами, а не между кортежами: дробление P(bad) на два
        # похожих повреждения не должно выглядеть как неуверенность
        s = summary([0.50, 0.30, 0.20, 0.0, 0.0])
        self.assertAlmostEqual(s["p_bad"], 0.80)
        self.assertAlmostEqual(s["p_ok"], 0.20)
        self.assertAlmostEqual(s["margin"], 0.60)
        divided = summary([0.40, 0.38, 0.22, 0.0, 0.0])
        self.assertAlmostEqual(divided["margin"], 0.56)
        self.assertEqual(cascade.decide(divided, CFG), "bad")

    def test_margin_gate_with_lower_status_threshold(self):
        # при сниженном status_conf_min отсекать неоднозначность должен именно margin
        s = summary([0.30, 0.24, 0.46, 0.0, 0.0])
        self.assertAlmostEqual(s["p_bad"], 0.54)
        self.assertAlmostEqual(s["p_ok"], 0.46)
        self.assertAlmostEqual(s["margin"], 0.08)
        loose = {"p_bad_min": 0.50, "p_bad_max_ok": 0.40, "status_conf_min": 0.50, "margin_min": 0.10}
        self.assertEqual(cascade.decide(s, loose), "review")

    def test_single_eligible_margin_is_its_score(self):
        s = summary([0.90, 0.0, 0.10, 0.0, 0.0], support={LABELS[0]: 5, LABELS[2]: 1,
                                                          LABELS[1]: 1, LABELS[3]: 1, LABELS[4]: 1})
        self.assertAlmostEqual(s["margin"], 0.90)
        self.assertEqual(s["eligible"], [0])
        self.assertFalse(s["ready"])  # поддержан только bad -> сам подтвердить не может

    def test_grey_zone_review(self):
        s = summary([0.25, 0.25, 0.50, 0.0, 0.0])
        self.assertAlmostEqual(s["p_bad"], 0.50)
        self.assertEqual(cascade.decide(s, CFG), "review")

    def test_low_confidence_review(self):
        s = summary([0.30, 0.05, 0.20, 0.0, 0.0])
        self.assertLess(s["status_confidence"], CFG["status_conf_min"])
        self.assertEqual(cascade.decide(s, CFG), "review")

    def test_low_support_review(self):
        support = {LABELS[0]: 1, LABELS[1]: 1, LABELS[2]: 5, LABELS[3]: 5, LABELS[4]: 5}
        s = summary([0.45, 0.45, 0.10, 0.0, 0.0], support=support)
        self.assertEqual(s["eligible"], [2])
        self.assertFalse(s["ready"])
        self.assertEqual(cascade.decide(s, CFG), "review")

    def test_old_checkpoint_without_support(self):
        s = summary([0.42, 0.42, 0.16, 0.0, 0.0], support=None)
        self.assertEqual(cascade.decide(s, CFG), "bad")

    def test_missing_status_review(self):
        classes = [BROKEN, OPEN]
        labels = [str(c) for c in classes]
        s = cascade.summarize(classes, [0.9, 0.1], labels=labels, type_hint="manhole")
        self.assertFalse(s["ready"])
        self.assertEqual(cascade.decide(s, CFG), "review")

    def test_pick_best_bad_damage(self):
        s = summary([0.30, 0.55, 0.15, 0.0, 0.0])
        self.assertEqual(cascade.pick(s, CLASSES, "bad"), ("manhole", "cover_open_or_shifted"))
        s = summary([0.10, 0.10, 0.80, 0.0, 0.0])
        self.assertEqual(cascade.pick(s, CLASSES, "ok"), ("manhole", None))
        self.assertEqual(cascade.pick(s, CLASSES, "review"), (None, None))

    def test_bad_policy_rejected(self):
        with self.assertRaises(ValueError):
            cascade.summarize(CLASSES, [0.2] * 5, policy="whatever")

    def test_short_inputs_rejected(self):
        with self.assertRaises(ValueError):
            cascade.summarize(CLASSES, [0.2] * 4)


class DecideDefaultsTest(unittest.TestCase):
    def test_defaults_without_cfg(self):
        s = summary([0.42, 0.42, 0.16, 0.0, 0.0])
        self.assertEqual(cascade.decide(s, None), "bad")
        s = summary([0.30, 0.30, 0.40, 0.0, 0.0])
        self.assertEqual(cascade.decide(s, None), "review")


if __name__ == "__main__":
    unittest.main()
