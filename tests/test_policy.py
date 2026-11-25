"""Общая policy: legacy/cascade, VLM description-only, review и checkpoint-конфликты."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cascade  # noqa: E402
import policy  # noqa: E402

TAX = {"object_types": {"manhole": "люк", "pole": "опора"},
       "damage": {"cover_broken": "сломана", "pole_tilted": "наклон"}}
CLASSES = [["manhole", "bad", "cover_broken"], ["manhole", "ok", None],
           ["pole", "ok", None], ["pole", "bad", "pole_tilted"]]
LABELS = [json.dumps(c, ensure_ascii=False) for c in CLASSES]
CFG = {"cascade": True, "p_bad_min": 0.60, "p_bad_max_ok": 0.40,
       "status_conf_min": 0.65, "margin_min": 0.10, "describe_bad": True,
       "not_target": "review", "min_support": 2}


def init(scores, mode="cascade", cfg=None, support=None, type_hint="manhole"):
    return policy.initial(LABELS, CLASSES, scores, mode=mode, cfg=cfg or CFG,
                          support=support, type_hint=type_hint, class_policy="detector")


class InitialTest(unittest.TestCase):
    def test_cascade_bad(self):
        d = init([0.80, 0.10, 0.0, 0.10])
        self.assertEqual(d["status"], "bad")
        self.assertEqual(d["type"], "manhole")
        self.assertTrue(d["needs_vlm"])  # describe_bad
        self.assertEqual(d["label"], LABELS[0])

    def test_cascade_review_reasons(self):
        d = init([0.35, 0.35, 0.15, 0.15], cfg={**CFG, "status_conf_min": 0.30})
        self.assertEqual(d["status"], "review")
        self.assertIn("low_margin", d["review_reasons"])

    def test_missing_support_review(self):
        support = {LABELS[0]: 1, LABELS[1]: 1, LABELS[2]: 5, LABELS[3]: 5}
        d = init([0.9, 0.05, 0.03, 0.02], support=support)
        self.assertEqual(d["status"], "review")
        self.assertIn("low_support", d["review_reasons"])

    def test_legacy_argmax_and_confidence(self):
        d = init([0.20, 0.70, 0.05, 0.05], mode="legacy", cfg={"min_confidence": 0.0})
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["label"], LABELS[1])
        d = init([0.20, 0.70, 0.05, 0.05], mode="legacy", cfg={"min_confidence": 0.9})
        self.assertEqual(d["status"], "review")
        self.assertIn("low_confidence", d["review_reasons"])

    def test_legacy_vlm_changes_class(self):
        d = init([0.40, 0.35, 0.15, 0.10], mode="legacy",
                 cfg={"min_confidence": 0.0, "qwen_weight": 0.5})
        self.assertTrue(d["needs_vlm"])
        out = policy.finalize(d, cfg={"qwen_weight": 0.5, "min_confidence": 0.0}, tax=TAX,
                              vlm={"label": LABELS[0], "available": True})
        self.assertEqual(out["status"], "bad")
        self.assertEqual(out["label"], LABELS[0])


class FinalizeTest(unittest.TestCase):
    def test_description_only_never_drops_bad(self):
        d = init([0.80, 0.10, 0.0, 0.10])
        out = policy.finalize(d, cfg={**CFG, "describe_bad": True}, tax=TAX,
                              vlm={"answer": {"is_object": False, "description": "не наш объект"},
                                   "available": True})
        self.assertEqual(out["status"], "bad")
        self.assertEqual(out["description"], "не наш объект")

    def test_not_object_review_keeps_finding(self):
        d = init([0.35, 0.35, 0.15, 0.15])
        out = policy.finalize(d, cfg={**CFG, "not_target": "review"}, tax=TAX,
                              vlm={"answer": {"is_object": False, "description": "решётка"},
                                   "available": True})
        self.assertEqual(out["status"], "review")
        self.assertEqual(out["type"], "manhole")
        self.assertIn("not_target", out["review_reasons"])

    def test_not_object_drop_is_only_explicit(self):
        d = init([0.35, 0.35, 0.15, 0.15])
        out = policy.finalize(d, cfg={**CFG, "not_target": "drop"}, tax=TAX,
                              vlm={"answer": {"is_object": False}, "available": True})
        self.assertIsNone(out)

    def test_unknown_status_is_error_not_ok(self):
        d = init([0.35, 0.35, 0.15, 0.15])
        out = policy.finalize(d, cfg=CFG, tax=TAX,
                              vlm={"answer": {"status": "maybe"}, "available": True})
        self.assertEqual(out["status"], "review")
        self.assertIn("vlm_error", out["review_reasons"])

    def test_explicit_review_answer(self):
        d = init([0.35, 0.35, 0.15, 0.15])
        out = policy.finalize(d, cfg=CFG, tax=TAX,
                              vlm={"answer": {"status": "review", "damage": "cover_broken"}, "available": True})
        self.assertEqual(out["status"], "review")
        self.assertIsNone(out["damage"])
        self.assertIn("low_confidence", out["review_reasons"])

    def test_vlm_resolves_review_to_bad(self):
        d = init([0.35, 0.35, 0.15, 0.15])
        out = policy.finalize(d, cfg=CFG, tax=TAX,
                              vlm={"answer": {"status": "bad", "damage": "cover_broken",
                                              "description": "крышка сломана"}, "available": True})
        self.assertEqual(out["status"], "bad")
        self.assertEqual(out["damage"], "cover_broken")
        self.assertEqual(out["description"], "крышка сломана")


class EffectivePolicyTest(unittest.TestCase):
    def test_legacy_checkpoint_uses_runtime(self):
        self.assertEqual(policy.effective_policy(None, CFG), CFG)

    def test_conflict_is_error(self):
        saved = {"p_bad_min": 0.7}
        with self.assertRaises(ValueError):
            policy.effective_policy(saved, {"p_bad_min": 0.6})

    def test_explicit_override(self):
        saved = {"p_bad_min": 0.7}
        out = policy.effective_policy(saved, {"p_bad_min": 0.6, "override_checkpoint_policy": True})
        self.assertEqual(out["p_bad_min"], 0.7)

    def test_deployment_keys_are_not_semantic(self):
        out = policy.effective_policy({"p_bad_min": 0.7}, {"device": "cpu", "p_bad_min": 0.7})
        self.assertEqual(out["device"], "cpu")

    def test_reasons(self):
        self.assertEqual(policy.decision_reasons({"ready": False, "eligible": []}, CFG), ["low_support"])
        self.assertEqual(policy.decision_reasons({"ready": False, "eligible": [0]}, CFG), ["missing_status"])
        self.assertEqual(policy.decision_reasons({"ready": True, "status_confidence": 0.5, "margin": 0.5,
                                                  "p_bad": 0.5}, CFG), ["low_status_confidence"])


class CascadeCompatTest(unittest.TestCase):
    def test_statuses_remains_training_classes(self):
        self.assertEqual(cascade.STATUSES, ("ok", "bad"))
        self.assertIn("review", cascade.ANSWER_STATUSES)

    def test_clean_state_unknown_is_review(self):
        self.assertEqual(cascade.clean_state({"status": "maybe"}, TAX), ("review", None))
        self.assertEqual(cascade.clean_state({"status": "review", "damage": "cover_broken"}, TAX),
                         ("review", None))


if __name__ == "__main__":
    unittest.main()
