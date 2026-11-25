"""Каскад + VLM: отказ сети/ключа, битый JSON, is_object=false, description-only, damage вне таксономии.

Без сети и без torch: VLM подменяется заглушкой. Проводка CropDino повторена здесь один в один
(модуль crop_dino локально не импортируется: ему нужны joblib/sklearn), а вся decision-логика —
настоящая, из src/cascade.py.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cascade  # noqa: E402

CLASSES = [["manhole", "bad", "cover_broken"],
           ["manhole", "bad", "cover_open_or_shifted"],
           ["manhole", "ok", None]]
LABELS = [json.dumps(c, ensure_ascii=False) for c in CLASSES]
TAX = {"object_types": {"manhole": "люк", "pole": "опора"},
       "damage": {"cover_broken": "сломана", "cover_open_or_shifted": "сдвинута"}}
CFG = {"p_bad_min": 0.60, "p_bad_max_ok": 0.40, "status_conf_min": 0.65, "margin_min": 0.10,
       "describe_bad": True, "not_target": "drop", "min_support": 2}
BAD_SCORES = [0.55, 0.10, 0.35]    # P(bad)=0.65 -> уверенный bad
REVIEW_SCORES = [0.28, 0.28, 0.44]  # P(bad)=0.56 -> серая зона


class StubVLM:
    def __init__(self, answer=None, error=None, bad_json=False, cost=0.0, key="secret", key_env="OPENROUTER_API_KEY",
                 raises=None):
        self.answer, self.error, self.bad_json, self.cost = answer, error, bad_json, cost
        self.key, self.raises, self.calls = key, raises, 0
        self.cfg = {"key_env": key_env}

    def classify_crop(self, crop, type_hint):
        self.calls += 1
        if self.raises:
            raise RuntimeError(self.raises)
        return {"answer": self.answer, "error": self.error, "bad_json": self.bad_json, "cost": self.cost}


def resolve(scores, vlm=None, *, cfg=None, type_hint="manhole", support=None,
            base_type="manhole", telemetry=None):
    """Та же последовательность, что в CropDino._cascade -> объект, изменённые telemetry или None."""
    cfg = CFG if cfg is None else cfg
    telemetry = {"cost": 0.0, "bad_json": False, "error": None} if telemetry is None else telemetry
    summary = cascade.summarize(CLASSES, scores, labels=LABELS, support=support, type_hint=type_hint,
                                policy="detector", min_support=cfg.get("min_support", 2))
    decision = cascade.decide(summary, cfg)
    result = {"type": base_type, "status": decision, "damage": None, "description": ""}
    if decision in ("ok", "bad"):
        kind, damage = cascade.pick(summary, CLASSES, decision)
        result.update(type=kind or base_type, damage=damage if decision == "bad" else None)
    describe = decision == "bad" and cfg.get("describe_bad")
    ready, reason = cascade.vlm_ready(vlm)
    if ready and (decision == "review" or describe):
        call = cascade.classify_crop_safe(vlm, "crop", type_hint)
        telemetry["cost"] += call["cost"]
        telemetry["bad_json"] = telemetry["bad_json"] or call["bad_json"]
        if call["error"] and not telemetry["error"]:
            telemetry["error"] = call["error"]
        answer = call["answer"]
        why = call["error"] or cascade.answer_error(answer)
        if answer is not None and why is None:
            merged = cascade.merge_vlm(result, answer, TAX, not_target=cfg.get("not_target", "review"),
                                       authoritative=decision == "review")
            if merged is None:
                return None
            result = merged
        elif why:
            result["vlm_error"] = why
            if not telemetry["error"]:
                telemetry["error"] = why
    elif decision == "review" and reason:
        result["vlm_error"] = reason
    if result["status"] == "review":
        result["damage"] = None
    return result


class AnswerValidationTest(unittest.TestCase):
    def test_valid(self):
        self.assertIsNone(cascade.answer_error({"status": "bad", "type": "manhole"}))

    def test_invalid_status(self):
        # clean_state превратил бы «maybe» в ok — для каскада это опаснее честного review
        self.assertIn("status", cascade.answer_error({"status": "maybe"}))

    def test_not_object_is_not_error(self):
        self.assertIsNone(cascade.answer_error({"is_object": False}))

    def test_not_dict(self):
        self.assertIsNotNone(cascade.answer_error(None))

    def test_unknown_damage_becomes_unknown(self):
        status, damage = cascade.clean_state({"status": "bad", "damage": "проржавело"}, TAX)
        self.assertEqual((status, damage), ("bad", "unknown"))

    def test_ok_clears_damage(self):
        self.assertEqual(cascade.clean_state({"status": "ok", "damage": "cover_broken"}, TAX), ("ok", None))


class DegradationTest(unittest.TestCase):
    def test_no_key_keeps_review_without_api_error(self):
        vlm = StubVLM(key="", key_env="OPENROUTER_API_KEY")
        result = resolve(REVIEW_SCORES, vlm)
        self.assertEqual(result["status"], "review")
        self.assertEqual(vlm.calls, 0)                      # запрос не делали
        self.assertIn("ключ", result["vlm_error"])           # причина видна в объекте

    def test_no_vlm_configured(self):
        result = resolve(REVIEW_SCORES, None)
        self.assertEqual(result["status"], "review")
        self.assertNotIn("vlm_error", result)

    def test_network_error_keeps_review(self):
        telemetry = {"cost": 0.0, "bad_json": False, "error": None}
        vlm = StubVLM(raises="connection reset")
        result = resolve(REVIEW_SCORES, vlm, telemetry=telemetry)
        self.assertEqual(result["status"], "review")
        self.assertIn("connection reset", result["vlm_error"])
        self.assertIn("connection reset", telemetry["error"])

    def test_bad_json_recorded(self):
        telemetry = {"cost": 0.0, "bad_json": False, "error": None}
        result = resolve(REVIEW_SCORES, StubVLM(answer=None, bad_json=True), telemetry=telemetry)
        self.assertEqual(result["status"], "review")
        self.assertTrue(telemetry["bad_json"])
        self.assertTrue(telemetry["error"])

    def test_invalid_status_is_error_not_bad_json(self):
        telemetry = {"cost": 0.0, "bad_json": False, "error": None}
        result = resolve(REVIEW_SCORES, StubVLM(answer={"status": "maybe"}), telemetry=telemetry)
        self.assertEqual(result["status"], "review")
        self.assertFalse(telemetry["bad_json"])
        self.assertIn("status", telemetry["error"])

    def test_cost_accumulates(self):
        telemetry = {"cost": 0.0, "bad_json": False, "error": None}
        resolve(REVIEW_SCORES, StubVLM(answer={"status": "bad", "damage": "cover_broken"}, cost=0.02),
                telemetry=telemetry)
        self.assertAlmostEqual(telemetry["cost"], 0.02)


class ResolveTest(unittest.TestCase):
    def test_review_resolved_by_vlm(self):
        vlm = StubVLM(answer={"is_object": True, "type": "manhole", "status": "bad",
                              "damage": "cover_broken", "description": "крышка сломана"})
        result = resolve(REVIEW_SCORES, vlm)
        self.assertEqual(result["status"], "bad")
        self.assertEqual(result["damage"], "cover_broken")
        self.assertEqual(result["description"], "крышка сломана")
        self.assertEqual(result["type"], "manhole")

    def test_review_resolved_to_ok(self):
        vlm = StubVLM(answer={"is_object": True, "type": "manhole", "status": "ok", "description": "всё цело"})
        result = resolve(REVIEW_SCORES, vlm)
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["damage"])

    def test_confident_ok_does_not_call_vlm(self):
        vlm = StubVLM(answer={"status": "bad", "damage": "cover_broken"})
        result = resolve([0.05, 0.05, 0.90], vlm)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(vlm.calls, 0)
        self.assertEqual(result["description"], "")

    def test_describe_bad_success(self):
        vlm = StubVLM(answer={"is_object": True, "status": "bad", "damage": "cover_broken",
                              "description": "крышка с дырой"})
        result = resolve(BAD_SCORES, vlm)
        self.assertEqual(result["status"], "bad")
        self.assertEqual(result["description"], "крышка с дырой")
        self.assertEqual(vlm.calls, 1)

    def test_describe_bad_failure_keeps_bad(self):
        # необязательное описание не должно превращать подтверждённый дефект в review
        result = resolve(BAD_SCORES, StubVLM(raises="timeout"))
        self.assertEqual(result["status"], "bad")
        self.assertEqual(result["description"], "")

    def test_describe_bad_can_be_disabled(self):
        vlm = StubVLM(answer={"status": "bad", "damage": "cover_broken", "description": "x"})
        result = resolve(BAD_SCORES, vlm, cfg={**CFG, "describe_bad": False})
        self.assertEqual(vlm.calls, 0)
        self.assertEqual(result["description"], "")

    def test_not_object_dropped_for_detections(self):
        vlm = StubVLM(answer={"is_object": False, "description": "ливневая решётка"})
        self.assertIsNone(resolve(REVIEW_SCORES, vlm, cfg={**CFG, "not_target": "drop"}))

    def test_not_object_kept_as_review_for_organizer_boxes(self):
        vlm = StubVLM(answer={"is_object": False, "description": "ливневая решётка"})
        result = resolve(REVIEW_SCORES, vlm, cfg={**CFG, "not_target": "review"})
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["damage"])

    def test_vlm_does_not_override_known_type(self):
        vlm = StubVLM(answer={"is_object": True, "type": "pole", "status": "bad", "damage": "cover_broken"})
        result = resolve(REVIEW_SCORES, vlm, base_type="manhole")
        self.assertEqual(result["type"], "manhole")

    def test_unknown_damage_kept_as_unknown(self):
        vlm = StubVLM(answer={"is_object": True, "status": "bad", "damage": "вмятина", "description": "x"})
        result = resolve(REVIEW_SCORES, vlm)
        self.assertEqual(result["status"], "bad")
        self.assertEqual(result["damage"], "unknown")

    def test_low_support_goes_to_vlm(self):
        support = {LABELS[0]: 1, LABELS[1]: 1, LABELS[2]: 5}
        vlm = StubVLM(answer={"is_object": True, "status": "bad", "damage": "cover_broken", "description": "x"})
        result = resolve(BAD_SCORES, vlm, support=support)
        self.assertEqual(vlm.calls, 1)
        self.assertEqual(result["status"], "bad")


if __name__ == "__main__":
    unittest.main()
