import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consensus import (  # noqa: E402
    annotate_verdict,
    apply_taxonomy_profile,
    build_display_names,
    build_export_tables,
    compare_predictions,
    summarize_run,
)
from run import load_config  # noqa: E402


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.names = build_display_names(self.cfg)

    def test_taxonomy_profile_does_not_mutate_base_config(self):
        cab_cfg = apply_taxonomy_profile(self.cfg, "cabinets")
        self.assertEqual(set(cab_cfg["taxonomy"]["object_types"]), {"tksh", "rsh"})
        self.assertIn("manhole", self.cfg["taxonomy"]["object_types"])

    def test_dual_model_agreement_on_cabinet_damage(self):
        pred_a = {
            "objects": [{
                "type": "tksh", "bbox": [10, 10, 110, 110], "status": "bad",
                "damage": "mechanical_damage", "description": "Вмятина на дверце ТКШ",
            }],
            "seconds": 1.2, "cost": 0.001, "error": None, "bad_json": False,
        }
        pred_b = {
            "objects": [{
                "type": "tksh", "bbox": [14, 12, 108, 108], "status": "bad",
                "damage": "mechanical_damage", "description": "Деформация передней панели шкафа",
            }],
            "seconds": 1.5, "cost": 0.001, "error": None, "bad_json": False,
        }
        verdict = compare_predictions(pred_a, pred_b, "qwen235", "gemma31", self.names)
        self.assertEqual(verdict["outcome"], "bad")
        self.assertTrue(verdict["agreed"])
        self.assertEqual(verdict["disagreement_reasons"], [])
        self.assertEqual(len(verdict["objects"]), 1)
        self.assertEqual(verdict["objects"][0]["damage"], "mechanical_damage")

    def test_dual_model_disagreement_routes_to_review_with_details(self):
        pred_a = {
            "objects": [{
                "type": "tksh", "bbox": [10, 10, 110, 110], "status": "bad",
                "damage": "coating_corrosion", "description": "Ржавчина на нижней части дверцы",
            }],
            "seconds": 1.0, "cost": 0.0, "error": None, "bad_json": False,
        }
        pred_b = {
            "objects": [{
                "type": "rsh", "bbox": [12, 12, 108, 112], "status": "ok",
                "damage": None, "description": "Корпус чистый, повреждений нет",
            }],
            "seconds": 1.1, "cost": 0.0, "error": None, "bad_json": False,
        }
        verdict = compare_predictions(pred_a, pred_b, "qwen235", "gemma31", self.names)
        self.assertEqual(verdict["outcome"], "review")
        self.assertFalse(verdict["agreed"])
        self.assertIn("status", verdict["disagreement_categories"])
        self.assertIn("type", verdict["disagreement_categories"])
        self.assertEqual(len(verdict["disagreement_reasons"]), 2)

        drawn = annotate_verdict(Image.new("RGB", (140, 140)), verdict, self.names, "qwen235", "gemma31")
        self.assertEqual(drawn.size, (140, 140))

    def test_unmatched_object_and_whole_frame_comparison(self):
        pred_a = {
            "objects": [
                {"type": "tksh", "bbox": [10, 10, 60, 80], "status": "bad", "damage": "mechanical_damage", "description": "Сломан замок"},
                {"type": "rsh", "bbox": [70, 10, 120, 80], "status": "ok", "damage": None, "description": "Щит цел"},
            ],
            "seconds": 1.0,
        }
        pred_b = {
            "objects": [
                {"type": "tksh", "bbox": [12, 12, 58, 78], "status": "bad", "damage": "mechanical_damage", "description": "Сломан замок"},
            ],
            "seconds": 1.0,
        }
        verdict = compare_predictions(pred_a, pred_b, "qwen235", "gemma31", self.names)
        self.assertEqual(verdict["outcome"], "review")
        self.assertIn("detection_count", verdict["disagreement_categories"])

        # Сопоставление whole-frame (bbox=None) с bbox-моделью
        whole = {"objects": [{"type": "tksh", "bbox": None, "status": "bad", "damage": "coating_corrosion", "description": ""}]}
        boxed = {"objects": [{"type": "tksh", "bbox": [5, 5, 50, 50], "status": "bad", "damage": "mechanical_damage", "description": "Вмятина"}]}
        v_whole = compare_predictions(whole, boxed, "photo_dino", "qwen235", self.names)
        self.assertEqual(v_whole["outcome"], "review")
        self.assertIn("damage", v_whole["disagreement_categories"])

    def test_single_model_sub_opinions_disagreement(self):
        pred = {
            "objects": [{"type": "ТКШ", "bbox": None, "status": "bad", "damage": "МП", "description": ""}],
            "sub_opinions": [
                {"model": "DINOv2 LogReg", "label": "ТКШ/МП", "confidence": 0.62},
                {"model": "DINOv2 Fine-tuned", "label": "ТКШ/Удовлетворительное состояние", "confidence": 0.54},
            ],
            "seconds": 0.4,
        }
        verdict = compare_predictions(pred, None, "photo_dino", None, self.names)
        self.assertEqual(verdict["outcome"], "review")
        self.assertIn("Разошлись внутренние модели", verdict["disagreement_reasons"][0])

    def test_summarize_run_and_export_with_overrides(self):
        items = [
            {"key": "a.jpg", "meta": {"ID": "1"}, "gt_label": "ТКШ / МП", "gt_type": "ТКШ", "gt_defect": "МП", "gt_status": "bad"},
            {"key": "b.jpg", "meta": {"ID": "2"}, "gt_label": "РЩ / Норма", "gt_type": "РЩ", "gt_defect": "Удовлетворительное состояние", "gt_status": "ok"},
        ]
        v_a = compare_predictions(
            {"objects": [{"type": "tksh", "bbox": [0, 0, 20, 20], "status": "bad", "damage": "mechanical_damage", "description": "Вмятина"}], "seconds": 1.0},
            {"objects": [{"type": "tksh", "bbox": [1, 1, 19, 19], "status": "bad", "damage": "mechanical_damage", "description": "Повреждена дверь"}], "seconds": 1.0},
            "m1", "m2", self.names,
        )
        v_b = compare_predictions(
            {"objects": [{"type": "rsh", "bbox": [0, 0, 20, 20], "status": "ok", "damage": None, "description": "Норма"}], "seconds": 1.0},
            {"objects": [{"type": "rsh", "bbox": [0, 0, 20, 20], "status": "bad", "damage": "coating_corrosion", "description": "Пятно"}], "seconds": 1.0},
            "m1", "m2", self.names,
        )
        verdicts = {"a.jpg": v_a, "b.jpg": v_b}
        summary = summarize_run(verdicts, items, "folders", self.names, "m1", "m2")
        self.assertEqual(summary["confirmed_bad_photos"], 1)
        self.assertEqual(summary["review_photos"], 1)
        self.assertAlmostEqual(summary["agreement_rate"], 0.5)
        self.assertIsNotNone(summary["gt_metrics"])
        self.assertAlmostEqual(summary["gt_metrics"]["consensus"]["defect_recall_strict"], 1.0)

        tables = build_export_tables(items, verdicts, self.names, overrides={"b.jpg": "ok"})
        self.assertEqual(len(tables["repair"]), 1)
        self.assertEqual(len(tables["review"]), 0)
        self.assertEqual(tables["frames"][1]["outcome"], "ok")


if __name__ == "__main__":
    unittest.main()
