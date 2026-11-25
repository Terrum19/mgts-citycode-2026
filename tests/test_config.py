"""config.toml: ссылки вариантов, пороги каскада, аугментации и совместимость со старым конфигом."""
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import augment  # noqa: E402
import cascade  # noqa: E402
import detector  # noqa: E402

CFG = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
VARIANTS = CFG["variants"]


class ConfigTest(unittest.TestCase):
    def test_required_sections(self):
        for section in ("taxonomy", "names", "detector", "augment", "variants"):
            self.assertIn(section, CFG)

    def test_detector_default_is_proven_backend(self):
        # финальный сценарий не должен по умолчанию зависеть от непроверенных весов
        self.assertEqual(detector.backend_name(CFG["detector"]), "mask2former")
        self.assertIn("model", CFG["detector"])
        self.assertIn("classes", CFG["detector"])

    def test_vlm_references_exist(self):
        for name, variant in VARIANTS.items():
            for key in ("vlm", "qwen"):
                if key in variant and variant[key]:
                    with self.subTest(variant=name, key=key):
                        self.assertIn(variant[key], VARIANTS,
                                      f"{name}.{key}={variant[key]} нет в [variants]")

    def test_photo_dino_has_qwen(self):
        # исходный баг: без ключа qwen VLM в photo_dino молча выключалась
        self.assertIn("qwen", VARIANTS["photo_dino"])
        self.assertTrue(VARIANTS["photo_dino"]["cascade"])

    def test_cascade_thresholds(self):
        for name in ("det_dino", "boxes_dino", "photo_dino"):
            variant = VARIANTS[name]
            with self.subTest(variant=name):
                self.assertTrue(variant["cascade"])
                self.assertLessEqual(variant["p_bad_max_ok"], variant["p_bad_min"])
                self.assertTrue(0 < variant["p_bad_min"] <= 1)
                self.assertTrue(0 <= variant["p_bad_max_ok"] <= 1)
                self.assertTrue(0 < variant["status_conf_min"] <= 1)
                self.assertTrue(0 <= variant["margin_min"] < 1)
                self.assertIn(variant.get("not_target", "review"), ("drop", "review"))

    def test_crop_variants_reference_existing_classes(self):
        self.assertIn(VARIANTS["det_dino"]["vlm"], VARIANTS)
        self.assertIn(VARIANTS["boxes_dino"]["vlm"], VARIANTS)
        self.assertIn(VARIANTS["det_qwen235"].get("context_mode", "crop"), ("crop", "context", "both"))
        self.assertIn("Pole", CFG["detector"]["classes"])

    def test_augment_section_complete(self):
        self.assertEqual(set(CFG["augment"]), set(augment.DEFAULTS))
        for key in ("brightness", "contrast", "saturation", "blur_radius", "noise_std", "jpeg_quality"):
            low, high = CFG["augment"][key]
            with self.subTest(key=key):
                self.assertLessEqual(low, high)

    def test_augment_toggle(self):
        self.assertTrue(VARIANTS["det_dino"]["augment"])
        self.assertLessEqual(1, VARIANTS["det_dino"]["augment_rare_bad_copies"])


class NewSectionsTest(unittest.TestCase):
    def test_selection_cost_quality_app(self):
        for section in ("selection", "cost", "quality_gate", "app"):
            self.assertIn(section, CFG)
        self.assertFalse(CFG["quality_gate"]["enabled"])  # некалиброванный gate выключен
        self.assertTrue(CFG["app"]["offline_map"])
        self.assertGreaterEqual(CFG["cost"]["fn"], CFG["cost"]["fp"])

    def test_offline_and_local_variants_resolve(self):
        import config_utils
        offline = config_utils.resolve_variant(CFG, "det_dino_offline")
        self.assertEqual(offline["vlm"], "")
        self.assertTrue(offline["offline"])
        local = config_utils.resolve_variant(CFG, "photo_dino_local")
        self.assertEqual(local["qwen"], "qwen8b_local")
        self.assertIn("boxes_vlm", {config_utils.resolve_variant(CFG, n).get("kind", "vlm")
                                    for n in VARIANTS})
        self.assertEqual(VARIANTS["qwen8b_local"]["prompt_mode"], "recall")

    def test_photo_dino_hint_is_none(self):
        self.assertEqual(VARIANTS["photo_dino"]["hint_mode"], "none")

    def test_vlm_variant_policies(self):
        self.assertEqual(VARIANTS["det_qwen235"]["not_target"], "drop")
        self.assertEqual(VARIANTS["boxes_qwen235"]["not_target"], "review")
        self.assertEqual(VARIANTS["boxes_dino"]["not_target"], "review")
        self.assertEqual(VARIANTS["gemma26"].get("prompt_mode", "default"), "default")
        self.assertEqual(VARIANTS["qwen235"]["prompt_mode"], "recall")


class LegacyCompatibilityTest(unittest.TestCase):
    """Конфиг без новых ключей обязан вести себя как прежний код."""

    def test_detector_default(self):
        self.assertEqual(detector.backend_name({}), "mask2former")

    def test_augment_off_by_default(self):
        self.assertFalse(VARIANTS.get("photo_dino", {}).get("augment", False))
        self.assertEqual(augment.n_copies("bad", 1, {"checkpoint": "x"}), 0)

    def test_cascade_off_by_default(self):
        # без cascade решение принимает прежняя ветка CropDino по min_confidence
        self.assertFalse({"checkpoint": "x"}.get("cascade", False))

    def test_decide_needs_cfg_only(self):
        summary = {"ready": True, "status_confidence": 0.9, "margin": 0.5, "p_bad": 0.8}
        self.assertEqual(cascade.decide(summary, {}), "bad")


if __name__ == "__main__":
    unittest.main()
