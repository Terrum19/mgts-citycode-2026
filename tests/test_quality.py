"""Quality Gate: признаки, выключенный режим и причины review."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import quality  # noqa: E402


def flat(value, n=64):
    return [[value] * n for _ in range(n)]


class QualityTest(unittest.TestCase):
    def test_constant_frame_has_no_contrast(self):
        m = quality.quality_metrics(flat(100))
        self.assertEqual(m["std"], 0.0)
        self.assertEqual(m["sharpness"], 0.0)

    def test_disabled_passes(self):
        q = quality.assess_quality(flat(100), {"enabled": False, "min_std": 18})
        self.assertTrue(q["passed"])
        self.assertEqual(q["review_reasons"], [])

    def test_dark_frame_fails(self):
        q = quality.assess_quality(flat(5), {"enabled": True, "min_mean": 25, "min_std": 1,
                                             "min_sharpness": 0, "max_overexposed": 1})
        self.assertFalse(q["passed"])
        self.assertIn("quality", q["review_reasons"])

    def test_overexposed_fails(self):
        q = quality.assess_quality(flat(255), {"enabled": True, "min_mean": 0, "max_mean": 255,
                                               "min_std": 0, "min_sharpness": 0, "max_overexposed": 0.1})
        self.assertFalse(q["passed"])

    def test_textured_frame_passes(self):
        rows = [[(x * 17 + y * 31) % 256 for x in range(64)] for y in range(64)]
        q = quality.assess_quality(rows, {"enabled": True, "min_mean": 10, "max_mean": 250,
                                          "min_std": 10, "min_sharpness": 10, "max_overexposed": 0.9})
        self.assertTrue(q["passed"])
        self.assertGreater(q["metrics"]["sharpness"], 0)


if __name__ == "__main__":
    unittest.main()
