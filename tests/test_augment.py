"""Погодные аугментации: свойства выхода, детерминизм, отсутствие геометрии, число копий."""
import re
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import augment  # noqa: E402

OFF = {"brightness": [1, 1], "contrast": [1, 1], "saturation": [1, 1], "color_gain": 0,
       "blur_prob": 0, "motion_prob": 0, "noise_prob": 0, "jpeg_prob": 0}
COLORFUL = {"brightness": [1.4, 1.4], "contrast": [1.3, 1.3], "saturation": [1.4, 1.4],
            "color_gain": 0.2, "blur_prob": 1, "motion_prob": 0, "noise_prob": 0, "jpeg_prob": 0}
NOISY = {"brightness": [1, 1], "contrast": [1, 1], "saturation": [1, 1], "color_gain": 0,
         "blur_prob": 0, "motion_prob": 1, "noise_prob": 1, "jpeg_prob": 1}


def photo(size=(96, 64)):
    """Градиент + пятна: на плоском фоне разница аугментаций не видна."""
    x = np.linspace(0, 255, size[0], dtype=np.uint8)
    array = np.tile(x, (size[1], 1))
    array = np.dstack([array, array // 2, 255 - array]).astype(np.uint8)
    return Image.fromarray(array, "RGB")


class AugmentTest(unittest.TestCase):
    def test_shape_and_mode(self):
        out = augment.augment(photo(), COLORFUL, 1)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, photo().size)
        self.assertEqual(np.asarray(out).dtype, np.uint8)

    def test_pixels_in_range(self):
        array = np.asarray(augment.augment(photo(), NOISY, 7))
        self.assertGreaterEqual(int(array.min()), 0)
        self.assertLessEqual(int(array.max()), 255)

    def test_deterministic(self):
        first = np.asarray(augment.augment(photo(), NOISY, 42))
        second = np.asarray(augment.augment(photo(), NOISY, 42))
        np.testing.assert_array_equal(first, second)

    def test_seed_changes_result(self):
        first = np.asarray(augment.augment(photo(), NOISY, 1))
        second = np.asarray(augment.augment(photo(), NOISY, 2))
        self.assertFalse(np.array_equal(first, second))

    def test_identity_when_disabled(self):
        # все операции выключены -> обучение не должно получать искажённых копий «на всякий случай»
        np.testing.assert_array_equal(np.asarray(augment.augment(photo(), OFF, 5)),
                                      np.asarray(photo().convert("RGB")))

    def test_motion_blur_smears_horizontally(self):
        # ровный вертикальный перепад + горизонтальный смаз -> появляются промежуточные значения
        half = Image.new("RGB", (64, 64), (0, 0, 0))
        half.paste((255, 255, 255), (32, 0, 64, 64))
        blurred = augment.motion_blur(half, 7, 0)
        row = np.asarray(blurred)[32, :, 0].astype(int)
        self.assertGreater(len(set(row.tolist())), 2)

    def test_settings_defaults(self):
        self.assertEqual(augment.settings(None)["blur_prob"], augment.DEFAULTS["blur_prob"])
        self.assertEqual(augment.settings({"brightness": [0.5, 0.6], "unknown": 1})["brightness"],
                         [0.5, 0.6])

    def test_no_geometric_operations(self):
        # повороты и отражения запрещены: они меняют физический смысл наклона опоры
        source = (ROOT / "src" / "augment.py").read_text(encoding="utf-8")
        for token in ("ImageOps", "rotate(", "transpose(", "FLIP_", "MIRROR", "resize("):
            with self.subTest(token=token):
                self.assertNotIn(token, source)


class CopiesTest(unittest.TestCase):
    CFG = {"augment": True, "augment_copies": 1, "augment_rare_bad_copies": 3, "augment_rare_threshold": 20}

    def test_disabled(self):
        self.assertEqual(augment.n_copies("bad", 3, None), 0)
        self.assertEqual(augment.n_copies("bad", 3, {"augment": False}), 0)

    def test_rare_bad_gets_more(self):
        self.assertEqual(augment.n_copies("bad", 3, self.CFG), 3)
        self.assertEqual(augment.n_copies("bad", 100, self.CFG), 1)

    def test_ok_is_not_boosted(self):
        self.assertEqual(augment.n_copies("ok", 3, self.CFG), 1)

    def test_defaults(self):
        self.assertEqual(augment.n_copies("bad", 1, {"augment": True}), 1)
        cfg = {"augment": True, "augment_copies": 2}
        self.assertEqual(augment.n_copies("bad", 1, cfg), 2)

    def test_seed_is_stable(self):
        self.assertEqual(augment.sample_seed(42, "a.jpg:1", 2), augment.sample_seed(42, "a.jpg:1", 2))
        self.assertNotEqual(augment.sample_seed(42, "a.jpg:1", 2), augment.sample_seed(42, "a.jpg:1", 3))


if __name__ == "__main__":
    unittest.main()
