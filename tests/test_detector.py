"""Детектор: выбор бэкенда, отпечаток конфига и геометрия рамок — без torch/transformers.

Сами модели (from_pretrained) локально не запускаются: тест проверяет только чистую часть,
общую для всех бэкендов.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import detector  # noqa: E402

BASE = {"model": "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
        "classes": {"Manhole": "manhole"}, "min_score": 0.5}


class BackendTest(unittest.TestCase):
    def test_default_is_mask2former(self):
        self.assertEqual(detector.backend_name({}), "mask2former")
        self.assertEqual(detector.backend_name(None), "mask2former")

    def test_explicit_backends(self):
        for name in detector.BACKENDS:
            with self.subTest(name=name):
                self.assertEqual(detector.backend_name({"backend": name}), name)

    def test_unknown_backend_fails_loudly(self):
        # молчаливой подмены модели быть не должно: иначе кэш и результат станут невалидными
        with self.assertRaises(ValueError):
            detector.backend_name({"backend": "yolo"})

    def test_fingerprint_is_stable(self):
        first = detector.config_fingerprint(BASE)
        second = detector.config_fingerprint(dict(reversed(list(BASE.items()))))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_fingerprint_changes_with_config(self):
        self.assertNotEqual(detector.config_fingerprint(BASE),
                            detector.config_fingerprint({**BASE, "min_score": 0.3}))
        self.assertNotEqual(detector.config_fingerprint(BASE),
                            detector.config_fingerprint({**BASE, "backend": "grounding_dino"}))


class CropBoxTest(unittest.TestCase):
    def test_pad_and_min_side(self):
        # объект 100x100: pad 0.2 дал бы 140, но min_side=160 важнее — VLM нужен контекст
        self.assertEqual(detector.crop_box([100, 100, 200, 200], 1000, 1000, 0.2, 160),
                         (70, 70, 230, 230))

    def test_clipped_at_edges(self):
        self.assertEqual(detector.crop_box([0, 0, 50, 50], 1000, 1000, 0.0, 100), (0, 0, 75, 75))
        self.assertEqual(detector.crop_box([990, 990, 1000, 1000], 1000, 1000, 0.5, 10),
                         (985, 985, 1000, 1000))

    def test_odd_center_is_rounded(self):
        x1, y1, x2, y2 = detector.crop_box([10, 20, 51, 71], 500, 500, 0.0, 10)
        self.assertTrue(all(isinstance(v, int) for v in (x1, y1, x2, y2)))
        self.assertLessEqual(x2 - x1, 500)


class BoxHelpersTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(detector.box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(detector.box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(detector.box_iou([0, 0, 10, 10], [0, 0, 5, 10]), 0.5)

    def test_box_clips_and_filters(self):
        self.assertEqual(detector._box([-5, -5, 20, 20], 100, 100, 4), [0, 0, 20, 20])
        self.assertIsNone(detector._box([0, 0, 3, 30], 100, 100, 12))   # слишком узкая
        self.assertEqual(detector._box([30, 30, 20, 20], 100, 100, 4), [20, 20, 30, 30])  # порядок сторон

    def test_dedup_keeps_best_score(self):
        found = [{"bbox": [0, 0, 100, 100], "score": 0.6, "type": "manhole", "label": "a"},
                 {"bbox": [2, 2, 98, 98], "score": 0.9, "type": "manhole", "label": "b"},
                 {"bbox": [300, 300, 400, 400], "score": 0.4, "type": "pole", "label": "c"}]
        kept = detector.dedup(found, 0.5)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["score"], 0.9)
        self.assertEqual(kept[0]["label"], "b")


if __name__ == "__main__":
    unittest.main()
