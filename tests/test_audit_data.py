"""Аудит данных: битые метки, утечки split и загрязнённые test-прогоны."""
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_data  # noqa: E402

TAX = {"object_types": {"manhole": "люк"}, "damage": {"cover_broken": "сломана"}}


def object(box=(1, 1, 10, 10), status="ok", typ="manhole", damage=None):
    return {"type": typ, "bbox": list(box), "status": status, "damage": damage}


def label(name, objects=None, split="dev", flags=(), location=None):
    return {"file": name, "width": 32, "height": 32, "split": split,
            "frame_flags": list(flags), "location_id": location or name,
            "objects": list(objects or [])}


class AuditTest(unittest.TestCase):
    def _images(self, root, names):
        for name in names:
            Image.new("RGB", (32, 32), (10, 10, 10)).save(root / name)

    def test_valid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._images(root, ["a.jpg"])
            report = audit_data.audit_labels([label("a.jpg", [object()])], taxonomy=TAX, images=root)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["counts"]["frames"], 1)
            self.assertEqual(report["counts"]["objects"], 1)

    def test_invalid_bbox_and_unknown_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._images(root, ["a.jpg"])
            labels = [label("a.jpg", [object(box=(0, 0, 0, 1)), object(status="maybe")])]
            report = audit_data.audit_labels(labels, taxonomy=TAX, images=root)
            self.assertTrue(any("некорректная рамка" in e for e in report["errors"]))
            self.assertTrue(any("status" in e for e in report["errors"]))

    def test_target_not_visible_with_objects(self):
        report = audit_data.audit_labels([label("a.jpg", [object()], flags=["target_not_visible"])],
                                         taxonomy=TAX)
        self.assertTrue(any("target_not_visible" in e for e in report["errors"]))
        report = audit_data.audit_labels([label("a.jpg", [object()], flags=["bad_frame"])], taxonomy=TAX)
        self.assertTrue(report["warnings"])

    def test_location_leak(self):
        labels = [label("a.jpg", split="dev", location="L"), label("b.jpg", split="test", location="L")]
        report = audit_data.audit_labels(labels, taxonomy=TAX)
        self.assertTrue(any("location_id" in e for e in report["errors"]))

    def test_duplicate_image_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (32, 32), (10, 10, 10)).save(root / "a.jpg")
            Image.new("RGB", (32, 32), (10, 10, 10)).save(root / "b.jpg")
            report = audit_data.audit_labels([label("a.jpg"), label("b.jpg")], taxonomy=TAX, images=root)
            self.assertTrue(any("SHA-256" in e for e in report["errors"]))

    def test_contaminated_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            (runs / "pano_test__qwen235").mkdir(parents=True)
            protocol = {"datasets": {"pano_test": {"role": "contaminated", "selection_eligible": False}}}
            report = audit_data.audit_labels([label("a.jpg")], taxonomy=TAX, runs=runs, protocol=protocol)
            self.assertIn("pano_test__qwen235", report["contaminated_runs"])
            self.assertTrue(report["warnings"])


if __name__ == "__main__":
    unittest.main()
