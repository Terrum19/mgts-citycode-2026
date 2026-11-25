"""VGCopy-paste: детерминизм, валидный manifest и bbox в границах кадра.

Синтетические донор/маска/фоны — без реальных данных. Сгенерированное остаётся во временном
каталоге и никогда не попадает в train автоматически.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "poc"))

import copy_paste  # noqa: E402


def make_donor(path: Path, size=(40, 40)) -> Image.Image:
    image = Image.new("RGB", size, (90, 90, 90))
    center = (size[0] // 2, size[1] // 2)
    for y in range(size[1]):
        for x in range(size[0]):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < 100:
                image.putpixel((x, y), (200, 40, 40))
    return image


def make_mask(path: Path, size=(40, 40)) -> None:
    mask = Image.new("L", size, 0)
    for y in range(10, 30):
        for x in range(10, 30):
            mask.putpixel((x, y), 255)
    mask.save(path)


class CopyPasteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "donors").mkdir()
        (self.root / "backgrounds").mkdir()
        make_donor(self.root / "donors" / "broken.jpg").save(self.root / "donors" / "broken.jpg")
        make_mask(self.root / "donors" / "broken_mask.png")
        for k, color in enumerate(((120, 130, 140), (60, 70, 80))):
            Image.new("RGB", (240, 160), color).save(self.root / "backgrounds" / f"bg{k}.jpg")
        self.manifest = self.root / "defects.json"
        self.manifest.write_text(json.dumps([{
            "file": "donors/broken.jpg", "mask": "donors/broken_mask.png",
            "type": "manhole", "status": "bad", "damage": "cover_broken"}]))
        self.defects = copy_paste.read_defects(self.manifest)
        self.backgrounds = copy_paste.background_list(self.root / "backgrounds")

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_and_files(self):
        out = self.root / "out"
        rows = copy_paste.generate(self.defects, self.backgrounds, out, 3, seed=7)
        self.assertEqual(len(rows), 3)
        labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
        self.assertEqual(labels, rows)
        for row in rows:
            with Image.open(out / row["file"]) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (240, 160))
            self.assertEqual(row["objects"][0]["type"], "manhole")
            self.assertEqual(row["objects"][0]["status"], "bad")
            self.assertEqual(row["objects"][0]["damage"], "cover_broken")

    def test_bbox_inside_frame(self):
        rows = copy_paste.generate(self.defects, self.backgrounds, self.root / "out", 5, seed=3)
        for row in rows:
            x1, y1, x2, y2 = row["objects"][0]["bbox"]
            self.assertTrue(0 <= x1 < x2 <= 240, row)
            self.assertTrue(0 <= y1 < y2 <= 160, row)

    def test_deterministic(self):
        first = copy_paste.generate(self.defects, self.backgrounds, self.root / "a", 2, seed=11)
        second = copy_paste.generate(self.defects, self.backgrounds, self.root / "b", 2, seed=11)
        self.assertEqual([r["objects"][0]["bbox"] for r in first],
                         [r["objects"][0]["bbox"] for r in second])
        self.assertEqual((self.root / "a" / first[0]["file"]).read_bytes(),
                         (self.root / "b" / second[0]["file"]).read_bytes())

    def test_seed_changes_layout(self):
        first = copy_paste.generate(self.defects, self.backgrounds, self.root / "a", 2, seed=1)
        second = copy_paste.generate(self.defects, self.backgrounds, self.root / "b", 2, seed=2)
        self.assertNotEqual([r["objects"][0]["bbox"] for r in first],
                            [r["objects"][0]["bbox"] for r in second])

    def test_mask_size_mismatch(self):
        bad_mask = self.root / "donors" / "small.png"
        Image.new("L", (10, 10), 255).save(bad_mask)
        with self.assertRaises(ValueError):
            copy_paste.cut(self.root / "donors" / "broken.jpg", bad_mask)

    def test_empty_mask(self):
        empty = self.root / "donors" / "empty.png"
        Image.new("L", (40, 40), 0).save(empty)
        with self.assertRaises(ValueError):
            copy_paste.cut(self.root / "donors" / "broken.jpg", empty)

    def test_feathering_softens_alpha(self):
        piece = copy_paste.cut(self.defects[0]["donor"], self.defects[0]["mask"])
        soft = copy_paste.feather(piece)
        hard_alpha = np.asarray(piece.getchannel("A"))
        soft_alpha = np.asarray(soft.getchannel("A"))
        self.assertTrue((soft_alpha < 255).any(), "у краёв должна быть полупрозрачность")
        self.assertGreaterEqual(int(soft_alpha.max()), int(hard_alpha.max()) - 1)


if __name__ == "__main__":
    unittest.main()
