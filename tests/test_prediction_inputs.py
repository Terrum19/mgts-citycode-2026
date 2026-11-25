import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from draw import annotate
from predict import read_boxes, run


class PredictionInputsTests(unittest.TestCase):
    def test_reuse_original_detections_including_empty_frames(self):
        found = {"bbox": [1, 2, 5, 6], "type": "pole", "score": 0.7}
        predictions = {"one.jpg": {"detections": [found], "objects": []},
                       "two.jpg": {"detections": [], "objects": []}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "boxes.json"
            path.write_text(json.dumps(predictions), encoding="utf-8")
            replay = read_boxes(path)
            self.assertEqual(replay, {"one.jpg": [found], "two.jpg": []})
            path.write_text(json.dumps([{"file": name, "objects": boxes}
                                        for name, boxes in replay.items()]), encoding="utf-8")
            self.assertEqual(read_boxes(path), replay)
            path.write_text('{"one.jpg": {"objects": []}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_boxes(path)

    def test_rgb_grayscale_and_alpha_reach_model_as_rgb(self):
        class Reader:
            def predict(self, image):
                return {"objects": [], "mode": image.mode, "pixels": image.tobytes()}

        with tempfile.TemporaryDirectory() as folder:
            for mode in ("RGB", "L", "RGBA", "P"):
                with self.subTest(mode=mode):
                    path = Path(folder) / "frame.png"
                    source = Image.new(mode, (8, 8), 1)
                    source.save(path)
                    with Image.open(path) as original:
                        expected = original.convert("RGB").tobytes()
                    _, predictions = run(Reader(), [path], {}, {}, workers=1)
                    self.assertEqual(predictions[path.name]["mode"], "RGB")
                    self.assertEqual(predictions[path.name]["pixels"], expected)
                    painted = annotate(source, [])
                    painted.putpixel((0, 0), (30, 40, 50))
                    self.assertNotEqual(source.convert("RGB").getpixel((0, 0)), (30, 40, 50))
