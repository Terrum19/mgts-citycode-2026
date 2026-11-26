import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grounding_qwen
from predict import run
from run import load_config
from variants import make


class GroundingQwenTests(unittest.TestCase):
    def test_every_proposal_is_sent_and_audited(self):
        cfg = load_config()
        boxes = [
            {"type": "rsh", "bbox": [12, 23, 54, 65], "score": 0.80},
            {"type": "tksh", "bbox": [58, 45, 105, 102], "score": 0.73},
            {"type": "rsh", "bbox": [110, 45, 140, 96], "score": 0.60},
            {"type": "tksh", "bbox": [31, 69, 100, 106], "score": 0.51},
        ]

        def response(raw, error=None):
            return {"value": grounding_qwen.parse_inspection(raw), "error": error,
                    "bad_json": False, "cost": 0.01}

        answers = [
            response('{"type":"tksh","verdict":"paint","evidence":"Ржавчина на двери"}'),
            response("", "HTTP error"),
            response('{"type":"not_target","verdict":"not_target","evidence":"Опора"}'),
            response('{"type":"rsh","verdict":"ok","evidence":"Целая дверь"}'),
        ]
        with patch.object(grounding_qwen, "GroundingDetector") as detector, patch.object(grounding_qwen, "VLM") as vlm:
            detector.return_value.detect.return_value = boxes
            client = vlm.return_value
            client.key = "test-key"
            client.max_side = 1600
            client._ask.side_effect = answers
            model = make("det_grounding_qwen235", cfg)
            with tempfile.TemporaryDirectory() as folder:
                image = Path(folder) / "cabinet.jpg"
                Image.new("RGB", (160, 110)).save(image)
                rows, predictions = run(model, [image], {}, cfg["names"], Path(folder) / "out")
            pred = predictions[image.name]

        self.assertEqual(client._ask.call_count, len(boxes))
        self.assertEqual(pred["detections"], boxes)
        self.assertEqual([r["proposal_id"] for r in pred["crop_results"]], [1, 2, 3, 4])
        self.assertEqual([o["proposal_id"] for o in pred["objects"]], [1, 2, 4])
        self.assertEqual([r["status"] for r in rows], ["bad", "review", "ok"])
        self.assertEqual(rows[0]["type"], "tksh")
        self.assertEqual(rows[0]["damage"], "coating_corrosion")
        self.assertEqual(pred["crop_results"][1]["error"], "HTTP error")
        self.assertEqual(pred["crop_results"][2]["answer"]["verdict"], "not_target")
        self.assertEqual(json.loads(json.dumps(pred))["detections"], boxes)

    def test_proposals_clip_and_deduplicate_overlapping_types(self):
        raw = [{"type": "rsh", "bbox": [-3, 5, 50, 50], "score": .8},
               {"type": "tksh", "bbox": [0, 5, 48, 52], "score": .7},
               {"type": "tksh", "bbox": [65, 55, 90, 85], "score": .6}]
        kept = grounding_qwen.detector_proposals(raw, 100, 90, 20, .5)
        self.assertEqual([row["bbox"] for row in kept], [[0, 5, 50, 50], [65, 55, 90, 85]])

    def test_uncertain_type_requires_review(self):
        self.assertIsNone(grounding_qwen.parse_inspection('{"type":null,"verdict":"paint","evidence":"Ржавчина"}'))
        answer = grounding_qwen.parse_inspection('{"type":null,"verdict":"review","evidence":"Корпус обрезан"}')
        self.assertEqual(answer["verdict"], "review")
