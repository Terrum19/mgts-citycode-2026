import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consensus import compare_predictions
from frame_crop_qwen import parse_frame
from run import load_config
from variants import make


class FrameCropQwenTests(unittest.TestCase):
    def simulate(self, frame, crops, scores=None):
        config = load_config()
        config["variants"]["frame_crop_qwen235"]["encoder"] = "missing-test-encoder"
        scores = scores or [0.9 - index / 10 for index in range(len(crops))]
        boxes = [{"type": "rsh", "bbox": [index*10, 10, index*10+10, 25], "score": score}
                 for index, score in enumerate(scores)]
        positions = iter(crops)
        prompts = []

        def reply(jpeg, prompt, parse):
            prompts.append(prompt)
            item = frame if parse is parse_frame else next(positions)
            if item is None:
                return {"value": None, "error": "API недоступен", "cost": 0, "bad_json": False}
            return {"value": parse(json.dumps(item)), "error": None, "cost": .001, "bad_json": False}

        with patch("grounding_qwen.GroundingDetector") as detector, patch("grounding_qwen.VLM") as vlm:
            detector.return_value.detect.return_value = boxes
            client = vlm.return_value
            client.key, client.max_side = "test", 1600
            client._ask.side_effect = reply
            model = make("frame_crop_qwen235", config)
            pred = model.predict(Image.new("RGB", (80, 40)))
        self.assertEqual(len(prompts), len(crops) + 1)
        self.assertIn(model.frame_prompt, prompts)
        self.assertFalse(pred["dinov2"]["available"])
        return pred

    def test_primary_score_controls_verdict_and_secondary_failure_keeps_agreement(self):
        frame = {"class": 3, "description": "Коррозия щита"}
        crops = [{"type": "rsh", "verdict": "ok", "evidence": "Целый"},
                 {"type": None, "verdict": "not_target", "evidence": "Фон"},
                 {"type": "rsh", "verdict": "paint", "evidence": "Ржавчина"}, None]
        pred = self.simulate(frame, crops, [.5, .99, .96, .2])
        self.assertEqual(pred["frame_class"], pred["crop_class"])
        self.assertEqual(pred["primary_proposal_id"], 3)
        self.assertEqual(pred["objects"][0]["status"], "bad")
        self.assertEqual(pred["objects"][0]["damage"], "coating_corrosion")
        self.assertEqual(pred["crop_results"][3]["error"], "API недоступен")
        self.assertEqual(compare_predictions(pred, None, "test", None, {})["objects"][0]["status"], "bad")
        self.assertAlmostEqual(pred["cost"], .004)

    def test_disagreement_and_missing_target_go_to_review(self):
        examples = [
            ({"class": 1, "description": "РЩ"}, [{"type": "tksh", "verdict": "mechanical", "evidence": "Вмятина"}]),
            ({"class": 2, "description": "РЩ"}, []),
            ({"class": 2, "description": "РЩ"}, [{"type": None, "verdict": "not_target", "evidence": "Фон"}]),
            (None, [{"type": "rsh", "verdict": "ok", "evidence": "Целый"}]),
            ({"class": 2, "description": "РЩ"}, [{"type": None, "verdict": "review", "evidence": "Скрыт"}]),
        ]
        for frame, crops in examples:
            with self.subTest(frame=frame, crops=crops):
                pred = self.simulate(frame, crops)
                self.assertEqual(pred["objects"][0]["status"], "review")
                self.assertTrue(pred["review_reason"])
