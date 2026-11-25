"""Тесты формирования OpenRouter-запросов в VLM (маршрутизация провайдеров и квантований)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import requests
    import draw  # noqa: E402
    import variants  # noqa: E402
    import vlm  # noqa: E402
except ModuleNotFoundError:
    requests = None
    draw = None
    variants = None
    vlm = None

TAX = {
    "object_types": {"manhole": "люк"},
    "damage": {"cover_broken": "сломана"},
}


@unittest.skipIf(vlm is None, "нужен requests/PIL")
class OpenRouterProviderTest(unittest.TestCase):
    def test_paid_openrouter_uses_cheapest_non_4bit_provider(self):
        client = vlm.VLM({"base_url": "https://openrouter.ai/api/v1", "model": "qwen/qwen3.8-27b"}, TAX)
        body = client._body(b"jpeg", "prompt")
        self.assertEqual(body["usage"], {"include": True})
        provider = body["provider"]
        self.assertEqual(provider["sort"], "price")
        self.assertTrue(provider["allow_fallbacks"])
        self.assertIn("fp8", provider["quantizations"])
        self.assertIn("bf16", provider["quantizations"])
        for stripped in ("int4", "fp4", "mxfp4", "nvfp4"):
            self.assertNotIn(stripped, provider["quantizations"])

    def test_free_openrouter_skips_quantization_filter(self):
        client = vlm.VLM({"base_url": "https://openrouter.ai/api/v1", "model": "google/gemma-4-31b-it:free"}, TAX)
        body = client._body(b"jpeg", "prompt")
        self.assertEqual(body["provider"], {"sort": "price", "allow_fallbacks": True})

    def test_local_endpoint_does_not_send_openrouter_fields(self):
        client = vlm.VLM({"base_url": "http://localhost:8080/v1", "model": "Qwen/Qwen3.8-27B"}, TAX)
        body = client._body(b"jpeg", "prompt")
        self.assertNotIn("provider", body)
        self.assertNotIn("usage", body)

    def test_explicit_provider_override_and_disable(self):
        custom = {"sort": "throughput", "quantizations": ["bf16"]}
        c1 = vlm.VLM({"base_url": "https://openrouter.ai/api/v1", "model": "qwen/qwen3.8-27b",
                      "provider": custom}, TAX)
        self.assertEqual(c1._body(b"jpeg", "prompt")["provider"], custom)

        c2 = vlm.VLM({"base_url": "https://openrouter.ai/api/v1", "model": "qwen/qwen3.8-27b",
                      "provider": False}, TAX)
        self.assertNotIn("provider", c2._body(b"jpeg", "prompt"))


@unittest.skipIf(vlm is None, "нужен requests/PIL")
class BboxFormatAndPromptTest(unittest.TestCase):
    def test_auto_detects_yxyx_for_gemma_and_xyxy_for_qwen(self):
        self.assertEqual(vlm.bbox_format_for({"model": "google/gemma-4-26b-a4b-it"}), "yxyx")
        self.assertEqual(vlm.bbox_format_for({"model": "google/gemma-4-31b-it:free"}), "yxyx")
        self.assertEqual(vlm.bbox_format_for({"model": "qwen/qwen3-vl-235b-a22b-instruct"}), "xyxy")
        self.assertEqual(vlm.bbox_format_for({"model": "qwen/qwen3.8-27b"}), "xyxy")

    def test_explicit_override_and_validation(self):
        self.assertEqual(vlm.bbox_format_for({"model": "google/gemma-4-31b-it", "bbox_format": "xyxy"}), "xyxy")
        with self.assertRaises(ValueError):
            vlm.bbox_format_for({"model": "qwen/qwen3.8-27b", "bbox_format": "cxcywh"})

    def test_prompt_matches_model_bbox_format_and_exclusions(self):
        qwen = vlm.VLM({"base_url": "http://localhost:8080/v1", "model": "qwen/qwen3.8-27b"}, TAX)
        gemma = vlm.VLM({"base_url": "http://localhost:8080/v1", "model": "google/gemma-4-26b-a4b-it"}, TAX)
        self.assertIn("[x1, y1, x2, y2]", qwen.prompt)
        self.assertIn("[ymin, xmin, ymax, xmax]", gemma.prompt)
        for prompt in (qwen.prompt, gemma.prompt):
            self.assertIn("КГХ", prompt)
            self.assertIn("строительные и дорожные ограждения", prompt)
        self.assertIn("строительное ограждение", vlm.build_crop_prompt(TAX, "cabinet"))

    def test_normalize_xyxy_vs_yxyx(self):
        objs_xyxy = [{"type": "manhole", "bbox": [522, 0, 578, 976], "status": "ok", "damage": None}]
        objs_yxyx = [{"type": "manhole", "bbox": [0, 522, 976, 578], "status": "ok", "damage": None}]
        res_xyxy = vlm.normalize(objs_xyxy, 1280, 657, TAX, bbox_format="xyxy")
        res_yxyx = vlm.normalize(objs_yxyx, 1280, 657, TAX, bbox_format="yxyx")
        self.assertEqual(res_xyxy[0]["bbox"], [668, 0, 740, 641])
        self.assertEqual(res_yxyx[0]["bbox"], [668, 0, 740, 641])

    def test_normalize_clamps_and_drops_invalid_boxes(self):
        objs = [
            {"type": "manhole", "box_2d": [-10, 500, 1050, 600], "status": "ok"},
            {"type": "manhole", "bbox": [100, 100, 100, 200], "status": "ok"},
            {"type": "manhole", "bbox": [float("nan"), 0, 10, 10], "status": "ok"},
        ]
        res = vlm.normalize(objs, 1000, 500, TAX, bbox_format="yxyx")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["bbox"], [500, 0, 600, 500])

    def test_fingerprint_changes_with_bbox_format(self):
        c1 = vlm.VLM({"base_url": "http://localhost:8080/v1", "model": "custom/vlm", "bbox_format": "xyxy"}, TAX)
        c2 = vlm.VLM({"base_url": "http://localhost:8080/v1", "model": "custom/vlm", "bbox_format": "yxyx"}, TAX)
        self.assertNotEqual(c1.fingerprint(), c2.fingerprint())

    def test_recall_prompt_mode_preserves_exclusions_and_changes_fingerprint(self):
        c_def = vlm.VLM({"base_url": "http://localhost:8080/v1", "model": "custom/vlm"}, TAX)
        c_rec = vlm.VLM({"base_url": "http://localhost:8080/v1", "model": "custom/vlm", "prompt_mode": "recall"}, TAX)
        self.assertIn("КГХ", c_rec.prompt)
        self.assertIn("строительные и дорожные ограждения", c_rec.prompt)
        self.assertIn("обочины и газоны", c_rec.prompt)
        self.assertIn("дополнительно висит знак или светофор", c_rec.prompt)
        self.assertNotEqual(c_def.fingerprint(), c_rec.fingerprint())
        with self.assertRaises(ValueError):
            vlm.VLM({"base_url": "http://localhost:8080/v1", "model": "custom/vlm", "prompt_mode": "unknown"}, TAX)


def _ok_resp(content: str, cost: float = 0.001):
    r = mock.Mock()
    r.status_code = 200
    r.ok = True
    r.json.return_value = {"choices": [{"message": {"content": content}}], "usage": {"cost": cost}}
    return r


def _status_resp(code: int, text: str = "err"):
    r = mock.Mock()
    r.status_code = code
    r.ok = False
    r.text = text
    r.json.side_effect = ValueError("not json")
    return r


@unittest.skipIf(vlm is None, "нужен requests/PIL")
class RetryTest(unittest.TestCase):
    def _client(self, **extra):
        cfg = {"base_url": "http://localhost:8080/v1", "model": "custom/vlm", "retries": 4, "retry_delay": 1.0, **extra}
        return vlm.VLM(cfg, TAX)

    def test_retries_ssl_connection_and_chunked_errors(self):
        client = self._client()
        side = [
            requests.exceptions.SSLError("EOF"),
            requests.exceptions.ConnectionError("remote disconnected"),
            requests.exceptions.ChunkedEncodingError("ended prematurely"),
            _ok_resp('{"objects": []}', cost=0.002),
        ]
        with mock.patch("vlm.requests.post", side_effect=side) as post, \
             mock.patch("vlm.time.sleep") as sleep:
            text, cost = client.call(b"jpeg", "prompt")
        self.assertEqual(text, '{"objects": []}')
        self.assertAlmostEqual(cost, 0.002)
        self.assertEqual(post.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.0, 2.0, 4.0])

    def test_exhausted_network_errors_recorded_in_ask(self):
        client = self._client(retries=3, retry_delay=0.5)
        with mock.patch("vlm.requests.post", side_effect=requests.exceptions.SSLError("EOF")) as post, \
             mock.patch("vlm.time.sleep") as sleep:
            res = client._ask(b"jpeg", "prompt", vlm.parse)
        self.assertEqual(post.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.5, 1.0])
        self.assertIsNone(res["value"])
        self.assertFalse(res["bad_json"])
        self.assertIn("EOF", res["error"])

    def test_retries_transient_http_status_but_not_400(self):
        client = self._client()
        with mock.patch("vlm.requests.post", side_effect=[_status_resp(429), _status_resp(503), _ok_resp("{}", 0.0)]) as post, \
             mock.patch("vlm.time.sleep") as sleep:
            text, _ = client.call(b"jpeg", "prompt")
        self.assertEqual(text, "{}")
        self.assertEqual(post.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.0, 2.0])

        with mock.patch("vlm.requests.post", return_value=_status_resp(400, "bad req")) as post400, \
             mock.patch("vlm.time.sleep") as sleep400:
            with self.assertRaises(requests.HTTPError):
                client.call(b"jpeg", "prompt")
        self.assertEqual(post400.call_count, 1)
        sleep400.assert_not_called()


@unittest.skipIf(vlm is None, "нужен requests/PIL")
class ContextAndDetVLMTest(unittest.TestCase):
    def test_body_accepts_single_and_multiple_images(self):
        client = vlm.VLM({"base_url": "http://localhost:8080/v1", "model": "qwen/qwen3-vl-235b-a22b-instruct"}, TAX)
        single = client._body(b"img1", "p")["messages"][0]["content"]
        self.assertEqual([c["type"] for c in single], ["image_url", "text"])
        both = client._body([b"img1", b"img2"], "p")["messages"][0]["content"]
        self.assertEqual([c["type"] for c in both], ["image_url", "image_url", "text"])

    def test_context_prompt_rules_and_highlight_box(self):
        prompt = vlm.build_context_crop_prompt(TAX, "pole", has_crop=True)
        self.assertIn("полный кадр", prompt)
        self.assertIn("вырез", prompt)
        self.assertIn("перспективный завал", prompt)
        self.assertIn("ливневую решётку", prompt)
        img = Image.new("RGB", (200, 100), (20, 20, 20))
        highlighted = draw.highlight_box(img, [50, 30, 80, 60])
        self.assertEqual(highlighted.size, (200, 100))
        # исходная картинка не изменена, рамка нарисована снаружи границ объекта
        self.assertEqual(img.getpixel((48, 45)), (20, 20, 20))
        self.assertEqual(highlighted.getpixel((48, 45)), (255, 0, 180))
        self.assertEqual(highlighted.getpixel((65, 45)), (20, 20, 20))

    def test_det_vlm_context_modes_and_fingerprint(self):
        fake_det = mock.Mock()
        fake_det.fingerprint.return_value = "det-fp"
        fake_det.detect.return_value = [{"type": "pole", "bbox": [20, 10, 40, 90], "score": 0.9, "label": "Pole"}]
        root_cfg = {
            "taxonomy": TAX,
            "detector": {"pad": 0.2, "crop_min_side": 50},
            "variants": {
                "v": {"base_url": "http://localhost:8080/v1", "model": "custom/vlm"},
                "d_crop": {"kind": "det_vlm", "vlm": "v", "context_mode": "crop"},
                "d_ctx": {"kind": "det_vlm", "vlm": "v", "context_mode": "context"},
                "d_both": {"kind": "det_vlm", "vlm": "v", "context_mode": "both"},
                "d_bad": {"kind": "det_vlm", "vlm": "v", "context_mode": "invalid"},
            },
        }
        with mock.patch("variants.get_detector", return_value=fake_det):
            m_crop = variants.make("d_crop", root_cfg)
            m_ctx = variants.make("d_ctx", root_cfg)
            m_both = variants.make("d_both", root_cfg)
            with self.assertRaises(ValueError):
                variants.make("d_bad", root_cfg)

        self.assertEqual(len({m_crop.fingerprint(), m_ctx.fingerprint(), m_both.fingerprint()}), 3)

        img = Image.new("RGB", (200, 100), (10, 10, 10))
        with mock.patch.object(m_both.vlm, "classify_crop_in_context",
                               return_value={"answer": {"is_object": True, "type": "pole", "status": "ok",
                                                        "damage": None, "description": "ровная"},
                                             "error": None, "bad_json": False, "cost": 0.001}) as ctx_call:
            res = m_both.predict(img)
        self.assertEqual(ctx_call.call_count, 1)
        ctx_arg, crop_arg, hint_arg = ctx_call.call_args.args
        self.assertEqual(ctx_arg.size, (200, 100))
        self.assertIsNotNone(crop_arg)
        self.assertEqual(hint_arg, "pole")
        self.assertEqual(len(res["objects"]), 1)
        self.assertEqual(res["objects"][0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
