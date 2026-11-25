"""Кэш VLM: ключ от JPEG/промпта/variant, explicit review кэшируется, ошибки нет."""
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vlm_cache  # noqa: E402


class StubVLM:
    max_side = 800
    key = ""
    prompt_mode = "default"
    fingerprint_value = "stub-fp"

    def __init__(self):
        self.calls = 0

    def fingerprint(self):
        return self.fingerprint_value

    def _ask(self, jpeg, prompt, extract):
        self.calls += 1
        value = {"class": 1, "description": "объект", "review": False}
        return {"value": value, "error": None, "bad_json": False, "cost": 0.001,
                "seconds": 0.01, "raw": ""}


class CacheKeyTest(unittest.TestCase):
    def test_different_payload(self):
        base = {"schema_version": 2, "variant": "a", "image_sha256": "1"}
        self.assertNotEqual(vlm_cache.cache_key(base), vlm_cache.cache_key({**base, "variant": "b"}))
        self.assertNotEqual(vlm_cache.cache_key(base), vlm_cache.cache_key({**base, "image_sha256": "2"}))

    def test_cacheable(self):
        self.assertTrue(vlm_cache.cacheable({"label": "manhole - ok"}))
        self.assertTrue(vlm_cache.cacheable({"label": None, "review": True}))
        self.assertFalse(vlm_cache.cacheable({"label": None, "review": False}))
        self.assertFalse(vlm_cache.cacheable({"label": "x", "error": "timeout"}))
        self.assertFalse(vlm_cache.cacheable({"label": "x", "bad_json": True}))


class QwenCacheTest(unittest.TestCase):
    def test_reuses_by_content_and_invalidates_on_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.jpg"
            Image.new("RGB", (64, 64), (10, 20, 30)).save(image)
            stub = StubVLM()
            q = vlm_cache.Qwen(["manhole - ok"], root / "qwen.json", variant="v1",
                               vcfg={"workers": 1}, tax={}, vlm=stub)
            first = q.get([image])
            self.assertEqual(stub.calls, 1)
            self.assertEqual(first[0]["label"], "manhole - ok")
            again = q.get([image])
            self.assertEqual(stub.calls, 1)  # из кэша
            self.assertEqual(again[0]["label"], "manhole - ok")
            import json
            data = json.loads((root / "qwen.json").read_text())
            self.assertEqual(data["schema_version"], vlm_cache.CACHE_SCHEMA)
            self.assertEqual(len(data["entries"]), 1)
            # заменили файл по тому же пути — кэш обязан инвалидироваться
            Image.new("RGB", (64, 64), (200, 10, 10)).save(image)
            third = q.get([image])
            self.assertEqual(stub.calls, 2)
            self.assertEqual(third[0]["label"], "manhole - ok")
            data = json.loads((root / "qwen.json").read_text())
            self.assertEqual(len(data["entries"]), 2)
            q.close()

    def test_errors_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.jpg"
            Image.new("RGB", (32, 32)).save(image)

            class Bad(StubVLM):
                def _ask(self, jpeg, prompt, extract):
                    self.calls += 1
                    return {"value": None, "error": "timeout", "bad_json": False, "cost": 0.0,
                            "seconds": 0.0, "raw": ""}

            stub = Bad()
            q = vlm_cache.Qwen(["x"], root / "qwen.json", variant="v1", vcfg={"workers": 1},
                               tax={}, vlm=stub)
            out = q.get([image])
            self.assertIn("timeout", out[0]["error"])
            q.get([image])
            self.assertEqual(stub.calls, 2)  # ошибка не закэширована
            q.close()


if __name__ == "__main__":
    unittest.main()
