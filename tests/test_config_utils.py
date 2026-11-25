"""extends, offline-ограничения и проброс detector-конфига."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config_utils  # noqa: E402

CFG = {"variants": {
    "base": {"kind": "det_dino", "vlm": "remote", "cascade": True},
    "child": {"extends": "base", "vlm": "", "offline": True},
    "remote": {"base_url": "https://openrouter.ai/api/v1", "model": "m"},
    "local": {"base_url": "http://localhost:8080/v1", "model": "m"},
    "cycle1": {"extends": "cycle2"},
    "cycle2": {"extends": "cycle1"},
    "bad": {"extends": "missing"},
}}


class ResolveTest(unittest.TestCase):
    def test_extends_merges(self):
        out = config_utils.resolve_variant(CFG, "child")
        self.assertTrue(out["offline"])
        self.assertEqual(out["vlm"], "")
        self.assertNotIn("extends", out)

    def test_cycle(self):
        with self.assertRaises(ValueError):
            config_utils.resolve_variant(CFG, "cycle1")

    def test_unknown_parent(self):
        with self.assertRaises(ValueError):
            config_utils.resolve_variant(CFG, "bad")

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            config_utils.resolve_variant(CFG, "nope")


class OfflineTest(unittest.TestCase):
    def test_local_url(self):
        self.assertTrue(config_utils.is_local_url("http://localhost:8080/v1"))
        self.assertTrue(config_utils.is_local_url("http://127.0.0.1:8080/v1"))
        self.assertFalse(config_utils.is_local_url("https://openrouter.ai/api/v1"))

    def test_remote_vlm_forbidden_offline(self):
        with self.assertRaises(ValueError):
            config_utils.check_offline_vlm(CFG, {"vlm": "remote"}, True)
        config_utils.check_offline_vlm(CFG, {"vlm": "local"}, True)

    def test_detector_offline_flag(self):
        self.assertTrue(config_utils.detector_config({"detector": {"backend": "mask2former"}},
                                                     {"offline": True})["offline"])
        self.assertNotIn("offline", config_utils.detector_config({"detector": {}}, {}))


if __name__ == "__main__":
    unittest.main()
