"""Таксономия: канонический fingerprint, legacy-режимы и хэши файлов."""
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import taxonomy  # noqa: E402

TAX = {"object_types": {"b": "B", "a": "A"}, "damage": {"d2": "D2", "d1": "D1"}}
OTHER_ORDER = {"damage": {"d1": "D1", "d2": "D2"}, "object_types": {"a": "A", "b": "B"}}


class FingerprintTest(unittest.TestCase):
    def test_order_does_not_matter(self):
        self.assertEqual(taxonomy.taxonomy_fingerprint(TAX), taxonomy.taxonomy_fingerprint(OTHER_ORDER))

    def test_descriptions_matter(self):
        changed = {"object_types": {"a": "A", "b": "B2"}, "damage": {"d1": "D1", "d2": "D2"}}
        self.assertNotEqual(taxonomy.taxonomy_fingerprint(TAX), taxonomy.taxonomy_fingerprint(changed))

    def test_names_are_presentation_only(self):
        with_names = {**OTHER_ORDER, "names": {"a": "совсем другое"}}
        self.assertEqual(taxonomy.taxonomy_fingerprint(TAX), taxonomy.taxonomy_fingerprint(with_names))

    def test_canonical_is_valid_json(self):
        self.assertEqual(json.loads(taxonomy.canonical_taxonomy(TAX))["object_types"], {"a": "A", "b": "B"})


class CheckpointTest(unittest.TestCase):
    def _state(self, **extra):
        return {"format_version": taxonomy.FORMAT_VERSION,
                "taxonomy_sha256": taxonomy.taxonomy_fingerprint(TAX), **extra}

    def test_match_ok(self):
        info = taxonomy.validate_checkpoint(self._state(), TAX, legacy="error")
        self.assertTrue(info["matches"])
        self.assertFalse(info["legacy"])

    def test_mismatch_always_fatal(self):
        state = self._state(taxonomy_sha256="deadbeef")
        for mode in ("warn", "allow", "error"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                taxonomy.validate_checkpoint(state, TAX, legacy=mode)

    def test_v2_without_fingerprint_fatal(self):
        with self.assertRaises(ValueError):
            taxonomy.validate_checkpoint({"format_version": 2}, TAX)

    def test_legacy_modes(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = taxonomy.validate_checkpoint({}, TAX, legacy="warn")
        self.assertTrue(info["legacy"])
        self.assertTrue(caught)
        with self.assertRaises(ValueError):
            taxonomy.validate_checkpoint({}, TAX, legacy="error")
        self.assertTrue(taxonomy.validate_checkpoint({}, TAX, legacy="allow")["legacy"])


class HashTest(unittest.TestCase):
    def test_deterministic_and_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "a" / "x.jpg").write_bytes(b"x")
            (root / "y.txt").write_bytes(b"y")
            first = taxonomy.hash_files([root / "a" / "x.jpg", root / "y.txt"], [root])
            second = taxonomy.hash_files([root / "y.txt", root / "a" / "x.jpg"], [root])
            self.assertEqual(first, second)
            self.assertEqual(set(first["files"]), {"a/x.jpg", "y.txt"})
            self.assertEqual(first["count"], 2)

    def test_package_versions_is_json_safe(self):
        versions = taxonomy.package_versions()
        self.assertIn("python", versions)
        json.dumps(versions)


if __name__ == "__main__":
    unittest.main()
