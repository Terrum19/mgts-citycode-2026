"""Атомарная запись: побочных temp-файлов нет, JSON читается обратно."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomic import read_json, write_json_atomic, write_text_atomic  # noqa: E402


class AtomicTest(unittest.TestCase):
    def test_json_roundtrip_without_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x" / "data.json"
            write_json_atomic(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            self.assertEqual(read_json(path), {"a": 1})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_text_replaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.txt"
            path.write_text("old")
            write_text_atomic(path, "new")
            self.assertEqual(path.read_text(), "new")

    def test_read_missing_is_default(self):
        self.assertEqual(read_json(Path("/nonexistent/file.json"), default={}), {})


if __name__ == "__main__":
    unittest.main()
