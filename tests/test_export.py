"""Тесты выгрузки Label Studio: разбор .env, защита от битых аннотаций и проверка скачанных картинок."""
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/label"))

try:
    import export as ls_export  # noqa: E402
except ModuleNotFoundError:
    ls_export = None


@unittest.skipIf(ls_export is None, "нужен requests/PIL")
class ExportTest(unittest.TestCase):
    def test_load_ls_env_strips_quotes_and_slash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / ".env").write_text('export LABEL_STUDIO_URL="http://10.180.132.80:8080/"\n'
                                     "LABEL_STUDIO_TOKEN='abc123'\n", encoding="utf-8")
            old_u = os.environ.pop("LABEL_STUDIO_URL", None)
            old_t = os.environ.pop("LABEL_STUDIO_TOKEN", None)
            try:
                url, token = ls_export.load_ls_env(p)
            finally:
                if old_u is not None:
                    os.environ["LABEL_STUDIO_URL"] = old_u
                if old_t is not None:
                    os.environ["LABEL_STUDIO_TOKEN"] = old_t
            self.assertEqual(url, "http://10.180.132.80:8080")
            self.assertEqual(token, "abc123")

    def test_task_filename_variants_and_safety(self):
        self.assertEqual(ls_export.task_filename({"data": {"file": "osm_1.jpg"}}), "osm_1.jpg")
        self.assertEqual(
            ls_export.task_filename({"data": {"image": "/data/local-files/?d=panoramas/osm_2.jpg"}}),
            "osm_2.jpg",
        )
        self.assertEqual(
            ls_export.task_filename({"data": {"image": "/data/upload/1/deadbeef-img.jpg"}}),
            "deadbeef-img.jpg",
        )
        self.assertEqual(ls_export.task_filename({"data": {"file": "../../etc/passwd"}}), "passwd")

    def test_convert_handles_partial_regions_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp)
            Image.new("RGB", (1000, 500)).save(img_dir / "manual.jpg", "JPEG")
            task = {
                "data": {"image": "/data/upload/1/manual.jpg"},
                "annotations": [
                    {"id": 1, "was_cancelled": True, "result": []},
                    {
                        "id": 2,
                        "updated_at": "2026-11-25T05:00:00Z",
                        "result": [
                            {"from_name": "frame", "value": {"choices": ["target_not_visible"]}},
                            {"id": "r1", "type": "rectanglelabels", "original_width": None,
                             "value": {"x": 10, "y": 20, "width": 30, "height": 40, "rectanglelabels": ["manhole"]}},
                            {"id": "r1", "from_name": "status", "value": {"choices": ["bad"]}},
                            {"id": "r1", "from_name": "damage", "value": {"choices": []}},
                            {"id": "r1", "from_name": "note", "value": {"text": "трещина"}},
                            {"id": "r2", "type": "rectanglelabels",
                             "value": {"x": 0, "y": 0, "width": 10, "height": 10, "rectanglelabels": []}},
                        ],
                    },
                ],
            }
            out = ls_export.convert(task, default_split="test", split_map={"manual.jpg": "dev"}, images_dir=img_dir)
            self.assertEqual(out["file"], "manual.jpg")
            self.assertEqual((out["width"], out["height"]), (1000, 500))
            self.assertEqual(out["split"], "dev")
            self.assertEqual(len(out["objects"]), 1)
            self.assertEqual(out["objects"][0], {
                "type": "manhole", "bbox": [100, 100, 400, 300],
                "status": "bad", "damage": None, "note": "трещина",
            })

    def test_download_image_skips_existing_and_rejects_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            existing = d / "a.jpg"
            existing.write_bytes(b"123")
            sess = MagicMock()
            self.assertEqual(ls_export.download_image(sess, "http://hub:8080", {}, "/data/a.jpg", existing), "exists")
            sess.request.assert_not_called()

            bad_resp = MagicMock(status_code=200, ok=True, content=b"<html>login</html>")
            sess.request.return_value = bad_resp
            with self.assertRaises(Exception):
                ls_export.download_image(sess, "http://hub:8080", {}, "/data/b.jpg", d / "b.jpg")
            self.assertFalse((d / "b.jpg").exists())

            buf = io.BytesIO()
            Image.new("RGB", (32, 32)).save(buf, "JPEG")
            ok_resp = MagicMock(status_code=200, ok=True, content=buf.getvalue())
            sess.request.return_value = ok_resp
            res = ls_export.download_image(sess, "http://hub:8080", {}, "/data/c.jpg", d / "c.jpg")
            self.assertEqual(res, "downloaded")
            self.assertTrue((d / "c.jpg").exists())


if __name__ == "__main__":
    unittest.main()
