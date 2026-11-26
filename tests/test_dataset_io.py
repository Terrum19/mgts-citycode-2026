import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_io import (  # noqa: E402
    ingest_uploaded_directory,
    ingest_uploaded_photos,
    ingest_zip_archive,
    inspect_dataset,
    is_ok_defect,
    split_dataset_label,
)


class _FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self) -> bytes:
        return self._data


def _img_bytes(color=(120, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (24, 24), color).save(buf, format="JPEG")
    return buf.getvalue()


class DatasetIOTests(unittest.TestCase):
    def test_dss_label_parsing_and_ok_detection(self):
        self.assertEqual(split_dataset_label("МП.РЩ"), ("РЩ", "МП"))
        self.assertEqual(split_dataset_label("МП. ТКШ"), ("ТКШ", "МП"))
        self.assertEqual(
            split_dataset_label("Неудовлетворительное состояние ЛКП. РЩ"),
            ("РЩ", "Неудовлетворительное состояние ЛКП"),
        )
        self.assertEqual(
            split_dataset_label("Удовлетворительное состояние ТКШ"),
            ("ТКШ", "Удовлетворительное состояние"),
        )
        self.assertTrue(is_ok_defect("Удовлетворительное состояние"))
        self.assertFalse(is_ok_defect("Неудовлетворительное состояние ЛКП"))
        self.assertFalse(is_ok_defect("МП"))

    def test_ingest_photos_handles_duplicates_as_unlabeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            uploads = [_FakeUpload("cam.jpg", _img_bytes()), _FakeUpload("cam.jpg", _img_bytes())]
            root = ingest_uploaded_photos(uploads, Path(tmp) / "input")
            bundle = inspect_dataset(root)
            self.assertEqual(len(bundle.items), 2)
            self.assertFalse(bundle.has_gt)
            self.assertEqual([it["key"] for it in bundle.items], ["cam.jpg", "cam__2.jpg"])

    def test_ingest_directory_with_dss_folders_and_excel(self):
        with tempfile.TemporaryDirectory() as tmp:
            excel_buf = io.BytesIO()
            pd.DataFrame([{"Имя файла": "a1.jpg", "Описание": "Смята дверь РЩ", "ID объекта": "БС-1001"}]).to_excel(
                excel_buf, index=False
            )
            uploads = [
                _FakeUpload("Датасет-ДСС/Фото.1.jpg", _img_bytes()),
                _FakeUpload("Датасет-ДСС/.DS_Store", b"junk"),
                _FakeUpload("Датасет-ДСС/МП.РЩ/a1.jpg", _img_bytes()),
                _FakeUpload("Датасет-ДСС/МП.РЩ/описание.xlsx", excel_buf.getvalue()),
                _FakeUpload("Датасет-ДСС/Удовлетворительное состояние ТКШ/b1.jpg", _img_bytes()),
            ]
            root = ingest_uploaded_directory(uploads, Path(tmp) / "input")
            bundle = inspect_dataset(root)
            self.assertTrue(bundle.has_gt)
            self.assertEqual(bundle.gt_kind, "folders")
            self.assertEqual(len(bundle.items), 2)
            by_key = {it["key"]: it for it in bundle.items}
            self.assertEqual(by_key["МП.РЩ/a1.jpg"]["gt_type"], "РЩ")
            self.assertEqual(by_key["МП.РЩ/a1.jpg"]["gt_status"], "bad")
            self.assertEqual(by_key["МП.РЩ/a1.jpg"]["meta_description"], "Смята дверь РЩ")
            self.assertEqual(by_key["Удовлетворительное состояние ТКШ/b1.jpg"]["gt_type"], "ТКШ")
            self.assertEqual(by_key["Удовлетворительное состояние ТКШ/b1.jpg"]["gt_status"], "ok")

    def test_ingest_zip_blocks_traversal_and_extracts_labels(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../evil.jpg", _img_bytes())
            zf.writestr("__MACOSX/._ghost.jpg", b"junk")
            zf.writestr("export/ТКШ/МП/c1.jpg", _img_bytes())
            zf.writestr("export/РЩ/Удовлетворительное состояние/c2.jpg", _img_bytes())
        buf.seek(0)

        with tempfile.TemporaryDirectory() as tmp:
            root = ingest_zip_archive(buf, Path(tmp) / "zip_in")
            self.assertFalse((Path(tmp) / "evil.jpg").exists())
            bundle = inspect_dataset(root)
            self.assertTrue(bundle.has_gt)
            self.assertEqual(bundle.gt_kind, "folders")
            self.assertEqual(len(bundle.items), 2)
            self.assertEqual(bundle.items[0]["gt_status"], "ok")
            self.assertEqual(bundle.items[1]["gt_status"], "bad")

    def test_inspect_dataset_with_labels_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (32, 32)).save(root / "p1.jpg")
            (root / "labels.json").write_text(
                json.dumps([{"file": "p1.jpg", "objects": [{"type": "tksh", "bbox": [2, 2, 20, 20], "status": "bad", "damage": "mechanical_damage"}]}]),
                encoding="utf-8",
            )
            bundle = inspect_dataset(root)
            self.assertTrue(bundle.has_gt)
            self.assertEqual(bundle.gt_kind, "objects")
            self.assertEqual(bundle.items[0]["gt_status"], "bad")


if __name__ == "__main__":
    unittest.main()
