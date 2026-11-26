import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _make_sample_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel in ("МП.ТКШ/cab_bad.jpg", "Удовлетворительное состояние РЩ/cab_ok.jpg"):
            img_buf = io.BytesIO()
            Image.new("RGB", (64, 64), (100, 110, 120)).save(img_buf, format="JPEG")
            zf.writestr(rel, img_buf.getvalue())
    return buf.getvalue()


class _FakeModelA:
    workers = 1

    def predict(self, img):
        return {
            "objects": [{
                "type": "tksh",
                "bbox": [4, 4, 56, 56],
                "status": "bad",
                "damage": "mechanical_damage",
                "description": "Смята дверца шкафа ТКШ",
            }],
            "seconds": 0.1,
            "cost": 0.0,
            "error": None,
            "bad_json": False,
        }


class _FakeModelB:
    workers = 1

    def __init__(self):
        self.calls = 0

    def predict(self, img):
        self.calls += 1
        # На первом кадре соглашается с Моделью А, на втором — расхождение по статусу и типу
        if self.calls == 1:
            return {
                "objects": [{
                    "type": "tksh",
                    "bbox": [6, 6, 54, 54],
                    "status": "bad",
                    "damage": "mechanical_damage",
                    "description": "Повреждение корпуса ТКШ",
                }],
                "seconds": 0.1,
                "cost": 0.0,
                "error": None,
                "bad_json": False,
            }
        return {
            "objects": [{
                "type": "rsh",
                "bbox": [6, 6, 54, 54],
                "status": "ok",
                "damage": None,
                "description": "РЩ без видимых повреждений",
            }],
            "seconds": 0.1,
            "cost": 0.0,
            "error": None,
            "bad_json": False,
        }


import shutil

class AppSmokeTests(unittest.TestCase):
    def test_app_initial_render_and_local_folder_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_dir = tmp_path / "ds"
            (ds_dir / "МП.ТКШ").mkdir(parents=True)
            (ds_dir / "Удовлетворительное состояние РЩ").mkdir(parents=True)
            Image.new("RGB", (64, 64), (150, 50, 50)).save(ds_dir / "МП.ТКШ" / "01_bad.jpg")
            Image.new("RGB", (64, 64), (50, 150, 50)).save(ds_dir / "Удовлетворительное состояние РЩ" / "02_ok.jpg")

            fake_b = _FakeModelB()
            fake_a = _FakeModelA()

            def fake_make(name, cfg):
                return fake_a if name == "qwen235" else fake_b

            def fake_ready(name, cfg):
                return (True, "OK") if name in ("qwen235", "gemma31") else (False, "no key")

            out_root = tmp_path / "app_root"
            out_root.mkdir()
            with (
                patch("variants.make", side_effect=fake_make),
                patch("consensus.variant_readiness", side_effect=fake_ready),
                patch.dict("sys.modules", {}),
            ):
                # Запускаем AppTest и удаляем временный каталог out/app после теста
                before = set((ROOT / "out" / "app").glob("*")) if (ROOT / "out" / "app").exists() else set()
                try:
                    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=20)
                    at.run()
                    self.assertFalse(at.exception)

                    # Проверяем режимы загрузки в сайдбаре: Папка, Фотографии, ZIP-архив
                    self.assertEqual(at.sidebar.radio[0].options, ["Папка", "Фотографии", "ZIP-архив"])

                    # Вводим путь к локальной папке в сайдбаре и запускаем анализ
                    at.sidebar.text_input[0].set_value(str(ds_dir)).run()
                    btn = at.sidebar.button[0]
                    self.assertFalse(btn.disabled)
                    btn.click().run()

                    self.assertFalse(at.exception)
                    subheaders = [el.value for el in at.subheader]
                    self.assertTrue(any("Общие результаты по дата-сету" in s for s in subheaders))
                    self.assertTrue(any("Успешно классифицированные поломки (1)" in s for s in subheaders))
                    self.assertTrue(any("Фото для дополнительного подтверждения (1)" in s for s in subheaders))

                    markdown_text = "\n".join(m.value for m in at.markdown)
                    self.assertIn("Как разошелось мнение моделей", markdown_text)
                    self.assertIn("ТКШ", markdown_text)

                    # Проверяем переключение решения оператора на спорном фото -> сразу переносится в поломки
                    op_radios = [r for r in at.radio if r.key and r.key.startswith("op_decision_")]
                    self.assertEqual(len(op_radios), 1)
                    op_radios[0].set_value("Подтвердить поломку").run()
                    self.assertFalse(at.exception)
                    subheaders_after = [el.value for el in at.subheader]
                    self.assertTrue(any("Успешно классифицированные поломки (2)" in s for s in subheaders_after))
                    self.assertTrue(any("Фото для дополнительного подтверждения (0)" in s for s in subheaders_after))
                finally:
                    after = set((ROOT / "out" / "app").glob("*")) if (ROOT / "out" / "app").exists() else set()
                    for created in after - before:
                        shutil.rmtree(created, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
