import csv
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

class _FakeSingleRoute:
    """Заглушка единственного маршрута qwen235: первый кадр — поломка, второй — расхождение."""

    workers = 1

    def __init__(self):
        self.calls = 0

    def predict(self, img):
        self.calls += 1
        if self.calls == 1:
            return {
                "objects": [{"type": "tksh", "bbox": [4, 4, 56, 56], "status": "bad",
                             "damage": "mechanical_damage", "description": "Смята дверца ТКШ",
                             "priority": "high", "review_reasons": []}],
                "crop_results": [], "review_reasons": [], "seconds": 0.1, "cost": 0.0,
                "error": None, "bad_json": False,
            }
        return {
            "objects": [{"type": "rsh", "bbox": [4, 4, 56, 56], "status": "review",
                         "damage": None, "description": "Промты разошлись",
                         "priority": "none", "review_reasons": ["verdict_disagreement"]}],
            "crop_results": [], "review_reasons": [], "seconds": 0.1, "cost": 0.0,
            "error": None, "bad_json": False,
        }


class AppSmokeTests(unittest.TestCase):
    def test_saved_frame_crop_demo_uses_real_results_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("vlm.ROOT", root), patch("variants.make", side_effect=AssertionError("live model")):
                app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=20).run()
                self.assertFalse(app.exception)
                next(button for button in app.sidebar.button if button.label == "Открыть готовый пример").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any("Сохранённый результат" in item.value for item in app.info))
                result = app.session_state["ui_result"]
                verdicts = result["verdicts"]
                self.assertEqual(verdicts["29159065.jpg"]["outcome"], "bad")
                self.assertEqual(verdicts["35453689.jpg"]["outcome"], "review")
                explanation = verdicts["35453689.jpg"]["checks"]["crop_description"]
                captions = " ".join(element.value for element in app.caption)
                self.assertIn(explanation, captions)
                self.assertIn("API в исходном прогоне", captions)
                self.assertNotIn("Одиночный режим", captions)
                self.assertEqual(verdicts["29159065.jpg"]["assessment"]["urgency"], "planned")
                self.assertEqual(verdicts["35453689.jpg"]["assessment"]["urgency"], "manual_review")
                with (result["out"] / "frames.csv").open(encoding="utf-8-sig") as stream:
                    frames = {row["file"]:row for row in csv.DictReader(stream)}
                self.assertEqual(frames["29159065.jpg"]["frame_class"], "rs_paint")
                self.assertEqual(frames["35453689.jpg"]["crop_class"], "rs_paint")
                self.assertTrue((result["out"] / "annotated/35453689.jpg").is_file())
                operator = next(radio for radio in app.radio if radio.key and radio.key.startswith("op_decision_"))
                operator.set_value("Признать исправным").run()
                self.assertFalse(app.exception)
                with (result["out"] / "frames.csv").open(encoding="utf-8-sig") as stream:
                    updated = {row["file"]: row for row in csv.DictReader(stream)}
                self.assertEqual(updated["35453689.jpg"]["outcome"], "ok")
                self.assertEqual(updated["35453689.jpg"]["operator_override"], "ok")
                self.assertEqual(updated["35453689.jpg"]["frame_class"], "rs_mech")

    def test_selected_live_route_uses_one_backend_and_preserves_each_check(self):
        from defense_examples import load_examples
        examples = load_examples()
        model = _FakeModelA()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "photos"
            folder.mkdir()
            source, answer = examples[0]
            shutil.copyfile(source, folder / "new.jpg")
            with (
                patch("vlm.ROOT", root),
                patch("variants.make", return_value=model) as make,
                patch.object(model, "predict", return_value=answer) as predict,
                patch("consensus.variant_readiness", return_value=(True, "Готов")),
            ):
                app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=20).run()
                app.sidebar.text_input[0].set_value(str(folder)).run()
                app.sidebar.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(make.call_args.args[0], "frame_crop_qwen235")
                predict.assert_called_once()
                result = app.session_state["ui_result"]
                self.assertFalse(result["example"])
                self.assertEqual(result["verdicts"]["new.jpg"]["checks"]["crop_description"], answer["crop_description"])

    def test_app_initial_render_and_local_folder_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_dir = tmp_path / "ds"
            (ds_dir / "МП.ТКШ").mkdir(parents=True)
            (ds_dir / "Удовлетворительное состояние РЩ").mkdir(parents=True)
            Image.new("RGB", (64, 64), (150, 50, 50)).save(ds_dir / "МП.ТКШ" / "01_bad.jpg")
            Image.new("RGB", (64, 64), (50, 150, 50)).save(ds_dir / "Удовлетворительное состояние РЩ" / "02_ok.jpg")

            fake = _FakeSingleRoute()

            def fake_make(name, cfg):
                return fake

            def fake_ready(name, cfg):
                return (True, "OK") if name == "frame_crop_qwen235" else (False, "no key")

            out_root = tmp_path / "app_root"
            out_root.mkdir()
            with (
                patch("variants.make", side_effect=fake_make),
                patch("consensus.variant_readiness", side_effect=fake_ready),
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
                    self.assertTrue(
                        "Как разошелось мнение моделей" in markdown_text
                        or "Почему фото передано на проверку" in markdown_text
                    )
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
