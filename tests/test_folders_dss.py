"""Разбор папок выдачи ДСС: шесть грамматик, старые форматы и группировка повторов."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classify import groups_of, is_ok  # noqa: E402
from folders import load, split_label  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
ROOTS = [DATA / "doc-1795639519" / "Датасет-ДСС", DATA / "doc-1795639569" / "Датасет-ДСС"]


class DssLabelTest(unittest.TestCase):
    CASES = [
        ("МП.РЩ", "РЩ", "МП"),
        ("МП. ТКШ", "ТКШ", "МП"),
        ("мп.рщ", "РЩ", "МП"),
        ("МП . РЩ", "РЩ", "МП"),
        ("  МП.ТКШ  ", "ТКШ", "МП"),
        ("Неудовлетворительное состояние ЛКП. РЩ", "РЩ", "Неудовлетворительное состояние ЛКП"),
        ("Неудовлетворительное состояние ЛКП. ТКШ", "ТКШ", "Неудовлетворительное состояние ЛКП"),
        ("неудовлетворительное  состояние лкп.ткш", "ТКШ", "Неудовлетворительное состояние ЛКП"),
        ("Удовлетворительное состояние РЩ", "РЩ", "Удовлетворительное состояние"),
        ("Удовлетворительное состояние ТКШ", "ТКШ", "Удовлетворительное состояние"),
    ]

    def test_dss_grammar(self):
        for label, typ, defect in self.CASES:
            with self.subTest(label=label):
                self.assertEqual(split_label(label, None), (typ, defect))

    def test_generic_formats_unchanged(self):
        self.assertEqual(split_label("Колодец/Норма", None), ("Колодец", "Норма"))
        self.assertEqual(split_label("Опора - Наклон", None), ("Опора", "Наклон"))
        self.assertEqual(split_label("Одиночная", None), ("Одиночная", ""))
        self.assertEqual(split_label("ТКШ__ЛКП", "__"), ("ТКШ", "ЛКП"))

    def test_ok_recognises_satisfactory(self):
        self.assertTrue(is_ok("Удовлетворительное состояние"))
        self.assertTrue(is_ok("Норма"))
        self.assertFalse(is_ok("Неудовлетворительное состояние ЛКП"))
        self.assertFalse(is_ok("МП"))

    def test_groups_use_stem(self):
        items = [{"file": "a/20211214.jpg", "extra": {}}, {"file": "b/20211214.jpg", "extra": {}}]
        self.assertEqual(groups_of(items), ["20211214", "20211214"])

    def test_groups_prefer_explicit_column(self):
        items = [{"file": "a/x.jpg", "extra": {"Группа": "site_1"}}, {"file": "b/y.jpg", "extra": {"Группа": ""}}]
        self.assertEqual(groups_of(items), ["site_1", "y"])


@unittest.skipUnless(all(p.is_dir() for p in ROOTS), "выдача ДСС не распакована")
class DssDatasetTest(unittest.TestCase):
    def test_six_classes_and_ok_split(self):
        items = [item for root in ROOTS for item in load(root)]
        self.assertEqual(len(items), 61)
        self.assertEqual(len({x["label"] for x in items}), 6)
        ok = sum(is_ok(x["defect"]) for x in items)
        self.assertEqual((ok, len(items) - ok), (20, 41))

    def test_every_photo_has_description(self):
        items = [item for root in ROOTS for item in load(root)]
        self.assertTrue(all(x["description"] for x in items))

    def test_types_are_only_tksh_and_rsh(self):
        items = [item for root in ROOTS for item in load(root)]
        self.assertEqual({x["type"] for x in items}, {"ТКШ", "РЩ"})
        self.assertEqual(dict(Counter(x["type"] for x in items)), {"РЩ": 31, "ТКШ": 30})


if __name__ == "__main__":
    unittest.main()
