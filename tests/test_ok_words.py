"""«Исправен/неисправен» по названию дефекта: без sklearn (его нет локально), через ast.

classify.py тянет joblib/sklearn, поэтому берём из исходника только OK_WORDS/is_ok/NOT_OK_WORDS
и исполняем их — тест проверяет настоящий код, а не его копию.
"""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "classify.py"


def load():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"OK_WORDS", "NOT_OK_WORDS", "is_ok"}
    body = [n for n in tree.body
            if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name in names)]
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["is_ok"]


class OkWordsTest(unittest.TestCase):
    def setUp(self):
        self.is_ok = load()

    def test_ok_words(self):
        for text in ("Норма", "исправен", "без дефектов", "без повреждений", "нет дефектов",
                     "удовлетворительное состояние", "ok", "good", "normal"):
            with self.subTest(text=text):
                self.assertTrue(self.is_ok(text))

    def test_not_ok_words(self):
        for text in ("неисправен", "неисправность", "неудовлетворительное состояние",
                     "не удовлетворяет требованиям", "оборваны кабели"):
            with self.subTest(text=text):
                self.assertFalse(self.is_ok(text))

    def test_negation_wins(self):
        # «неудовлетворительно» содержит «удовлетвор»: отрицание обязано проверяться раньше
        self.assertFalse(self.is_ok("Состояние неудовлетворительное"))


if __name__ == "__main__":
    sys.exit(unittest.main())
