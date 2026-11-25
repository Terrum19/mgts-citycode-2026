"""CSV-контракт: objects/frames/repair/review/triage без дублей и потери frame-only."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import export_rows  # noqa: E402
from results import make_frame, make_result  # noqa: E402


def obj(status, box=(0, 0, 10, 10), typ="manhole"):
    return {"type": typ, "bbox": list(box), "status": status, "damage": None,
            "description": "", "review_reasons": ["low_confidence"] if status == "review" else []}


class ExportTest(unittest.TestCase):
    def test_frame_only_bad(self):
        pred = make_result(frame=make_frame("bad", type="manhole", damage="cover_broken"),
                           objects=[], scope="frame")
        t = export_rows.export_rows({"a.jpg": pred}, {"a.jpg": {"ID": "1"}}, {"manhole": "Колодец"})
        self.assertEqual(t["objects"], [])
        self.assertEqual(len(t["frames"]), 1)
        self.assertEqual(t["repair"][0]["scope"], "frame")
        self.assertEqual(t["repair"][0]["ID"], "1")
        self.assertEqual(t["review"], [])
        self.assertEqual(len(t["triage"]), 1)

    def test_frame_only_review(self):
        pred = make_result(frame=make_frame("review", review_reasons=["quality"]), objects=[], scope="frame")
        t = export_rows.export_rows({"a.jpg": pred}, {}, {})
        self.assertEqual(t["repair"], [])
        self.assertEqual(len(t["review"]), 1)
        self.assertEqual(t["triage"][0]["triage"], "review")

    def test_mixed_bad_and_review_no_duplicate(self):
        pred = make_result(objects=[obj("bad"), obj("review", box=(20, 20, 30, 30))], scope="mixed")
        t = export_rows.export_rows({"a.jpg": pred}, {}, {})
        self.assertEqual(len(t["frames"]), 1)
        self.assertEqual(len(t["objects"]), 2)
        self.assertEqual([r["status"] for r in t["repair"]], ["bad"])
        self.assertEqual([r["status"] for r in t["review"]], ["review"])
        self.assertEqual(len(t["triage"]), 2)
        self.assertTrue(all(r["scope"] == "object" for r in t["triage"]))

    def test_object_only_has_derived_frame(self):
        pred = make_result(objects=[obj("ok")])
        t = export_rows.export_rows({"a.jpg": pred}, {}, {})
        self.assertEqual(len(t["frames"]), 1)
        self.assertEqual(t["frames"][0]["status"], "ok")
        self.assertEqual(t["repair"], [])

    def test_metadata_collision_is_prefixed(self):
        pred = make_result(objects=[obj("bad")])
        t = export_rows.export_rows({"a.jpg": pred}, {"a.jpg": {"status": "organizer", "ID": "7"}}, {})
        self.assertEqual(t["objects"][0]["meta_status"], "organizer")
        self.assertEqual(t["objects"][0]["ID"], "7")

    def test_legacy_bbox_none_becomes_frame(self):
        pred = {"objects": [{"type": "manhole", "bbox": None, "status": "bad", "damage": "cover_broken"}]}
        t = export_rows.export_rows({"a.jpg": pred}, {}, {})
        self.assertEqual(t["objects"], [])
        self.assertEqual(len(t["repair"]), 1)
        self.assertEqual(t["repair"][0]["scope"], "frame")


if __name__ == "__main__":
    unittest.main()
