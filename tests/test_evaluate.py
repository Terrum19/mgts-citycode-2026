import unittest

from src.evaluate import evaluate, match_centers


def obj(bbox, *, kind="manhole", status="bad"):
    return {"bbox": bbox, "type": kind, "status": status, "damage": None}


class CenterMatchingTests(unittest.TestCase):
    def test_nested_box_counts_as_repair_but_not_at_iou_50(self):
        labels = [{"file": "photo.jpg", "objects": [obj([0, 0, 100, 100])]}]
        preds = {"photo.jpg": {"objects": [obj([25, 20, 75, 80])]}}

        result = evaluate(labels, preds)
        self.assertEqual(result["matching"]["method"], "center")
        self.assertEqual((result["repair_tp"], result["repair_fp"], result["repair_fn"]), (1, 0, 0))
        self.assertEqual(result["det_recall@0.5"], 0.0)
        self.assertEqual(result["det_recall@0.3"], 1.0)

        strict = evaluate(labels, preds, thr=0.5)
        self.assertEqual(strict["matching"], {"method": "iou", "threshold": 0.5})
        self.assertEqual((strict["repair_tp"], strict["repair_fp"], strict["repair_fn"]), (0, 1, 1))

    def test_rejects_distant_centers_and_weak_overlap(self):
        gt = [obj([0, 0, 60, 80])]
        self.assertEqual(match_centers(gt, [obj([25, 0, 85, 80])]), [(0, 0)])
        self.assertEqual(match_centers(gt, [obj([26, 0, 86, 80])]), [])
        self.assertEqual(match_centers(gt, [obj([25, 35, 35, 45])]), [])
        self.assertEqual(match_centers([obj([0, 0, 10, 200])], [obj([10, 0, 20, 200])]), [])

    def test_uses_nearest_center_and_does_not_count_duplicate(self):
        gt = [obj([0, 0, 100, 100])]
        pred = [obj([10, 0, 110, 100]), obj([25, 20, 75, 80])]
        self.assertEqual(match_centers(gt, pred), [(0, 1)])
        result = evaluate([{"file": "photo.jpg", "objects": gt}], {"photo.jpg": {"objects": pred}})
        self.assertEqual((result["repair_tp"], result["repair_fp"], result["repair_fn"]), (1, 1, 0))

    def test_wrong_type_and_review_do_not_become_repairs(self):
        box = [0, 0, 60, 80]
        labels = [{"file": "photo.jpg", "objects": [obj(box)]}]
        for candidate, det_recall, repair_fp in [(obj(box, kind="pole"), 0.0, 1),
                                                 (obj(box, status="review"), 1.0, 0)]:
            with self.subTest(candidate=candidate):
                result = evaluate(labels, {"photo.jpg": {"objects": [candidate]}})
                self.assertEqual(result["det_recall"], det_recall)
                self.assertEqual(result["det_recall_anytype"], 1.0)
                self.assertEqual((result["repair_tp"], result["repair_fp"], result["repair_fn"]),
                                 (0, repair_fp, 1))
