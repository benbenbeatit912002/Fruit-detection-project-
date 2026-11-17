import tempfile
import unittest
from pathlib import Path

import torch

from object_detection_utils import FileBasedAPCalculator, HungarianMatcher


class EvaluationTests(unittest.TestCase):
    def evaluate(self, targets, predictions):
        with tempfile.TemporaryDirectory() as directory:
            truth_path = Path(directory) / "truth.txt"
            prediction_path = Path(directory) / "predictions.txt"
            truth_path.write_text(targets, encoding="utf-8")
            prediction_path.write_text(predictions, encoding="utf-8")
            return FileBasedAPCalculator(truth_path, prediction_path).calculate_map()

    def test_missing_predictions_reduce_recall(self):
        targets = "one.png,0.5,0.5,0.2,0.2,1\ntwo.png,0.5,0.5,0.2,0.2,1\n"
        predictions = "one.png,0.5,0.5,0.2,0.2,1,0.9\n"
        result = self.evaluate(targets, predictions)
        self.assertLess(float(result["map_50"]), 0.6)
        self.assertGreater(float(result["map_50"]), 0.4)

    def test_no_predictions_have_zero_ap(self):
        result = self.evaluate("one.png,0.5,0.5,0.2,0.2,1\n", "")
        self.assertEqual(float(result["map"]), 0.0)

    def test_false_positive_on_background_reduces_ap(self):
        targets = "one.png,0.5,0.5,0.2,0.2,1\n"
        predictions = "one.png,0.5,0.5,0.2,0.2,1,0.8\n"
        perfect = self.evaluate(targets, predictions)
        extra = self.evaluate(targets, predictions + "background.png,0.5,0.5,0.2,0.2,1,0.9\n")
        self.assertAlmostEqual(float(perfect["map"]), 1.0)
        self.assertLess(float(extra["map"]), float(perfect["map"]))

    def test_cpu_matching_and_empty_targets(self):
        outputs = {
            "pred_logits": torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]),
            "pred_boxes": torch.tensor([[[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]]),
        }
        matcher = HungarianMatcher()
        matched = matcher(outputs, [{"labels": torch.tensor([1]), "boxes": outputs["pred_boxes"][0, 1:]}])
        self.assertEqual(matched[0][0].device.type, "cpu")
        self.assertEqual(matched[0][0].tolist(), [1])
        empty = matcher(outputs, [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}])
        self.assertEqual(empty[0][0].numel(), 0)


if __name__ == "__main__":
    unittest.main()
