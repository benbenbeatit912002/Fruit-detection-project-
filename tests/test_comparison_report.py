import unittest

import torch

from scripts.build_comparison_report import diagnose, display_predictions


class ReportTests(unittest.TestCase):
    def test_display_threshold_and_error_accounting(self):
        target = {"boxes": torch.tensor([[0.,0.,10.,10.], [20.,20.,30.,30.]]),
                  "labels": torch.tensor([0, 1])}
        preds = dict(boxes=[[0.,0.,10.,10.], [0.,0.,10.,10.], [20.,20.,30.,30.],
                            [40.,40.,50.,50.], [20.,20.,30.,30.]],
                     labels=[0, 0, 2, 0, 1], scores=[.9, .8, .7, .30, .29])
        self.assertEqual(diagnose(target, display_predictions(preds)),
                         dict(correct=1, missed=1, wrong_class=1, duplicate=1, other_fp=1))
