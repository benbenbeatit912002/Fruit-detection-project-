import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from comparison.dual import ObjectDetectorMultiScale, detection_loss
from comparison.evaluation import Geometry, postprocess, preprocess, score
from comparison.prepare import decode_rows, yolo_row
from object_detection_utils import HungarianMatcher


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_custom_stages_heads_backward(self):
        model = ObjectDetectorMultiScale(custom=True)
        image = torch.zeros(1, 3, 512, 512)
        shallow = model.backbone(image)
        self.assertEqual(shallow.shape, (1, 256, 32, 32))
        self.assertEqual(model.block_1(shallow).shape, (1, 512, 16, 16))
        outputs = model(image)
        self.assertEqual(outputs["pred_boxes"].shape, (1, 320, 4))
        self.assertEqual(outputs["pred_logits"].shape, (1, 320, 4))
        self.assertEqual(outputs["pred_objectness"].shape, (1, 320, 1))
        target = {"labels": torch.tensor([2]), "boxes": torch.tensor([[.5, .5, .4, .3]])}
        detection_loss(outputs, [target], HungarianMatcher()).backward()
        for component in (model.backbone, model.block_1, model.prediction_head, model.prediction_head_2):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters()))

    def test_empty_target_loss(self):
        model = ObjectDetectorMultiScale(custom=True).eval()
        outputs = model(torch.zeros(1, 3, 128, 128))
        target = {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}
        loss = detection_loss(outputs, [target], HungarianMatcher())
        self.assertTrue(torch.isfinite(loss))

    def test_strict_threshold_classwise_nms_global_cap(self):
        geo = Geometry(512, 512, 512, 512)
        boxes = torch.tensor([[50., 50., 10., 10.]] * 4)
        probs = torch.tensor([[.8, .1], [.7, .2], [.1, .9], [.05, .03]])
        selected = postprocess(boxes, probs, geo)
        self.assertEqual(selected["labels"].tolist(), [1, 0])
        self.assertEqual(len(postprocess(boxes[:1], probs[3:], geo)["boxes"]), 0)
        centers = torch.arange(1, 152).float()
        boxes = torch.stack([centers, centers, torch.ones(151), torch.ones(151)], dim=1)
        probs = torch.linspace(.1, .99, 151)[:, None]
        selected = postprocess(boxes, probs, geo)
        self.assertEqual(len(selected["scores"]), 100)
        torch.testing.assert_close(selected["scores"], probs[-100:, 0].flip(0))

    def test_inverse_geometry_stretch_and_odd_letterbox(self):
        xyxy = torch.tensor([[32., 4., 867., 538.]])
        for geo in (Geometry(1001, 561, 512, 512), Geometry(1001, 561, 512, 287, 0, 112)):
            scale = xyxy.new_tensor([geo.scaled_w / geo.width, geo.scaled_h / geo.height] * 2)
            padded = xyxy * scale + xyxy.new_tensor([geo.left, geo.top] * 2)
            torch.testing.assert_close(geo.original_boxes(padded), xyxy)

    def test_yolo_preprocessing_matches_official_letterbox(self):
        from ultralytics.data.augment import LetterBox
        pixels = np.random.default_rng(42).integers(0, 256, (561, 1001, 3), dtype=np.uint8)
        native = LetterBox(new_shape=(512, 512), auto=False)(image=pixels)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "odd.png"
            Image.fromarray(pixels).save(path)
            batch, geo = preprocess(path, yolo=True, device="cpu")
        self.assertEqual(geo.top, 112)
        np.testing.assert_array_equal((batch[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8"), native)

    def test_missing_detections_and_background_fp_reduce_common_ap(self):
        box = torch.tensor([[10., 10., 20., 20.]])
        target = {"boxes": box, "labels": torch.tensor([0])}
        pred = dict(target, scores=torch.tensor([.8]))
        empty = {"boxes": box[:0], "labels": torch.empty(0, dtype=torch.long)}
        blank = dict(empty, scores=torch.empty(0))
        self.assertAlmostEqual(score([pred], [target])["map"], 1.)
        self.assertLess(score([pred, blank], [target, target])["map"], .6)
        self.assertLess(score([pred, dict(pred, scores=torch.tensor([.9]))], [target, empty])["map"], .6)

    def test_yolo_roundtrip_all_classes_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "labels.txt"
            for i in range(1, 5):
                ann = {"bbox": [7.5, 0, 33., 20.], "category_id": i}
                row = yolo_row(ann, 77, 40)
                label.write_text(" ".join(map(str, row)))
                (cid, xywh), = decode_rows(label, 77, 40)
                self.assertEqual(cid, i - 1)
                for val, expected in zip(xywh, ann["bbox"]):
                    self.assertAlmostEqual(val, expected)
            with self.assertRaises(ValueError):
                yolo_row({"bbox": [64, 0, 40, 10], "category_id": 1}, 77, 40)


if __name__ == "__main__":
    unittest.main()
