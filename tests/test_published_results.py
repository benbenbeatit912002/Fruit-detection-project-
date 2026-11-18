"""Check the report using only files that ship with the repository."""

import json
from pathlib import Path
from statistics import mean
import unittest


RESULTS = Path(__file__).resolve().parents[1] / "results"
GPU = RESULTS / "seed42-gpu"


class PublishedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads((GPU / "metrics.json").read_bytes())

    def test_same_split_as_original_notebook(self):
        historical = json.loads((RESULTS / "seed42-cpu/summary.json").read_bytes())["dataset"]
        for split in ("train", "val"):
            for key in ("images", "instances", "instances_per_class", "annotations_sha256", "image_manifest_sha256"):
                self.assertEqual(self.metrics["dataset"][split][key], historical[split][key])

    def test_summary_includes_all_four_detectors(self):
        models = self.metrics["models"]
        self.assertEqual(set(models), {"custom-scratch", "resnet18-imagenet", "yolo11n-scratch", "yolo11n-coco"})
        objects = self.metrics["dataset"]["val"]["instances"]
        for name, row in models.items():
            with self.subTest(model=name):
                self.assertAlmostEqual(row["map"], mean(row["per_class_ap"].values()), delta=1e-6)
                self.assertEqual(row["display_counts"]["correct"] + row["display_counts"]["missed"], objects)
                for scope in ("forward", "end_to_end"):
                    times = row["latency"][scope]
                    self.assertGreater(times["median_ms"], 0)
                    self.assertGreaterEqual(times["p90_ms"], times["median_ms"])

    def test_gallery_includes_every_validation_image_in_order(self):
        report = (GPU / "measurements.md").read_text(encoding="utf-8")
        total = self.metrics["dataset"]["val"]["images"]
        names = [f"figures/val_{i:02d}.jpg" for i in range(1, total + 1)]
        positions = [report.index(f"]({name})") for name in names]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(all((GPU / name).is_file() for name in names))
