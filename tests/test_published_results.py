"""Audit the published experiment records without CUDA, weights, or the course dataset."""

import json
from pathlib import Path
from statistics import median
import struct
import unittest

from scripts.capture_portfolio import case_study, overview, validation_set


ROOT = Path(__file__).resolve().parents[1] / "results" / "seed42-gpu"
MODELS = ("custom-scratch", "yolo11n-scratch", "resnet18-imagenet", "yolo11n-coco")


def read(path):
    return json.loads((ROOT / path).read_bytes())


class PublishedResultsTests(unittest.TestCase):
    def test_portfolio_screenshots_and_source_views(self):
        names = ("01-benchmark", "02-validation-set", "03-detection-example")
        for name in names:
            with (ROOT / "screenshots" / f"{name}.png").open("rb") as image:
                header = image.read(24)
            self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", header[16:24]), (1440, 900))
        results = overview()
        for name in MODELS:
            self.assertIn(f"{100 * read(f'{name}/summary.json')['map']:.1f}%", results)
        rows = read("display_errors.json")
        gallery = validation_set(rows)
        indices = [gallery.index(f'Ground truth for {row["image"]}') for row in rows]
        self.assertEqual(indices, sorted(indices))
        self.assertIn(rows[8]["image"], case_study(rows))

    def test_data_fingerprints_match_historical_split(self):
        historical = read(Path("../seed42-cpu") / "summary.json")["dataset"]
        for name in MODELS:
            with self.subTest(model=name):
                dataset = read(Path(name) / "provenance.json")["dataset"]
                self.assertEqual(dataset["exact_train_val_image_overlap"], 0)
                for split in ("train", "val"):
                    for field in ("images", "instances", "instances_per_class", "annotations_sha256", "image_manifest_sha256"):
                        self.assertEqual(dataset[split][field], historical[split][field])

    def test_selected_epochs_and_timing_samples(self):
        checks = read("evaluator_crosscheck.json")
        for name in MODELS:
            with self.subTest(model=name):
                model = Path(name)
                config, history, summary, bench = (read(model / f"{key}.json") for key in
                                                   ("config", "history", "summary", "benchmark"))
                self.assertEqual((config["seed"], config["batch"], config["input_size"]), (42, 4, 512))
                self.assertEqual([entry["epoch"] for entry in history], list(range(1, 31)))
                best = max(history, key=lambda record: record["map"])
                self.assertTrue(summary["verified"])
                self.assertEqual(summary["selected_epoch"], best["epoch"])
                for key in ("map", "map50", "map75"):
                    self.assertAlmostEqual(summary[key], best[key], delta=1e-6)
                self.assertLessEqual(checks[name]["max_absolute_difference"], 1e-6)
                self.assertEqual(summary["checkpoint_sha256"], bench["checkpoint_sha256"])
                self.assertEqual((bench["batch"], bench["precision"], bench["warmup_per_scope"]), (1, "float32", 10))
                for scope in ("forward", "end_to_end"):
                    samples = bench[scope]["samples_ms"]
                    self.assertEqual(len(samples), 50)
                    self.assertGreater(min(samples), 0)
                    self.assertEqual(median(samples), bench[scope]["median_ms"])

    def test_gallery_covers_all_models_and_images(self):
        rows = read("display_errors.json")
        images = [row["image"] for row in rows]
        self.assertEqual(len(set(images)), 10)
        self.assertEqual(sum(row["objects"] for row in rows), 72)
        for name in MODELS:
            self.assertEqual([r["image"] for r in read(Path(name) / "predictions.json")], images)
        for i, row in enumerate(rows, 1):
            self.assertTrue((ROOT / f"val_{i:02d}.jpg").is_file())
            for name in MODELS:
                counts = row["models"][name]
                self.assertEqual(counts["correct"] + counts["missed"], row["objects"])
