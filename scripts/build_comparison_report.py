"""Export public validation measurements and all ten paired examples."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.ops import box_iou

from comparison.evaluation import val_items
from comparison.prepare import CLASSES, CONVERTED, ORIGINAL, PROJECT, load_split, save_json
from comparison.train import MODELS
from object_detection_utils import FileBasedAPCalculator


NAMES = {
    "resnet18-imagenet": "ResNet-18 / ImageNet",
    "custom-scratch": "Custom CNN / scratch",
    "yolo11n-scratch": "YOLO11n / scratch",
    "yolo11n-coco": "YOLO11n / COCO",
}
ORDER = ("custom-scratch", "yolo11n-scratch", "resnet18-imagenet", "yolo11n-coco")
COLORS = ["#14689a", "#9c4035", "#347251", "#80418a"]
DISPLAY = .30
ERROR_KEYS = ["correct", "missed", "wrong_class", "duplicate", "other_fp"]
TRAIN_FIELDS = ("seed", "epochs", "batch", "input_size", "training_amp", "accumulation")
EVAL_FIELDS = ("confidence_strict_gt", "classwise_nms_iou", "max_detections", "eval_precision")
BENCH_FIELDS = ("batch", "precision", "tf32", "warmup_per_scope", "timed_per_scope", "source_commit")


def read(path):
    return json.loads(path.read_bytes())


def display_predictions(record):
    return [(box, cls, conf) for box, cls, conf in zip(record["boxes"], record["labels"], record["scores"], strict=True)
            if conf >= DISPLAY]


def diagnose(target, detections):
    truth, classes = target["boxes"], target["labels"]
    used, counts = set(), Counter()
    for box, label, _ in sorted(detections, key=lambda row: -row[2]):
        ious = box_iou(torch.tensor([box]), truth)[0]
        overlaps = [i for i in ious.argsort(descending=True).tolist() if ious[i] >= .5]
        same = [i for i in overlaps if int(classes[i]) == label]
        free = next((i for i in same if i not in used), None)
        if free is not None:
            counts["correct"] += 1
            used.add(free)
        elif same:
            counts["duplicate"] += 1
        elif overlaps:
            counts["wrong_class"] += 1
        else:
            counts["other_fp"] += 1
    counts["missed"] = len(truth) - len(used)
    return {k: counts[k] for k in ERROR_KEYS}


def curves(output, runs):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    for name, color in zip(ORDER, COLORS):
        history = runs[name]["history"]
        epochs = [r["epoch"] for r in history]
        values = [r["map"] for r in history]
        axes[0].plot(epochs, values, label=NAMES[name], color=color)
        best = runs[name]["summary"]["selected_epoch"]
        axes[0].scatter(best, values[best - 1], color=color, s=24)
        ax = axes[2 if name.startswith("yolo") else 1]
        ax.plot(epochs, [r["loss"] for r in history], label=NAMES[name], color=color)
    for ax, title, label in zip(axes, ("Common validation AP", "Dual-scale detectors", "YOLO11n"),
                                ("mAP50-95", "Mean training loss", "Sum of mean box, class and DFL losses")):
        ax.set(title=title, xlabel="Epoch", ylabel=label, xlim=(1, 30))
        ax.grid(alpha=.2)
    axes[0].set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4, frameon=False, fontsize=9)
    fig.savefig(output / "curves.png", dpi=155)
    plt.close(fig)


def gallery(output, runs):
    fontfile = font_manager.findfont("DejaVu Sans")
    label_font = ImageFont.truetype(fontfile, 15)
    heading = ImageFont.truetype(fontfile, 20)
    colors = ["#ef695f", "#f7c646", "#ff944c", "#67d39d"]
    rows = []
    for index, (path, target) in enumerate(val_items(), 1):
        with Image.open(path) as src:
            image = src.convert("RGB")
        pw = 600
        ph = round(pw * image.height / image.width)
        tile = ph + 36
        canvas = Image.new("RGB", (pw * 3, tile * 2), "#f4f6f7")
        draw = ImageDraw.Draw(canvas)
        counts = {}

        def panel(col, row, title, detections):
            x, y = col * pw, row * tile
            draw.text((x + 8, y + 5), title, font=heading, fill="#172a33")
            img = image.resize((pw, ph), resample=Image.Resampling.LANCZOS)
            pen = ImageDraw.Draw(img)
            annotations = []
            for box, label, conf in detections:
                x1, y1, x2, y2 = (round(v * scale) for v, scale in zip(box, [pw/image.width, ph/image.height]*2))
                pen.rectangle((x1, y1, x2, y2), outline=colors[label], width=2)
                text = CLASSES[label] + ("" if conf is None else f" {conf:.2f}")
                annotations.append((x1, y1, x2, y2, label, text))
            occupied = []
            for x1, y1, x2, y2, label, text in annotations:
                tw = int(pen.textlength(text, font=label_font)) + 6
                candidates = [(max(0, min(x, pw - tw - 1)), max(0, min(y, ph - 20)))
                              for y in (y1-20, y1, y2-19, y2+1, y1-40, y1+20, y2+21)
                              for x in (x1, x2-tw, x1-tw-1, x2+1)]
                def overlap(pos):
                    x, y = pos
                    return sum(max(0, min(x+tw+1, r+1)-max(x, l)) * max(0, min(y+20, b+1)-max(y, t))
                               for l, t, r, b in occupied)
                tx, ty = min(candidates, key=overlap)
                rect = (tx, ty, tx+tw, ty+19)
                occupied.append(rect)
                pen.rectangle(rect, fill=colors[label])
                pen.text((tx+3, ty), text, fill="#141b20", font=label_font)
            canvas.paste(img, (x, y+36))

        panel(0, 0, f"{index:02d}  |  Ground truth", [(b, c, None) for b,c in
              zip(target["boxes"].tolist(), target["labels"].tolist())])
        for name, (col, row) in zip(ORDER, [(1,0), (2,0), (1,1), (2,1)]):
            record = runs[name]["predictions"][index - 1]
            if record["image"] != path.name:
                raise ValueError(f"Gallery order mismatch in {name}")
            shown = display_predictions(record)
            counts[name] = diagnose(target, shown)
            panel(col, row, NAMES[name], shown)
        draw.text((14, tile + 10), path.name, fill="#172a33", font=heading)
        draw.text((14, tile + 56), f"{len(target['boxes'])} labeled fruit   |   Display score >= 0.30",
                  fill="#172a33", font=heading)
        for n, name in enumerate(ORDER):
            c = counts[name]
            draw.text((14, tile+112+32*n), f"{NAMES[name]}: {c['correct']} matched, {c['missed']} missed",
                      font=label_font, fill="#283e48")
        canvas.save(output / f"val_{index:02d}.jpg", quality=91, subsampling=0)
        rows.append({"image": path.name, "objects": len(target["boxes"]), "models": counts})
    return rows


def crosscheck_archived_evaluator(run):
    """Re-score the same predictions using the corrected file-based CPU reference."""
    lines = []
    for (im, _), pred in zip(load_split("val"), run["predictions"], strict=True):
        if im["file_name"] != pred["image"]:
            raise ValueError("Unexpected prediction order")
        w, h = im["width"], im["height"]
        for (x1, y1, x2, y2), cls, conf in zip(pred["boxes"], pred["labels"], pred["scores"], strict=True):
            fields = [pred["image"], (x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h, cls+1, conf]
            lines.append(",".join(map(str, fields)))
    with tempfile.TemporaryDirectory() as tmp:
        text = Path(tmp) / "predictions.txt"
        text.write_text("\n".join(lines), encoding="utf-8")
        scores = FileBasedAPCalculator(ORIGINAL / "annotations/val.txt", text).calculate_map()
    metrics = {k: float(scores[col]) for k, col in (("map", "map"), ("map50", "map_50"), ("map75", "map_75"))}
    metrics["per_class_ap"] = {CLASSES[int(c)-1]: float(ap) for c, ap in zip(scores["classes"], scores["map_per_class"])}
    selected = run["summary"]
    differences = [abs(metrics[k] - selected[k]) for k in ("map", "map50", "map75")]
    differences.extend(abs(metrics["per_class_ap"][c] - selected["per_class_ap"][c]) for c in CLASSES)
    return dict(file_based_metrics=metrics, max_absolute_difference=max(differences))


def compact_metrics(runs, errors):
    reference = runs[ORDER[0]]
    config, provenance, bench = (reference[k] for k in ("config", "provenance", "benchmark"))
    models = {}
    for name in ORDER:
        run = runs[name]
        summary, timing = run["summary"], run["benchmark"]
        models[name] = {
            "initialization": run["config"]["initialization"],
            "source_commit": run["provenance"]["source_commit"],
            **{k: summary[k] for k in ("selected_epoch", "checkpoint_sha256", "parameters", "training_seconds",
                                       "peak_allocated_mib", "map", "map50", "map75", "per_class_ap")},
            "latency": {scope: {k: timing[scope][k] for k in ("median_ms", "p90_ms")}
                        for scope in ("forward", "end_to_end")},
            "display_counts": {k: sum(row["models"][name][k] for row in errors) for k in ERROR_KEYS},
        }
    return {
        "training": {k: config[k] for k in TRAIN_FIELDS},
        "evaluation": {**{k: config[k] for k in EVAL_FIELDS},
                       "display_confidence_gte": DISPLAY},
        "dataset": provenance["dataset"],
        "hardware": provenance["hardware"],
        "packages": provenance["packages"],
        "benchmark": {k: bench[k] for k in BENCH_FIELDS},
        "models": models,
    }


def export(local, output, diagnostics=None):
    runs = {name: {key: read(local / name / f"{key}.json") for key in
                   ("summary", "history", "config", "benchmark", "provenance", "predictions")} for name in MODELS}
    reference = runs[ORDER[0]]
    for name, run in runs.items():
        s, c, b = run["summary"], run["config"], run["benchmark"]
        for section, fields in (("config", TRAIN_FIELDS + EVAL_FIELDS), ("benchmark", BENCH_FIELDS),
                                ("provenance", ("dataset", "hardware", "packages"))):
            if any(run[section][k] != reference[section][k] for k in fields):
                raise ValueError(f"Inconsistent {section} across runs: {name}")
        if not (s["verified"] and s["actual_epochs"] == len(run["history"]) == 30 and
                [r["epoch"] for r in run["history"]] == list(range(1, c["epochs"]+1)) and
                s["selected_epoch"] == 1 + np.argmax([r["map"] for r in run["history"]]) and
                len(run["predictions"]) == 10 and c["seed"] == 42 and
                b["checkpoint_sha256"] == s["checkpoint_sha256"] and b["tf32"] is False):
            raise ValueError(f"Incomplete or inconsistent run: {name}")
        for scope in ("forward", "end_to_end"):
            samples = b[scope]["samples_ms"]
            if not (len(samples) == b["timed_per_scope"] == 50 and min(samples) > 0 and
                    np.isclose(np.median(samples), b[scope]["median_ms"]) and
                    np.isclose(np.percentile(samples, 90), b[scope]["p90_ms"])):
                raise ValueError(f"Inconsistent {scope} latency samples for {name}")
    crosscheck = {name: crosscheck_archived_evaluator(run) for name, run in runs.items()}
    for name, check in crosscheck.items():
        if not check["max_absolute_difference"] <= 1e-6:
            raise ValueError(f"Archived evaluator differs from common metrics for {name}: {check}")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    curves(figures, runs)
    errors = gallery(figures, runs)
    metrics = compact_metrics(runs, errors)
    save_json(output / "metrics.json", metrics)
    if diagnostics is not None:
        diagnostics.mkdir(parents=True, exist_ok=True)
        save_json(diagnostics / "display_errors.json", errors)
        save_json(diagnostics / "evaluator_crosscheck.json", crosscheck)
    shutil.copyfile(CONVERTED / "conversion.jpg", figures / "conversion.jpg")
    md = ["# Validation results", "",
          "All scores use the reloaded checkpoint selected on the ten validation images. AP is shown as a percentage.",
          "", "| Detector / initialization | Epoch | mAP50-95 | mAP50 | mAP75 | Apple AP | Banana AP | Orange AP | Pear AP |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        s = runs[name]["summary"]
        values = [s[k] for k in ("map","map50","map75")] + [s["per_class_ap"][c] for c in CLASSES]
        md.append(f"| {NAMES[name]} | {s['selected_epoch']} | " + " | ".join(f"{100*v:.1f}" for v in values) + " |")
    md += ["", "| Detector / initialization | Parameters | Train + validation (s) | Peak allocated (MiB) | Forward p50 / p90 (ms) | End-to-end p50 / p90 (ms) |",
           "|---|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        s, b = runs[name]["summary"], runs[name]["benchmark"]
        times = [f"{b[k]['median_ms']:.2f} / {b[k]['p90_ms']:.2f}" for k in ("forward", "end_to_end")]
        md.append(f"| {NAMES[name]} | {s['parameters']:,} | {s['training_seconds']:.1f} | {s['peak_allocated_mib']:.0f} | "
                  + " | ".join(times) + " |")
    md += ["", "Training time includes all epochs and their validation passes. Peak memory is PyTorch allocated memory during that interval. "
           "Latency uses one 512-square FP32 image on the same GPU. Each scope gets 10 warm-ups and 50 synchronized samples. "
           "Forward starts with a resident tensor. End-to-end includes warm-filesystem PNG read/decode, preprocessing, upload, model, class-wise NMS, and CPU results.",
           "", "![Training loss and validation AP for all four models](figures/curves.png)", "", "## Detections at confidence 0.30", "",
           "| Detector / initialization | Correct | Missed GT | Wrong class | Duplicate | Other FP |",
           "|---|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        counts = [metrics["models"][name]["display_counts"][k] for k in ERROR_KEYS]
        md.append("| " + NAMES[name] + " | " + " | ".join(map(str, counts)) + " |")
    md += ["", "At score >= 0.30, match boxes in descending confidence order at IoU >= 0.50. "
           "Each ground truth can match once. An extra match to the same class is a duplicate. "
           "An overlap with only another class is a wrong-class prediction. Other false positives include poorly localized boxes. "
           "Misses count all unmatched ground truths, including those with only wrong-class predictions.",
           "", "## Paired validation gallery", "",
           "Every detector sees the same ten images in their original COCO order. The upper pair compares the two scratch models. "
           "The lower pair shows transfer learning. Each model's displayed boxes use its own scores at the same fixed threshold. Validation photos reproduced with permission."]
    for i, row in enumerate(errors, 1):
        md += ["", f"### {i:02d}. {row['image']}", "", f"![Ground truth and four detectors, validation image {i}](figures/val_{i:02d}.jpg)"]
    (output / "measurements.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "runs/gpu-seed42")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/seed42-gpu")
    parser.add_argument("--diagnostics", type=Path, help="Optional local directory for per-image counts and evaluator checks")
    args = parser.parse_args()
    export(args.input, args.output, args.diagnostics)
