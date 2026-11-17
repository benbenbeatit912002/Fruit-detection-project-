"""Export measured validation results and figures from a completed notebook run."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
from torchmetrics.detection import MeanAveragePrecision

from object_detection_utils import FileBasedAPCalculator


def write_report(*, model, dataset, device, run_dir, nms_function, class_names,
                 history, best_epoch, seed, image_size, score_threshold,
                 nms_threshold, training_seconds):
    run_dir = Path(run_dir)
    model.eval()
    predictions = []
    rows = []
    latencies = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            batch = image.unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = perf_counter()
            outputs = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(perf_counter() - started)
            boxes = outputs["pred_boxes"][0].cpu()
            scores, labels = (outputs["pred_objectness"][0] * outputs["pred_logits"][0].softmax(-1)).max(-1)
            scores, labels = scores.cpu(), labels.cpu()
            keep = scores > score_threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = nms_function(boxes, labels, scores, iou_thresh=nms_threshold)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            predictions.append({"boxes": boxes, "scores": scores, "labels": labels, "target": target})
            for box, label, score in zip(boxes, labels, scores):
                values = [Path(dataset.img_files[index]).name, *box.tolist(), int(label) + 1, float(score)]
                rows.append(",".join(str(value) for value in values))

    prediction_path = run_dir / "best_validation_predictions.txt"
    prediction_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ground_truth_path = Path("fruit-detection-challenge/detection/annotations/val.txt")
    calculator = FileBasedAPCalculator(ground_truth_path, prediction_path)
    metrics = calculator.calculate_map()
    calculator.metric = MeanAveragePrecision(box_format="cxcywh", iou_thresholds=[0.5], class_metrics=True)
    metrics50 = calculator.calculate_map()
    selected = history[best_epoch - 1]
    if abs(float(metrics["map"]) - selected["map"]) > 1e-6:
        raise ValueError("Reloaded checkpoint does not reproduce the selected epoch's validation mAP.")

    class_results = {}
    for class_id, ap, ap50 in zip(metrics["classes"].reshape(-1),
                                 metrics["map_per_class"].reshape(-1),
                                 metrics50["map_per_class"].reshape(-1)):
        class_results[class_names[int(class_id) - 1]] = {"ap_50_95": float(ap), "ap_50": float(ap50)}
    audit = json.loads((run_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    summary = {
        "model": "ResNet-18 with two prediction heads",
        "weights_initialization": "torchvision ResNet18_Weights.DEFAULT",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "seed": seed,
        "epochs": len(history),
        "best_epoch": best_epoch,
        "image_size": list(image_size),
        "batch_size": 4,
        "score_threshold": score_threshold,
        "visualization_score_threshold": 0.30,
        "nms_iou_threshold": nms_threshold,
        "ap_iou_thresholds": [round(0.50 + 0.05 * index, 2) for index in range(10)],
        "max_detections_per_image": 100,
        "evaluation_split": "validation",
        "checkpoint_selection": "Highest validation mAP50-95 across the training run",
        "map_50_95": float(metrics["map"]),
        "map_50": float(metrics["map_50"]),
        "map_75": float(metrics["map_75"]),
        "per_class": class_results,
        "training_seconds": training_seconds,
        "mean_forward_ms_excluding_first_image": sum(latencies[1:]) / max(len(latencies) - 1, 1) * 1000,
        "timing_scope": "Batch-one model forward only; excludes decoding, transforms and NMS",
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.system(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "torchvision", "torchmetrics", "pycocotools", "numpy", "Pillow")},
        "dataset": audit,
        "notebook_code_sha256": hashlib.sha256("\n".join(
            "".join(cell["source"]) for cell in json.loads(Path("fruit_detection_one_step.ipynb").read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code").encode()).hexdigest(),
        "source_hash_format": "UTF-8 text with LF line endings",
        "source_sha256": {name: hashlib.sha256(Path(name).read_text(encoding="utf-8").encode()).hexdigest()
                          for name in ("object_detection_utils.py", "experiment_report.py", "scripts/run_notebook.py", "uv.lock")},
        "checkpoint_sha256": hashlib.sha256((run_dir / "best_fruit_detector.pth").read_bytes()).hexdigest(),
        "limitations": [f"One seed; {len(dataset)} validation images.",
                        "Validation data selects the checkpoint and is not an independent test set.",
                        "No comparison with another detector has been performed.",
                        "No small/medium/large-object AP is reported because boxes are normalized."],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_metrics(summary, history, run_dir)
    render_predictions(dataset, predictions, class_names, run_dir)
    return summary


def render_metrics(summary, history, output):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].plot(epochs, [row["loss"] for row in history], color="#2864b4")
    axes[0].set(xlabel="Epoch", ylabel="Mean batch loss", title="Training loss")
    for key, title, color in (("map", "mAP50-95", "#2864b4"), ("map_50", "mAP50", "#18835b"), ("map_75", "mAP75", "#b44b63")):
        axes[1].plot(epochs, [row[key] for row in history], label=title, color=color)
    axes[1].axvline(summary["best_epoch"], color="#666666", linestyle="--", linewidth=1, label="Selected epoch")
    validation = summary["dataset"]["val"]
    axes[1].set(xlabel="Epoch", ylabel="Average precision", ylim=(0, 1),
                title=f"Validation: {validation['images']} images / {validation['instances']} instances")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.savefig(output / "training_curves.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    names = list(summary["per_class"])
    for offset, key, label, color in ((-0.18, "ap_50_95", "AP50-95", "#2864b4"), (0.18, "ap_50", "AP50", "#18835b")):
        values = [summary["per_class"][name][key] for name in names]
        bars = ax.bar([i + offset for i in range(len(names))], values, width=0.36, label=label, color=color)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    ax.set(xticks=range(len(names)), xticklabels=names, ylim=(0, 1.1), ylabel="Average precision",
           title=f"Best validation checkpoint: epoch {summary['best_epoch']}")
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    fig.savefig(output / "per_class_ap.png", dpi=170)
    plt.close(fig)


def render_predictions(dataset, predictions, class_names, output):
    # Export every validation image in source order, without selecting favorable examples.
    for start in range(0, len(dataset), 4):
        rows = (min(4, len(dataset) - start) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 5 * rows), squeeze=False, layout="constrained")
        labels_by_axis = []
        fig.suptitle("Ground truth: green dashed | Predictions: magenta | Display score >= 0.30", fontsize=13)
        for index, ax in zip(range(start, start + 4), axes.flat):
            ax.axis("off")
            if index >= len(dataset):
                continue
            with Image.open(dataset.img_files[index]) as image:
                image = image.convert("RGB")
                width, height = image.size
                ax.imshow(image)
            item = predictions[index]
            label_positions = []
            for box in item["target"]["boxes"]:
                cx, cy, w, h = box.tolist()
                ax.add_patch(Rectangle(((cx - w / 2) * width, (cy - h / 2) * height),
                                       w * width, h * height, fill=False, color="#00b94c", linewidth=1.5, linestyle="--"))
            for box, label, score in zip(item["boxes"], item["labels"], item["scores"]):
                if score < 0.30:
                    continue
                cx, cy, w, h = box.tolist()
                x1, y1 = max(0, cx - w / 2) * width, max(0, cy - h / 2) * height
                x2, y2 = min(1, cx + w / 2) * width, min(1, cy + h / 2) * height
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#d32666", linewidth=1.8))
                label_positions.append((x1, y1, f"{class_names[int(label)]} {float(score):.2f}"))
            ax.set_title(f"Validation image {index + 1}", fontsize=11)
            labels_by_axis.append((ax, label_positions, width, height))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, positions, width, height in labels_by_axis:
            occupied = []
            for x1, y1, label in positions:
                text = ax.text(x1, y1, label, fontsize=8, va="bottom", color="white",
                               bbox={"facecolor": "#a21c50", "alpha": 0.9, "edgecolor": "none", "pad": 2})
                text.set_in_layout(False)
                candidates = [(max(2, min(x1 + dx, width - 155)), max(24, min(y1 + dy, height - 2)))
                              for dy in (-4, -34, 26, -64, 56, -94, 86)
                              for dx in (0, 150, -150)]
                for x, y in candidates:
                    text.set_position((x, y))
                    extent = text.get_window_extent(renderer).padded(4)
                    if not any(extent.overlaps(previous) for previous in occupied):
                        break
                occupied.append(extent)
                if abs(y - y1) > 8 or abs(x - x1) > 8:
                    ax.plot([x1, x], [y1, y], color="#a21c50", linewidth=0.7)
        fig.savefig(output / f"validation_examples_{start // 4 + 1:02d}.png", dpi=150)
        plt.close(fig)
