"""Execute the notebook in the current Python environment and retain local artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def audit_dataset():
    from collections import Counter
    from PIL import Image

    result = {}
    split_hashes = {}
    categories = None
    for split in ("train", "val"):
        annotation_path = ROOT / "fruit-detection-challenge/detection/annotations" / f"{split}.json"
        raw = annotation_path.read_bytes()
        data = json.loads(raw)
        mapping = [(item["id"], item["name"]) for item in data["categories"]]
        if categories is not None and categories != mapping:
            raise ValueError("Training and validation categories must have the same order.")
        categories = mapping
        image_ids = {item["id"] for item in data["images"]}
        counts = Counter(annotation["category_id"] for annotation in data["annotations"])
        digests = []
        for item in data["images"]:
            image_path = ROOT / "fruit-detection-challenge/detection" / split / item["file_name"]
            with Image.open(image_path) as image:
                if image.size != (item["width"], item["height"]):
                    raise ValueError("Image dimensions do not match annotations.")
                image.verify()
            digests.append(hashlib.sha256(image_path.read_bytes()).hexdigest())
        for annotation in data["annotations"]:
            if annotation["image_id"] not in image_ids or annotation["category_id"] not in dict(mapping):
                raise ValueError("Annotation references an unknown image or category.")
            if min(annotation["bbox"][2:]) <= 0:
                raise ValueError("Found a zero-area annotation.")
        split_hashes[split] = set(digests)
        result[split] = {
            "images": len(data["images"]),
            "instances": len(data["annotations"]),
            "instances_per_class": {name: counts[category_id] for category_id, name in mapping},
            "annotations_sha256": hashlib.sha256(raw).hexdigest(),
            "image_manifest_sha256": hashlib.sha256("".join(sorted(digests)).encode()).hexdigest(),
        }
    overlap = len(split_hashes["train"] & split_hashes["val"])
    result["exact_train_val_image_overlap"] = overlap
    if overlap:
        raise ValueError("Identical image files occur in training and validation.")
    images_by_id = {image["id"]: image for image in data["images"]}
    expected = []
    for annotation in data["annotations"]:
        image = images_by_id[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        expected.append((image["file_name"], annotation["category_id"],
                         (x + width / 2) / image["width"], (y + height / 2) / image["height"],
                         width / image["width"], height / image["height"]))
    actual = []
    for line in (annotation_path.parent / "val.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            parts = [part.strip() for part in line.split(",")]
            actual.append((parts[0], int(parts[5]), *map(float, parts[1:5])))
    if len(expected) != len(actual):
        raise ValueError("Validation JSON and text contain different instance counts.")
    for first, second in zip(sorted(expected), sorted(actual)):
        if first[:2] != second[:2] or max(abs(a - b) for a, b in zip(first[2:], second[2:])) > 1e-5:
            raise ValueError("Validation JSON and text annotations disagree.")
    result["validation_json_matches_text"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/seed42-cpu")
    parser.add_argument("--preflight", action="store_true", help="Check data and time three training batches.")
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path(args.output).resolve()
    if (output / "history.json").exists() or (output / "executed.ipynb").exists():
        raise FileExistsError("Choose a new output directory to preserve the previous run.")
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache"
    os.environ["FRUIT_RUN_DIR"] = str(output)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(cache / "jupyter/runtime")
    os.environ["IPYTHONDIR"] = str(cache / "ipython")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    for name in ("JUPYTER_RUNTIME_DIR", "IPYTHONDIR", "MPLCONFIGDIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    audit = audit_dataset()
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    notebook = nbformat.read(ROOT / "fruit_detection_one_step.ipynb", as_version=4)
    if args.preflight:
        notebook.cells = notebook.cells[:9]
        notebook.cells.append(nbformat.v4.new_code_cell("""
from time import perf_counter
model.train()
batch_times = []
for batch_index, (images, targets) in enumerate(train_loader):
    if batch_index == 3:
        break
    started = perf_counter()
    model.zero_grad(set_to_none=True)
    outputs = model(torch.stack(images).to(device))
    loss = sum(value.float().square().mean() for value in outputs.values())
    loss.backward()
    batch_times.append(perf_counter() - started)
print({"device": str(device), "batch_seconds": batch_times,
       "estimated_training_minutes": sum(batch_times[1:]) / 2 * len(train_loader) * NUM_EPOCHS / 60})
"""))
    else:
        notebook.cells.append(nbformat.v4.new_code_cell("""
from experiment_report import write_report
summary = write_report(
    model=model, dataset=val_dataset, device=device, run_dir=RUN_DIR,
    nms_function=apply_classwise_nms, class_names=CLASS_NAMES,
    history=epoch_history, best_epoch=best_epoch, seed=SEED,
    image_size=IMAGE_SIZE, score_threshold=BEST_SCORE_THRESH,
    nms_threshold=BEST_NMS_THRESH, training_seconds=training_seconds,
)
print(json.dumps(summary, indent=2))
"""))

    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=None, record_timing=False,
                            resources={"metadata": {"path": str(ROOT)}})

    def started(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Running cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    def finished(cell, cell_index, **kwargs):
        for item in cell.get("outputs", []):
            if item.output_type == "stream" and item.name == "stdout":
                print(item.text[-3000:], flush=True)
        nbformat.write(notebook, output / "executed.ipynb")

    client.on_cell_start = started
    client.on_cell_executed = finished
    try:
        client.execute(cleanup_kc=True)
    finally:
        nbformat.write(notebook, output / "executed.ipynb")
    print(f"Completed: {output.name}", flush=True)


if __name__ == "__main__":
    main()
