"""Convert only the authorized training and validation splits and verify every box."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from PIL import Image, ImageDraw

from scripts.run_notebook import audit_dataset


PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL = PROJECT / "fruit-detection-challenge" / "detection"
CONVERTED = PROJECT / "datasets" / "fruit-yolo"
CLASSES = ("apple", "banana", "orange", "pear")
CLASS_ID = {1: 0, 2: 1, 3: 2, 4: 3}


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(split):
    data = json.loads((ORIGINAL / "annotations" / f"{split}.json").read_bytes())
    if [(c["id"], c["name"]) for c in data["categories"]] != list(enumerate(CLASSES, 1)):
        raise ValueError("Unexpected COCO category IDs, names or order")
    images = data["images"]
    ids = {im["id"] for im in images}
    if len(ids) != len(images) or len({Path(im["file_name"]).stem.casefold() for im in images}) != len(images):
        raise ValueError("Repeated image ID or ambiguous label filename")
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] not in ids or ann["category_id"] not in CLASS_ID:
            raise ValueError("Unknown annotation foreign key")
        if ann.get("iscrowd") or ann.get("ignore"):
            raise ValueError("YOLO text cannot represent crowd/ignored objects")
        annotations[ann["image_id"]].append(ann)
    return [(im, annotations[im["id"]]) for im in images]


def yolo_row(ann, width, height):
    x, y, w, h = ann["bbox"]
    if not (all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0
            and 0 <= x < x + w <= width and 0 <= y < y + h <= height):
        raise ValueError(f"Invalid COCO box {ann['bbox']}")
    return CLASS_ID[ann["category_id"]], (x + w / 2) / width, (y + h / 2) / height, w / width, h / height


def decode_rows(path, width, height):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label, cx, cy, w, h = line.split()
        cx, cy, w, h = map(float, (cx, cy, w, h))
        rows.append((int(label), ((cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height)))
    return rows


def inspect_conversion(directory, write=False):
    audit = audit_dataset()
    reference = json.loads((PROJECT / "results/seed42-cpu/summary.json").read_bytes())["dataset"]
    for split in ("train", "val"):
        fields = ("images", "instances", "instances_per_class", "image_manifest_sha256")
        if any(audit[split][k] != reference[split][k] for k in fields):
            raise ValueError(f"{split} does not match the recorded reference split")
        images_dir, labels_dir = directory / "images" / split, directory / "labels" / split
        if write:
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
        expected_images, expected_labels = set(), set()
        labels_hash = hashlib.sha256()
        max_error = 0.0
        for im, anns in load_split(split):
            name, width, height = im["file_name"], im["width"], im["height"]
            if Path(name).name != name:
                raise ValueError("Expected a simple image filename")
            original = ORIGINAL / split / name
            copy, label = images_dir / name, labels_dir / Path(name).with_suffix(".txt")
            expected_images.add(name)
            expected_labels.add(label.name)
            rows = [yolo_row(ann, width, height) for ann in anns]
            if len(set(rows)) != len(rows):
                raise ValueError("Identical labels would be dropped by the YOLO loader")
            if write:
                shutil.copyfile(original, copy)
                label.write_text("".join(f"{c} {x:.10f} {y:.10f} {w:.10f} {h:.10f}\n"
                                         for c, x, y, w, h in rows), encoding="utf-8")
            if sha(copy) != sha(original):
                raise ValueError(f"Converted image differs from {name}")
            labels_hash.update(label.name.encode() + b"\0" + label.read_text(encoding="utf-8").encode())
            for ann, (cid, box) in zip(anns, decode_rows(label, width, height), strict=True):
                if cid != CLASS_ID[ann["category_id"]]:
                    raise ValueError(f"Class mismatch in {name}")
                max_error = max(max_error, *(abs(v - w) for v, w in zip(ann["bbox"], box)))
        if max_error > 1e-5 or expected_images != {f.name for f in images_dir.iterdir()} or expected_labels != {f.name for f in labels_dir.iterdir()}:
            raise ValueError(f"YOLO conversion mismatch in {split}")
        audit[split].update(yolo_labels_sha256=labels_hash.hexdigest(), max_roundtrip_error_px=max_error)
    audit["category_ids_coco_to_yolo"] = CLASS_ID
    if (directory / "data.yaml").read_bytes() != Path(__file__).with_name("data.yaml").read_bytes():
        raise ValueError("Dataset YAML differs from the checked-in configuration")
    return audit


def draw_audit(directory):
    im, anns = load_split("val")[0]
    name, w, h = im["file_name"], im["width"], im["height"]
    panels = []
    for source, boxes in [
        (ORIGINAL / "val", [(CLASS_ID[a["category_id"]], a["bbox"]) for a in anns]),
        (directory / "images/val", decode_rows(directory / "labels/val" / Path(name).with_suffix(".txt"), w, h)),
    ]:
        with Image.open(source / name) as image:
            panel = image.convert("RGB")
        pen = ImageDraw.Draw(panel)
        for c, (x, y, bw, bh) in boxes:
            bounds = tuple(round(v) for v in (x, y, x + bw, y + bh))
            pen.rectangle(bounds, outline="#22ff66", width=4)
            pen.text(bounds[:2], CLASSES[c], fill="white", stroke_width=2, stroke_fill="black")
        panels.append(panel.resize((640, round(640 * h / w))))
    canvas = Image.new("RGB", (1280, panels[0].height + 30), "white")
    for i, title in enumerate(("COCO pixel boxes", "Decoded YOLO text boxes")):
        canvas.paste(panels[i], (640 * i, 30))
        ImageDraw.Draw(canvas).text((640 * i + 8, 8), title, fill="black")
    canvas.save(directory / "conversion.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check an existing conversion")
    args = parser.parse_args()
    if args.verify:
        audit = inspect_conversion(CONVERTED)
    else:
        if CONVERTED.exists():
            raise FileExistsError("Converted dataset exists. Use --verify to check it.")
        CONVERTED.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CONVERTED.parent) as tmp:
            staging = Path(tmp) / "fruit-yolo"
            staging.mkdir()
            shutil.copyfile(Path(__file__).with_name("data.yaml"), staging / "data.yaml")
            audit = inspect_conversion(staging, write=True)
            draw_audit(staging)
            save_json(staging / "audit.json", audit)
            staging.rename(CONVERTED)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
