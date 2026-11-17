"""Train a single fresh GPU run from the repository root."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
import torch
from torchvision.ops import nms

from object_detection_utils import HungarianMatcher
from .dual import ObjectDetectorMultiScale, detection_loss
from .evaluation import benchmark, evaluate, SIZE, SCORE, IOU, MAX_DET
from .prepare import CONVERTED, ORIGINAL, PROJECT, inspect_conversion, save_json, sha
from .transforms import DetectionTrainTransform, FruitDetectionDataset


MODELS = ("resnet18-imagenet", "custom-scratch", "yolo11n-scratch", "yolo11n-coco")
EPOCHS, BATCH, SEED = 30, 4, 42


def hardware():
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable: torch={torch.__version__}, compiled CUDA={torch.version.cuda}")
    gpu = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                      "--format=csv,noheader", "-i", "0"], text=True).strip()
    return {"name": gpu.name, "compute_capability": [gpu.major, gpu.minor], "memory_bytes": gpu.total_memory,
            "driver": driver, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cuda_available": True, "platform": platform.system(), "python": platform.python_version()}


def smoke():
    info = hardware()
    device = torch.device("cuda:0")
    a = torch.ones((8, 8), device=device)
    assert (a @ a).eq(8).all()
    boxes = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], device=device, dtype=torch.float32)
    assert nms(boxes, a[0, :2], 0.7).numel() == 1
    for custom in (False, True):
        model = ObjectDetectorMultiScale(custom=custom).to(device).train()
        x = torch.rand(2, 3, SIZE, SIZE, device=device)
        with torch.autocast("cuda"):
            outputs = model(x)
        assert outputs["pred_logits"].shape == (2, 320, 4)
        targets = [{"boxes": x.new_tensor([[.5, .4, .2, .3]]), "labels": torch.tensor([1], device=device)}] * 2
        loss = detection_loss(outputs, targets, HungarianMatcher())
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps(info), "tensor/forward/backward/NMS OK", flush=True)


class Run:
    def __init__(self, name, directory, audit, config):
        self.name, self.directory = name, directory
        self.history = []
        self.best = -1.
        self.best_epoch = None
        self.device = torch.device("cuda:0")
        self.yolo = name.startswith("yolo")
        self.config = dict(model=name, epochs=EPOCHS, seed=SEED, batch=BATCH, input_size=SIZE,
                           confidence_strict_gt=SCORE, classwise_nms_iou=IOU, max_detections=MAX_DET,
                           training_amp=True, eval_precision="float32", accumulation=1,
                           selection="maximum common validation mAP50-95; earliest tie", **config)
        sources = list((PROJECT / "comparison").glob("*.py")) + [
            PROJECT / "object_detection_utils.py", PROJECT / "scripts/run_notebook.py",
            PROJECT / "comparison/uv.lock", PROJECT / "comparison/data.yaml"]
        self.provenance = {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "reference_commit": "d48699f", "source_sha256_lf": {
                p.relative_to(PROJECT).as_posix(): __import__("hashlib").sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
                for p in sources},
            "hardware": hardware(), "packages": {p: version(p) for p in
                ("torch", "torchvision", "ultralytics", "torchmetrics", "pycocotools", "numpy", "Pillow",
                 "opencv-python", "scipy", "matplotlib")},
            "dataset": audit,
        }
        self.directory.mkdir(parents=True, exist_ok=False)
        save_json(self.directory / "config.json", self.config)
        save_json(self.directory / "provenance.json", self.provenance)

    def start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = perf_counter()

    def epoch(self, model, epoch, loss, rates):
        metrics, _ = evaluate(model, self.yolo, self.device)
        self.history.append(dict(epoch=epoch, loss=float(loss), lr=rates, **metrics))
        if metrics["map"] > self.best:
            self.best, self.best_epoch = metrics["map"], epoch
            tmp = self.directory / "best.tmp"
            torch.save(model.state_dict(), tmp)
            tmp.replace(self.directory / "best.pth")
        save_json(self.directory / "history.json", self.history)
        print(f"{self.name} {epoch:02}/{EPOCHS} loss={loss:.4f} mAP={metrics['map']:.4f}", flush=True)
        return metrics

    def finish(self, model):
        torch.cuda.synchronize()
        seconds = perf_counter() - self.started
        peak = torch.cuda.max_memory_allocated()
        if len(self.history) != EPOCHS:
            raise RuntimeError("Incomplete epoch history")
        path = self.directory / "best.pth"
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        metrics, records = evaluate(model, self.yolo, self.device)
        selected = self.history[self.best_epoch - 1]
        for key in ("map", "map50", "map75"):
            if abs(metrics[key] - selected[key]) > 1e-6:
                raise RuntimeError(f"Reloaded {key} differs from selected epoch")
        for name, value in metrics["per_class_ap"].items():
            if abs(value - selected["per_class_ap"][name]) > 1e-6:
                raise RuntimeError(f"Reloaded AP differs for {name}")
        save_json(self.directory / "predictions.json", records)
        timing = benchmark(model, self.yolo, self.device)
        save_json(self.directory / "benchmark.json", timing)
        summary = dict(model=self.name, selected_epoch=self.best_epoch, actual_epochs=len(self.history),
                       parameters=sum(p.numel() for p in model.parameters()),
                       training_seconds=seconds, peak_allocated_mib=peak / 2**20,
                       checkpoint_sha256=sha(path), verified=True, **metrics)
        save_json(self.directory / "summary.json", summary)
        print(json.dumps(summary), flush=True)


def collate(batch):
    return tuple(zip(*batch))


def train_dual(run):
    custom = run.name == "custom-scratch"
    device = run.device
    torch.hub.set_dir(str(PROJECT / "weights"))
    dataset = FruitDetectionDataset(ORIGINAL / "annotations/train.json", ORIGINAL / "train",
                                    (SIZE, SIZE), DetectionTrainTransform(img_size=(SIZE, SIZE)))
    batches = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0,
                                          pin_memory=True, collate_fn=collate)
    model = ObjectDetectorMultiScale(custom=custom, pretrained=not custom).to(device)
    if not custom:
        run.config["initial_weights_sha256"] = sha(PROJECT / "weights/checkpoints/resnet18-f37072fd.pth")
        save_json(run.directory / "config.json", run.config)
    backbone_lr = run.config["backbone_lr"]
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.block_1.parameters(), "lr": backbone_lr},
        {"params": model.prediction_head.parameters(), "lr": 1e-4},
        {"params": model.prediction_head_2.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    scaler, matcher = torch.amp.GradScaler("cuda"), HungarianMatcher()
    run.start()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.
        for images, targets in batches:
            x = torch.stack(images).to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(x)
            loss = detection_loss(outputs, targets, matcher)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        metrics = run.epoch(model, epoch, total / len(batches), [g["lr"] for g in optimizer.param_groups])
        scheduler.step(metrics["map"])
    run.finish(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("smoke", *MODELS))
    parser.add_argument("--output", type=Path, help="New local run directory")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.model == "smoke":
        smoke()
        return
    hardware()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    audit = inspect_conversion(CONVERTED)
    directory = args.output or PROJECT / "runs/gpu-seed42" / args.model
    if args.model.startswith("yolo"):
        from .yolo import train_yolo
        train_yolo(args.model, directory, audit)
    else:
        config = dict(initialization="random" if args.model == "custom-scratch" else "ImageNet-1K",
                      optimizer="AdamW", backbone_lr=1e-5, head_lr=1e-4, betas=[0.9, 0.999], weight_decay=1e-4,
                      scheduler="ReduceLROnPlateau(max, factor=0.5, patience=10)",
                      augmentation=dict(scale=[.8, 1.2], centered_crop_or_pad=True, hflip=.5,
                                        brightness=.2, contrast=.2, saturation=.2, hue=.05),
                      preprocessing="stretch to 512 square, RGB / 255", workers=0, shuffle=True,
                      loss="CE + 5 L1 + 5 DIoU + 2 BCE; float32 assignment and loss",
                      matcher_costs={"class": 1, "L1": 1, "GIoU": 1})
        train_dual(Run(args.model, directory, audit, config))


if __name__ == "__main__":
    main()
