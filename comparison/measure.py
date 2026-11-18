"""Measure selected FP32 checkpoints under identical inference backend settings."""

import argparse
import json
from pathlib import Path
import subprocess

import torch

from .train import MODELS, hardware
from .dual import ObjectDetectorMultiScale
from .evaluation import benchmark
from .prepare import PROJECT, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "runs/gpu-seed42")
    args = parser.parse_args()
    info = hardware()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    for name in MODELS:
        directory = args.root / name
        summary = json.loads((directory / "summary.json").read_bytes())
        checkpoint = directory / "best.pth"
        if sha(checkpoint) != summary["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {name}")
        yolo = name.startswith("yolo")
        if yolo:
            from ultralytics.nn.tasks import DetectionModel
            model = DetectionModel("yolo11n.yaml", nc=4, verbose=False)
        else:
            model = ObjectDetectorMultiScale(custom=name == "custom-scratch")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device).eval()
        timing = benchmark(model, yolo, device)
        timing.update(hardware=info, tf32=False, cudnn_benchmark=False, cudnn_deterministic=False,
                      cpu_threads=torch.get_num_threads(), checkpoint_sha256=summary["checkpoint_sha256"],
                      source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
        save_json(directory / "benchmark.json", timing)
        print(name, {k: timing[k]["median_ms"] for k in ("forward", "end_to_end")}, flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
