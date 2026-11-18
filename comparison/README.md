# Running the detectors

Run these commands from the repository root. The comparison has its own uv lockfile for Windows x64 and Python 3.11. The original notebook uses the lockfile in the root.

## Environment

```powershell
uv sync --project comparison --locked --python 3.11.15 --extra cu128
$py = ".\comparison\.venv\Scripts\python.exe"
& $py -m comparison.train smoke
& $py -m unittest discover -s tests
```

The recorded runs used torch 2.10.0, torchvision 0.25.0, CUDA 12.8 and NVIDIA driver 581.42. The smoke test runs tensor operations, NMS and forward/backward passes on CUDA. For conversion, report generation or tests on a CPU machine, select `--extra cpu` instead and omit the CUDA smoke test. The four training commands require CUDA. [PyTorch wheel pairs](https://pytorch.org/get-started/previous-versions/) · [NVIDIA driver compatibility](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)

## Data and starting weights

Place the course files in this layout:

```text
fruit-detection-challenge/detection/
  train/
  val/
  annotations/
    train.json
    val.json
    val.txt
```

```powershell
& $py -m comparison.prepare
curl.exe --fail --location --create-dirs https://download.pytorch.org/models/resnet18-f37072fd.pth --output weights/checkpoints/resnet18-f37072fd.pth
curl.exe --fail --location --create-dirs https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt --output weights/yolo11n.pt
```

Preparation checks the split against the original run and writes YOLO labels under `datasets/fruit-yolo/`. Use `comparison.prepare --verify` for an existing conversion. The YOLO weights are also needed for Ultralytics' AMP check; the scratch training model itself is initialized from YAML.

## Training and evaluation

```powershell
& $py -m comparison.train resnet18-imagenet
& $py -m comparison.train custom-scratch
& $py -m comparison.train yolo11n-scratch
& $py -m comparison.train yolo11n-coco
& $py -m comparison.measure
& $py -m scripts.build_comparison_report --output runs/report
```

Each trainer writes to a new directory under `runs/gpu-seed42/`. For a separate experiment pass `--output <new-run-directory>` to the trainer, then use `comparison.measure --root <run-parent>` and `scripts.build_comparison_report --input <run-parent>`. The run directories contain checkpoints, training arguments, epoch histories, box predictions and individual timing samples. The report exports a [metrics summary](../results/seed42-gpu/metrics.json), tables and figures. Add `--diagnostics runs/diagnostics` to the report command for per-image error counts and the evaluator cross-check.

## Original notebook

Use `uv sync --locked` and `uv run jupyter lab` in the repository root to open [the notebook](../fruit_detection_one_step.ipynb). To execute the notebook and save its outputs to a new directory, run `uv run python scripts/run_notebook.py --output runs/notebook-run`.

The dual-scale prediction heads and Hungarian matching code come from the original project. The ResNet-18 backbone is from [TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). The [YOLO11n detector](https://docs.ultralytics.com/models/yolo11/) uses [Ultralytics 8.3.253](https://github.com/ultralytics/ultralytics/tree/v8.3.253) under its [AGPL-3.0 / enterprise licensing terms](https://www.ultralytics.com/license).
