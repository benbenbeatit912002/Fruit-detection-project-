# Tabletop fruit detection

A PyTorch detector for apples, bananas, oranges and pears. The [original course notebook](fruit_detection_one_step.ipynb) uses a pretrained ResNet-18. This experiment replaces it with a [five-stage CNN](comparison/dual.py), keeps the two detection heads, and compares it with YOLO11n on the same photos.

| Detector | Starting weights | Validation mAP50-95 | ms / image |
| --- | --- | ---: | ---: |
| Custom CNN | Random | 22.4% | 22.97 |
| YOLO11n | Random | 83.4% | 25.63 |
| ResNet-18 | ImageNet | 56.7% | 26.93 |
| YOLO11n | COCO | 96.5% | 25.81 |

All four ran for 30 epochs on 218 training images. The same 10 validation images selected the checkpoints, so these are not test scores. Times are median end-to-end latency, including PNG decoding, on an RTX 5070 Laptop GPU at batch size one.

![Ground truth and the four detectors on a table with apples, bananas, pears and an orange](results/seed42-gpu/figures/val_08.jpg)

In image 8, the custom CNN finds only the orange at a confidence threshold of 0.30. YOLO11n trained from scratch finds seven of the eight fruit but labels the long pear at the lower right as a banana.

[All 10 images and per-class scores](results/seed42-gpu/measurements.md) · [Model and training notes](results/seed42-gpu/README.md)

## Running it

The [setup instructions](comparison/README.md) cover the dataset, CPU and CUDA environments, training, and evaluation. The original ResNet-18 [CPU run](results/seed42-cpu/README.md) is kept separately.

The dataset and starter notebook come from the [Wageningen University ABE course](https://git.wur.nl/abe-datasets/education/fruit-detection-challenge). The validation photos are shown with permission; the training data is not distributed here.
