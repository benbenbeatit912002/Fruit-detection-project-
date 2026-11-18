# Fruit detection, seed 42

The main comparison is between two detectors trained from scratch. The custom CNN reached 22.4% validation mAP50-95 and YOLO11n reached 83.4%. With COCO weights, the same YOLO model reached 96.5%. The [results table and image gallery](measurements.md) include all four runs, in the same order for every photo.

## The custom backbone

The [CNN](../../comparison/dual.py) has five stages. Each starts with a stride-two 3x3 convolution, followed by a depthwise 3x3 and a pointwise 1x1 convolution. Each convolution uses batch normalization and SiLU. Keeping the expensive early feature maps narrow saves work; the last two stages produce exactly the channel counts expected by the notebook's detection heads.

| Stage | Channels | Feature map at 512 px | Input stride | Parameters |
| --- | ---: | ---: | ---: | ---: |
| 1 | 24 | 256 x 256 | 2 | 1,584 |
| 2 | 48 | 128 x 128 | 4 | 13,392 |
| 3 | 96 | 64 x 64 | 8 | 52,128 |
| 4, fine head input | 256 | 32 x 32 | 16 | 290,560 |
| 5, coarse head input | 512 | 16 x 16 | 32 | 1,449,472 |

The backbone has 1,807,136 parameters. The two heads add 2,436,242 and output a 16 x 16 grid and an 8 x 8 grid (320 candidates). The ResNet-18 run uses the same heads on an ImageNet-pretrained backbone. YOLO11n has its own three-scale head, assignment rule and losses. Its COCO run loaded 448 of 499 checkpoint entries when the head was adapted to four classes.

## Training and scoring

Seed 42, 30 epochs, 512 x 512 inputs, batch 4 and one optimizer step per batch. The training split has 218 photos and 1,603 boxes. The 10 validation photos contain 25 apples, 17 bananas, 15 oranges and 15 pears. Image and annotation fingerprints match the original CPU split. The [conversion check](figures/conversion.jpg) shows original COCO boxes beside boxes decoded from the YOLO labels.

| | Custom CNN and ResNet-18 | YOLO11n |
| --- | --- | --- |
| Optimizer | AdamW; backbone LR 0.00001, heads 0.0001, weight decay 0.0001 | AdamW; LR 0.001, weight decay 0.0005; clip norm 10 |
| Schedule | Halve LR after a validation plateau (patience 10) | Three warm-up epochs, then linear decay to 1% of the starting LR |
| Augmentation | Centered scale jitter (0.8-1.2), crop/pad, flip and color jitter | Mosaic through epoch 20, translation, scaling, flip and HSV jitter |
| Assignment and loss | Hungarian matching; CE + 5 L1 + 5 DIoU + 2 objectness BCE | Task-aligned assignment; box 7.5, class 0.5, DFL 1.5 |
| Validation | Stretch to a square; use trained weights | Letterbox to a square; use exponential moving average weights |

Training used FP16 autocast; the dual-scale detector's matching and loss ran in FP32. For validation, the [common evaluator](../../comparison/evaluation.py) maps both models' boxes back to the original pixels, accounting for the resize and any padding. It keeps scores above 0.05, applies class-wise NMS at IoU 0.70 and keeps at most 100 boxes per image. Each run reloaded the checkpoint with the best validation mAP50-95 (earliest epoch in a tie). Scoring the same predictions with the notebook's corrected evaluator gave identical AP values.

## Where the models struggled

At display confidence 0.30 and matching IoU 0.50, the custom CNN found 15 of 72 fruit. Orange was its best class (56.3% AP); banana and pear reached only 5.7% and 3.3%. Scratch YOLO matched 70 fruit. In [image 9](figures/val_09.jpg) it found all eight, but drew three duplicate boxes and two other false positives. The COCO model matched all 72 across the set, yet [image 5](figures/val_05.jpg) still shows pear boxes along the left edge, away from the labeled fruit.

The custom model took 3.75 ms for the forward pass versus 8.83 ms for scratch YOLO. Once PNG decoding, resizing, transfer and postprocessing were included, the difference fell to 2.67 ms per image. Both figures are medians on the same RTX 5070 Laptop GPU. [Latency percentiles, memory and training time](measurements.md)

## What to try next

The 10 validation images selected the checkpoints; there is no independent test score. Both scratch models peaked at epoch 27. The custom CNN also inherited a backbone learning rate chosen for fine-tuning. To isolate the backbone, train it and a randomly initialized ResNet-18 with the same heads and recipe. A new test set grouped by capture session and two additional seeds would be more useful than tuning to these ten photos. A fixed 60-epoch run for both scratch models could then check whether the original budget was too short.

[Training instructions](../../comparison/README.md) · [Metrics as JSON](metrics.json) · [Screenshots](screenshots/) · [Original CPU run](../seed42-cpu/README.md)
