"""One FP32 validation path for model selection, checkpoint verification and timing."""

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.ops import batched_nms, box_convert, clip_boxes_to_image
from torchmetrics.detection import MeanAveragePrecision

from .prepare import CLASSES, ORIGINAL, load_split


SIZE, SCORE, IOU, MAX_DET = 512, 0.05, 0.70, 100


@dataclass
class Geometry:
    width: int
    height: int
    scaled_w: int
    scaled_h: int
    left: int = 0
    top: int = 0

    def original_boxes(self, boxes):
        offset = boxes.new_tensor([self.left, self.top] * 2)
        scale = boxes.new_tensor([self.width / self.scaled_w, self.height / self.scaled_h] * 2)
        return clip_boxes_to_image((boxes - offset) * scale, (self.height, self.width))


def preprocess(path, yolo, device):
    with Image.open(path) as source:
        image = source.convert("RGB")
    w, h = image.size
    if yolo:
        ratio = SIZE / max(w, h)
        sw, sh = round(w * ratio), round(h * ratio)
        left, top = (SIZE - sw) // 2, (SIZE - sh) // 2
        resized = cv2.resize(np.asarray(image), (sw, sh), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, top, SIZE-sh-top, left, SIZE-sw-left,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        tensor = TF.to_tensor(padded)
        geo = Geometry(w, h, sw, sh, left, top)
    else:
        tensor = TF.to_tensor(TF.resize(image, [SIZE, SIZE]))
        geo = Geometry(w, h, SIZE, SIZE)
    return tensor.unsqueeze(0).to(device), geo


def postprocess(boxes, probs, geo):
    scores, labels = probs.max(-1)
    selected = scores > SCORE
    boxes = geo.original_boxes(box_convert(boxes[selected], "cxcywh", "xyxy"))
    scores, labels = scores[selected], labels[selected]
    valid = (boxes[:, 2:] > boxes[:, :2]).all(1)
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    keep = batched_nms(boxes, scores, labels, IOU)[:MAX_DET]
    return {"boxes": boxes[keep].cpu(), "labels": labels[keep].cpu(), "scores": scores[keep].cpu()}


def predict(model, path, yolo, device):
    tensor, geo = preprocess(path, yolo, device)
    outputs = model(tensor)
    if yolo:
        # YOLO11 inference decodes DFL distances and sigmoid class scores before NMS.
        candidates = outputs[0][0].T
        boxes, probs = candidates.split([4, len(CLASSES)], dim=1)
    else:
        boxes = outputs["pred_boxes"][0] * SIZE
        probs = outputs["pred_objectness"][0] * outputs["pred_logits"][0].softmax(-1)
    return postprocess(boxes, probs, geo)


def val_items():
    for im, anns in load_split("val"):
        target = {
            "boxes": box_convert(torch.tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4), "xywh", "xyxy"),
            "labels": torch.tensor([a["category_id"] - 1 for a in anns], dtype=torch.long),
        }
        yield ORIGINAL / "val" / im["file_name"], target


def score(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Expected one prediction record per validation image")
    metric = MeanAveragePrecision(box_format="xyxy", backend="pycocotools", class_metrics=True,
                                  max_detection_thresholds=[1, 10, MAX_DET])
    metric.update(predictions, targets)
    values = metric.compute()
    return {"map": float(values["map"]), "map50": float(values["map_50"]), "map75": float(values["map_75"]),
            "per_class_ap": {CLASSES[int(c)]: float(ap) for c, ap in
                             zip(values["classes"].reshape(-1), values["map_per_class"].reshape(-1))}}


@torch.inference_mode()
def evaluate(model, yolo, device):
    model.eval()
    preds, targets, records = [], [], []
    for path, target in val_items():
        pred = predict(model, path, yolo, device)
        preds.append(pred)
        targets.append(target)
        records.append({"image": path.name, **{k: t.tolist() for k, t in pred.items()}})
    return score(preds, targets), records


@torch.inference_mode()
def benchmark(model, yolo, device):
    model.eval()
    paths = [path for path, _ in val_items()]
    tensors = [preprocess(p, yolo, device)[0] for p in paths]
    result = {"device": "cuda:0", "batch": 1, "input": [1, 3, SIZE, SIZE], "precision": "float32",
              "warmup_per_scope": 10, "timed_per_scope": 50}
    for scope in ("forward", "end_to_end"):
        samples = []
        for i in range(60):
            index = i % len(paths)
            torch.cuda.synchronize()
            start = perf_counter()
            if scope == "forward":
                model(tensors[index])
            else:
                predict(model, paths[index], yolo, device)
            torch.cuda.synchronize()
            if i >= 10:
                samples.append(1000 * (perf_counter() - start))
        result[scope] = {"median_ms": median(samples), "p90_ms": float(np.percentile(samples, 90)),
                         "mean_ms": sum(samples) / len(samples), "samples_ms": samples}
    return result
