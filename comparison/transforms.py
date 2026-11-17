"""Image transforms and COCO dataset used by the original detector."""

import json
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class DetectionTrainTransform:
    def __init__(
        self,
        img_size=(256, 256),
        hflip_prob=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ):
        self.img_size = img_size
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img, target):
        target_h, target_w = self.img_size

        scale = random.uniform(0.8, 1.2)
        new_h = int(target_h * scale)
        new_w = int(target_w * scale)
        img = TF.resize(img, (new_h, new_w))

        if scale >= 1.0:
            crop_top = (new_h - target_h) // 2
            crop_left = (new_w - target_w) // 2
            img = TF.crop(img, crop_top, crop_left, target_h, target_w)

            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                px_cx = boxes[:, 0] * new_w - crop_left
                px_cy = boxes[:, 1] * new_h - crop_top
                px_w = boxes[:, 2] * new_w
                px_h = boxes[:, 3] * new_h

                x1 = (px_cx - px_w / 2).clamp(0, target_w)
                y1 = (px_cy - px_h / 2).clamp(0, target_h)
                x2 = (px_cx + px_w / 2).clamp(0, target_w)
                y2 = (px_cy + px_h / 2).clamp(0, target_h)

                valid = (x2 - x1 > 2) & (y2 - y1 > 2)
                new_cx = ((x1 + x2) / 2) / target_w
                new_cy = ((y1 + y2) / 2) / target_h
                new_w_box = (x2 - x1) / target_w
                new_h_box = (y2 - y1) / target_h

                target["boxes"] = torch.stack(
                    [new_cx, new_cy, new_w_box, new_h_box], dim=1
                )[valid]
                target["labels"] = target["labels"][valid]
        else:
            pad_top = (target_h - new_h) // 2
            pad_left = (target_w - new_w) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_right = target_w - new_w - pad_left
            img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom])

            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                cx = (boxes[:, 0] * new_w + pad_left) / target_w
                cy = (boxes[:, 1] * new_h + pad_top) / target_h
                w = boxes[:, 2] * new_w / target_w
                h = boxes[:, 3] * new_h / target_h
                target["boxes"] = torch.stack([cx, cy, w, h], dim=1)

        if random.random() < self.hflip_prob:
            img = TF.hflip(img)
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]
                target["boxes"] = boxes

        img = TF.adjust_brightness(img, 1.0 + random.uniform(-self.brightness, self.brightness))
        img = TF.adjust_contrast(img, 1.0 + random.uniform(-self.contrast, self.contrast))
        img = TF.adjust_saturation(img, 1.0 + random.uniform(-self.saturation, self.saturation))
        img = TF.adjust_hue(img, random.uniform(-self.hue, self.hue))

        return TF.to_tensor(img), target


class DetectionEvalTransform:
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def __call__(self, img, target):
        img = TF.resize(img, self.img_size)
        return TF.to_tensor(img), target


class FruitDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_folder, img_size, transforms=None):
        self.img_folder = Path(img_folder)
        self.img_size = img_size
        self.transforms = transforms

        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        self.images = coco_data["images"]
        self.category_id_to_index = {
            category["id"]: index for index, category in enumerate(coco_data["categories"])
        }
        self.img_id_to_anns = {}
        for annotation in coco_data["annotations"]:
            self.img_id_to_anns.setdefault(annotation["image_id"], []).append(annotation)

        self.img_files = [str(self.img_folder / image["file_name"]) for image in self.images]

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        boxes = []
        labels = []
        for annotation in self.img_id_to_anns.get(img_info["id"], []):
            x, y, w, h = annotation["bbox"]
            boxes.append([(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h])
            labels.append(self.category_id_to_index[annotation["category_id"]])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            img = TF.resize(img, self.img_size)
            img = TF.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.images)
