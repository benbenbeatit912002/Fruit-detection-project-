"""Dual-scale detector derived from the project notebook at d48699f."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops import distance_box_iou_loss

from object_detection_utils import get_src_permutation_idx


def stage(in_channels, out_channels):
    """Downsample and mix channels, then refine each channel spatially."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


class PredictionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        hidden_channels = in_channels // 4

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.box_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(hidden_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=3, stride=1, padding=1),
        )

        self.objectness_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        cls_logits = self.classifier(x)
        bbox_pred = self.box_predictor(x)
        objectness_pred = self.objectness_predictor(x)

        grid_h, grid_w = x.shape[-2], x.shape[-1]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=x.device),
            torch.arange(grid_w, device=x.device),
            indexing="ij",
        )

        cx = (bbox_pred[:, 0, :, :] + grid_x) / grid_w
        cy = (bbox_pred[:, 1, :, :] + grid_y) / grid_h
        w = bbox_pred[:, 2, :, :] * 2
        h = bbox_pred[:, 3, :, :] * 2
        bbox_pred = torch.stack([cx, cy, w, h], dim=1)

        return cls_logits, bbox_pred, objectness_pred


class ObjectDetectorMultiScale(nn.Module):
    def __init__(self, n_classes=4, pretrained=False, custom=False):
        super().__init__()
        self.n_classes = n_classes

        if custom:
            if pretrained:
                raise ValueError("Custom backbone requires random initialization")
            self.backbone = nn.Sequential(
                stage(3, 24), stage(24, 48), stage(48, 96), stage(96, 256)
            )
            self.block_1 = stage(256, 512)
        else:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.backbone = nn.Sequential(*list(backbone.children())[:-3])
            self.block_1 = nn.Sequential(*list(backbone.children())[-3])

        self.prediction_head = PredictionHead(512, self.n_classes)
        self.prediction_head_2 = PredictionHead(256, self.n_classes)

    def forward(self, x):
        shallow_features = self.backbone(x)
        deep_features = self.block_1(shallow_features)

        cls_logits, bbox_pred, objectness_pred = self.prediction_head(deep_features)
        cls_preds = cls_logits.flatten(2, 3).permute(0, 2, 1)
        box_preds = bbox_pred.flatten(2, 3).permute(0, 2, 1)
        obj_preds = objectness_pred.flatten(2, 3).permute(0, 2, 1)

        cls_logits_2, bbox_pred_2, objectness_pred_2 = self.prediction_head_2(shallow_features)
        cls_preds_2 = cls_logits_2.flatten(2, 3).permute(0, 2, 1)
        box_preds_2 = bbox_pred_2.flatten(2, 3).permute(0, 2, 1)
        obj_preds_2 = objectness_pred_2.flatten(2, 3).permute(0, 2, 1)

        return {
            "pred_logits": torch.cat([cls_preds, cls_preds_2], dim=1),
            "pred_boxes": torch.cat([box_preds, box_preds_2], dim=1),
            "pred_objectness": torch.cat([obj_preds, obj_preds_2], dim=1),
        }


def cxcywh_to_xyxy_tensor(boxes):
    boxes = boxes.float()
    cx = boxes[:, 0].clamp(0.0, 1.0)
    cy = boxes[:, 1].clamp(0.0, 1.0)
    w = boxes[:, 2].clamp(min=1e-6, max=1.0)
    h = boxes[:, 3].clamp(min=1e-6, max=1.0)

    x1 = (cx - w / 2).clamp(0.0, 1.0)
    y1 = (cy - h / 2).clamp(0.0, 1.0)
    x2 = (cx + w / 2).clamp(0.0, 1.0)
    y2 = (cy + h / 2).clamp(0.0, 1.0)

    return torch.stack([x1, y1, x2, y2], dim=1)


def detection_loss(outputs, targets, matcher):
    # Keep assignment and regression in float32 when the backbone runs under AMP.
    outputs = {key: value.float() for key, value in outputs.items()}
    indices = matcher(outputs, targets)
    idx = get_src_permutation_idx(indices)

    num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
    src_logits = outputs["pred_logits"]
    src_boxes = outputs["pred_boxes"]
    src_objectness = outputs["pred_objectness"]

    target_classes_o = torch.cat(
        [t["labels"][J] for t, (_, J) in zip(targets, indices)]
    )
    target_classes = torch.full(
        src_logits.shape[:2],
        -100,
        dtype=torch.int64,
        device=src_logits.device,
    )
    target_classes[idx] = target_classes_o

    loss_ce = F.cross_entropy(
        src_logits.transpose(1, 2),
        target_classes,
        ignore_index=-100,
    ) if target_classes_o.numel() else src_logits.sum() * 0.0

    src_boxes_matched = src_boxes[idx]
    target_boxes = torch.cat(
        [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
    )
    loss_bbox = F.l1_loss(
        src_boxes_matched,
        target_boxes,
        reduction="none",
    ).sum() / num_boxes

    loss_diou = distance_box_iou_loss(
        cxcywh_to_xyxy_tensor(src_boxes_matched),
        cxcywh_to_xyxy_tensor(target_boxes),
        reduction="mean",
    ) if src_boxes_matched.numel() else src_boxes_matched.sum() * 0.0

    batch_idx, matched_pred_indices = idx
    target_objectness = torch.zeros(
        src_objectness.shape[0],
        src_objectness.shape[1],
        device=src_logits.device,
    )
    target_objectness[batch_idx, matched_pred_indices] = 1.0

    loss_objectness = F.binary_cross_entropy(src_objectness.squeeze(-1), target_objectness)

    return loss_ce + 5.0 * loss_bbox + 5.0 * loss_diou + 2.0 * loss_objectness
