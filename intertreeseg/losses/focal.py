"""Sigmoid/one-hot focal loss (migrated; fixes the ``loss.total()`` typo)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (https://arxiv.org/abs/1708.02002), sigmoid/one-hot form."""

    def __init__(self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1):
        super().__init__()
        assert reduction in ("mean", "sum"), "reduction should be 'mean' or 'sum'"
        assert isinstance(alpha, (float, list)), "alpha should be a float or list"
        assert isinstance(gamma, float), "gamma should be a float"
        assert isinstance(loss_weight, float), "loss_weight should be a float"
        assert isinstance(ignore_index, int), "ignore_index must be an int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # (N, C) expected; generalize from (B, C, d1, ...) if needed.
        pred = pred.transpose(0, 1)
        pred = pred.reshape(pred.size(0), -1)
        pred = pred.transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(0), "pred/target shape mismatch"

        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]
        if len(target) == 0:
            return pred.sum() * 0.0  # keeps a differentiable zero on the right device

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes).type_as(pred)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(self.gamma)

        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()  # was loss.total() -- not a tensor method
        return self.loss_weight * loss
