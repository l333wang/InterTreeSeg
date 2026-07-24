"""Lovasz-Softmax loss (migrated; helpers made proper staticmethods)."""

from __future__ import annotations

from itertools import filterfalse
from typing import Optional

import torch
from torch.nn.modules.loss import _Loss

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


class LovaszLoss(_Loss):
    """Multi-class Lovasz-Softmax (only the multiclass path is used here)."""

    def __init__(
        self,
        mode: str = "multiclass",
        class_seen: Optional[int] = None,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.class_seen = class_seen
        self.loss_weight = loss_weight

    def forward(self, logits, labels):
        if self.mode == MULTICLASS_MODE:
            probs = logits.softmax(dim=1) if logits.dim() > 2 else logits.softmax(dim=-1)
            loss = self._lovasz_softmax(
                probs,
                labels,
                class_seen=self.class_seen,
                per_image=self.per_image,
                ignore=self.ignore_index,
            )
        else:
            raise ValueError(f"Unsupported mode {self.mode}.")
        return loss * self.loss_weight

    @staticmethod
    def _lovasz_softmax(probas, labels, classes="present", class_seen=None, per_image=False, ignore=None):
        if per_image:
            return LovaszLoss.mean(
                LovaszLoss._lovasz_softmax_flat(
                    *LovaszLoss._flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                    classes=classes,
                )
                for prob, lab in zip(probas, labels)
            )
        return LovaszLoss._lovasz_softmax_flat(
            *LovaszLoss._flatten_probas(probas, labels, ignore),
            classes=classes,
            class_seen=class_seen,
        )

    @staticmethod
    def _lovasz_softmax_flat(probas, labels, classes="present", class_seen=None):
        if probas.numel() == 0:
            return probas * 0.0
        C = probas.size(1)
        losses = []
        for c in labels.unique():
            if class_seen is not None and c not in class_seen:
                continue
            fg = (labels == c).type_as(probas)
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm.data]
            losses.append(torch.dot(errors_sorted, LovaszLoss._lovasz_grad(fg_sorted)))
        return LovaszLoss.mean(losses)

    @staticmethod
    def _flatten_probas(probas, labels, ignore=None):
        if probas.dim() == 3:
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        C = probas.size(1)
        probas = torch.movedim(probas, 1, -1).contiguous().view(-1, C)
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = labels != ignore
        return probas[valid], labels[valid]

    @staticmethod
    def isnan(x):
        return x != x

    @staticmethod
    def mean(values, ignore_nan=False, empty=0):
        """Nan-mean compatible with generators."""
        values = iter(values)
        if ignore_nan:
            values = filterfalse(LovaszLoss.isnan, values)
        try:
            n = 1
            acc = next(values)
        except StopIteration:
            if empty == "raise":
                raise ValueError("Empty mean")
            return empty
        for n, v in enumerate(values, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Gradient of the Lovasz extension w.r.t sorted errors (Alg. 1)."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
