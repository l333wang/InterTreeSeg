"""Softmax Tversky loss (migrated verbatim)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """Multi-class Tversky loss over softmax probabilities."""

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, labels):
        """logits: ``[B*N, C]``; labels: ``[B*N]`` class indices."""
        probs = torch.softmax(logits, dim=-1)
        num_classes = logits.shape[-1]
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

        tp = torch.sum(probs * labels_one_hot, dim=0)
        fp = torch.sum(probs * (1 - labels_one_hot), dim=0)
        fn = torch.sum((1 - probs) * labels_one_hot, dim=0)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky_index).mean()
