"""CrossEntropy + Lovasz combined loss (migrated verbatim)."""

from __future__ import annotations

import torch.nn as nn

from .lovasz import LovaszLoss


class CombinedLoss(nn.Module):
    """Equal-weight sum of CrossEntropy and Lovasz-Softmax (both ignore_index=-1)."""

    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lovasz_loss = LovaszLoss(mode="multiclass", ignore_index=ignore_index)

    def forward(self, logits, labels):
        return self.cross_entropy_loss(logits, labels) + self.lovasz_loss(logits, labels)
