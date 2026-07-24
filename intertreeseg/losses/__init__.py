"""Loss functions and the ``build_loss`` factory.

The factory and the CLI share :data:`LOSS_CHOICES`, so ``--loss`` can never
offer an option the factory doesn't handle (the legacy code listed ``focal`` as
a choice but had no branch for it, raising at runtime).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from .focal import FocalLoss
from .tversky import TverskyLoss
from .lovasz import LovaszLoss
from .combined import CombinedLoss

LOSS_CHOICES = ("crossentropy", "focal", "tversky", "combined")

__all__ = ["FocalLoss", "TverskyLoss", "LovaszLoss", "CombinedLoss", "build_loss", "LOSS_CHOICES"]


def build_loss(cfg: Mapping[str, Any]) -> nn.Module:
    """Build the training loss from ``cfg.loss`` (+ optional ``cfg.loss_params``)."""
    name = str(cfg.get("loss", "crossentropy")).lower()
    params = cfg.get("loss_params", {}) or {}

    if name == "crossentropy":
        return nn.CrossEntropyLoss(ignore_index=params.get("ignore_index", -1))
    if name == "focal":
        return FocalLoss(
            gamma=float(params.get("gamma", 2.0)),
            alpha=float(params.get("alpha", 0.5)),
            ignore_index=int(params.get("ignore_index", -1)),
        )
    if name == "tversky":
        return TverskyLoss(
            alpha=float(params.get("alpha", 0.7)),
            beta=float(params.get("beta", 0.3)),
        )
    if name == "combined":
        return CombinedLoss(ignore_index=int(params.get("ignore_index", -1)))
    raise ValueError(f"Unsupported loss '{name}'. Choose from {LOSS_CHOICES}.")
