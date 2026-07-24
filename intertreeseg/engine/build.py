"""Factories: model input dict, optimizer, scheduler.

The scheduler factory takes ``steps_per_epoch`` as an argument, so OneCycleLR no
longer references an undefined ``train_loader`` (a latent NameError in the legacy
``main`` where the scheduler was built before the dataloader existed).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


def build_data_dict(points: torch.Tensor, grid_size: float, device) -> dict:
    """Split a ``(B, N, C)`` batch into the PTv3 input dict.

    ``coord`` = xyz (first 3 columns); ``feat`` = everything after (raw feature
    channels + click channels), whose width equals the model's ``in_channels``.
    """
    B1, N1, _ = points.shape
    feat = points[:, :, 3:].reshape(B1 * N1, -1)
    coord = points[:, :, :3].reshape(B1 * N1, 3)
    batch = torch.arange(B1, dtype=torch.long).unsqueeze(1).repeat(1, N1).view(-1)
    return {
        "feat": feat.to(device),
        "coord": coord.to(device),
        "batch": batch.to(device),
        "grid_size": grid_size,
    }


def build_optimizer(cfg: Mapping[str, Any], model: torch.nn.Module) -> optim.Optimizer:
    opt_cfg = cfg["optimizer"]
    otype = opt_cfg["type"]
    if otype == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
        )
    if otype == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg["weight_decay"],
            nesterov=opt_cfg.get("nesterov", False),
        )
    raise ValueError(f"Unsupported optimizer type: {otype}")


def build_scheduler(cfg: Mapping[str, Any], optimizer: optim.Optimizer, steps_per_epoch: int):
    sch_cfg = cfg["scheduler"]
    stype = sch_cfg["type"]
    if stype == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=sch_cfg["max_lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=cfg["epoch"],
            pct_start=sch_cfg["pct_start"],
            anneal_strategy=sch_cfg["anneal_strategy"],
            div_factor=sch_cfg["div_factor"],
            final_div_factor=sch_cfg["final_div_factor"],
        )
    if stype == "Cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg["epoch"],
            eta_min=cfg["optimizer"]["lr"] / 100,
        )
    raise ValueError(f"Unsupported scheduler type: {stype}")
