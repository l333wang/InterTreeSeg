"""Training loop (migrated from the legacy ``train`` + ``main`` epoch loop)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from tqdm import tqdm

from ..metrics import AverageMeter, ConfusionMatrix
from .build import build_data_dict


@dataclass
class TrainResult:
    loss: float
    miou: float
    macc: float
    oa: float
    ious: np.ndarray
    accs: np.ndarray


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer,
    criterion,
    device,
    scheduler,
    cfg: Mapping[str, Any],
    epoch: int = 1,
) -> TrainResult:
    model.train()
    cm = ConfusionMatrix(num_classes=cfg["model"]["num_classes"], ignore_index=None)
    loss_meter = AverageMeter()
    grid_size = cfg["grid_size"]
    num_classes = int(cfg["model"]["num_classes"])

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, (points, labels) in pbar:
        data_dict = build_data_dict(points, grid_size, device)

        optimizer.zero_grad()
        logits = model(data_dict).view(-1, num_classes)
        labels = labels.view(-1).to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # per-batch step (matches legacy)

        cm.update(logits.argmax(dim=-1), labels)
        loss_meter.update(loss.item())
        pbar.set_description(
            f"Train Epoch [{epoch}/{cfg['epoch']}] "
            f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}"
        )

    miou, macc, oa, ious, accs = cm.all_metrics()
    return TrainResult(loss=loss_meter.avg, miou=miou, macc=macc, oa=oa, ious=ious, accs=accs)
