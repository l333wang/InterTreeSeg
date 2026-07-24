"""Evaluation loop returning a structured :class:`EvalResult`.

The legacy ``test()`` returned a 9-tuple that callers unpacked by position — one
site (``onlyTest``) unpacked only 8 values and crashed. A dataclass with named
fields removes that whole class of bug.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..metrics import AverageMeter, ConfusionMatrix
from .build import build_data_dict


@dataclass
class EvalResult:
    pred: np.ndarray            # (num_blocks, num_point) argmax predictions
    logits: np.ndarray          # (num_blocks, num_point, num_classes)
    loss: float
    miou: float
    macc: float
    oa: float
    ious: np.ndarray
    accs: np.ndarray
    elapsed: float


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device,
    cfg: Mapping[str, Any],
) -> EvalResult:
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg["model"]["num_classes"], ignore_index=None)
    loss_meter = AverageMeter()
    num_point = int(cfg["data"]["num_point"])
    num_classes = int(cfg["model"]["num_classes"])
    grid_size = cfg["grid_size"]

    pre_list, logits_list = [], []
    start = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, (points, labels) in pbar:
        data_dict = build_data_dict(points, grid_size, device)
        logits = model(data_dict).view(-1, num_classes)
        labels = labels.view(-1).to(device)
        loss = criterion(logits, labels)

        cm.update(logits.argmax(dim=-1), labels)
        loss_meter.update(loss.item())

        pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1, num_point).astype(int)
        pre_list.append(pred)
        logits_list.append(logits.cpu().numpy().reshape(-1, num_point, num_classes))

    pre_list = np.concatenate(pre_list, axis=0)
    logits_list = np.concatenate(logits_list, axis=0)
    miou, macc, oa, ious, accs = cm.all_metrics()
    return EvalResult(
        pred=pre_list,
        logits=logits_list,
        loss=loss_meter.avg,
        miou=miou,
        macc=macc,
        oa=oa,
        ious=ious,
        accs=accs,
        elapsed=time.time() - start,
    )
