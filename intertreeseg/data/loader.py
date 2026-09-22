"""Build train / validation dataloaders from config (migrated ``load_data_cfg``)."""

from __future__ import annotations

import os
from typing import Any, Mapping, Tuple

import torch

from .channels import ChannelSpec
from .h5_dataset import build_h5_dataset


def build_dataloaders(
    cfg: Mapping[str, Any],
    augment: bool = False,
    data_root: str | None = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build ``(train_loader, val_loader)`` from ``<data_dir>/{train,validate}``.

    ``data_root`` optionally prefixes a relative ``data.data_dir``. Unlike the
    legacy loader (``shuffle=False`` for training), the training loader shuffles
    batches each epoch; the validation loader does not.
    """
    spec = ChannelSpec.from_cfg(cfg["data"])
    data_dir = cfg["data"]["data_dir"]
    if data_root is not None and not os.path.isabs(data_dir):
        data_dir = os.path.join(data_root, data_dir)

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "validate")

    train_dataset = build_h5_dataset(train_path, spec, cfg, augment=augment)
    val_dataset = build_h5_dataset(val_path, spec, cfg, augment=False)

    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_worker", 0))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader
