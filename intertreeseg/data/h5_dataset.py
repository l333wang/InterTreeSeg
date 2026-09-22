"""HDF5 training dataset: read blocks, sample, augment, generate clicks.

Migrated from the legacy ``Data_Input_h5_combined``. Channel selection is now
driven by :class:`ChannelSpec` (``coord_channels`` + ``feat_channels``) instead
of the ad-hoc ``data_channel`` string, so extra feature channels flow through in
the canonical ``[xyz | feat]`` layout and reach the model as part of ``feat``.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import torch.utils.data as Data

from .augment import dataaugment_z
from .channels import ChannelSpec
from .clicks import gen_clicks
from .sampling import sample, sample_divide


def click_kwargs(cfg: Mapping[str, Any]) -> dict:
    """Extract gen_clicks parameters from the config ``data.clicks`` block."""
    c = cfg["data"].get("clicks", {}) or {}
    return dict(
        sigma=float(c.get("sigma", 0.01)),
        pos_max=int(c.get("pos_max", 16)),
        neg_max=int(c.get("neg_max", 16)),
        min_obj_points=int(c.get("min_obj_points", 100)),
    )


def build_h5_dataset(
    path: str,
    spec: ChannelSpec,
    cfg: Mapping[str, Any],
    augment: bool = False,
) -> Data.TensorDataset:
    """Build a training TensorDataset from all ``.h5`` files under ``path``.

    Each h5 has datasets ``data`` ``(B, N, C)`` and ``label`` ``(B, N)``. Points
    are downsampled to ``num_point * num_sample`` then split into ``num_point``
    blocks; optional z-rotation augmentation can double the set; finally
    positive/negative click channels are appended.
    """
    data_cfg = cfg["data"]
    num_point = int(data_cfg["num_point"])
    num_sample = int(data_cfg["num_sample"])
    cols = list(spec.coord_channels) + list(spec.feat_channels)  # [xyz | feat] source cols
    num_channels = len(cols)  # = 3 + F

    all_points, all_labels = [], []
    for file in sorted(os.listdir(path)):
        if not file.endswith(".h5"):
            continue
        with h5py.File(os.path.join(path, file), "r") as f:
            points = f["data"][:][:, :, cols]
            label = f["label"][:]
        all_points.append(points)
        all_labels.append(label)

    all_points = np.concatenate(all_points, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_labels = np.expand_dims(all_labels, axis=-1)
    blocks = np.concatenate((all_points, all_labels), axis=2)
    blocks = sample(blocks, num_point * num_sample)
    blocks = sample_divide(blocks, num_point)
    all_points = blocks[:, :, :num_channels]
    all_labels = blocks[:, :, num_channels]

    aug_cfg = data_cfg.get("augment", {}) or {}
    if augment and aug_cfg.get("enabled", True):
        rot_axis = aug_cfg.get("rot_axis", "z")
        new_points = dataaugment_z(all_points, axis=rot_axis)
        if aug_cfg.get("double", True):
            all_points = np.concatenate((all_points, new_points), axis=0)
            all_labels = np.concatenate((all_labels, all_labels.copy()), axis=0)
        else:
            all_points = new_points

    blocks, all_labels = gen_clicks(all_points, all_labels, spec, **click_kwargs(cfg))

    blocks = torch.tensor(blocks, dtype=torch.float)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    return Data.TensorDataset(blocks, all_labels)
