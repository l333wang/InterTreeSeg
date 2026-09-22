"""Scene (txt) inference dataset: per-tree cropping + first-click generation.

Migrated from the legacy ``Data_Input_txt`` / ``process_block``. Compared to the
original:

* the block size, bbox buffer (``delta``), and label columns are configurable
  (were hardcoded 8192 / ``np.random.uniform(0, 0.5)`` / columns 4 and 5);
* channel selection is driven by :class:`ChannelSpec`, and the seg / position /
  instance label columns are located dynamically so extra feature channels do
  not shift them.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

from .channels import ChannelSpec
from .clicks import gen_click_center
from .sampling import scene_divide


def _resolve_delta(bbox_delta) -> float:
    """Resolve the bbox buffer: a scalar is fixed; a 2-list is uniform(low, high).

    Legacy behavior was ``np.random.uniform(0, 0.5)``; set ``bbox_delta: [0.0, 0.5]``
    to reproduce it exactly.
    """
    if isinstance(bbox_delta, (list, tuple)):
        low, high = float(bbox_delta[0]), float(bbox_delta[1])
        return float(np.random.uniform(low, high))
    return float(bbox_delta)


def process_block(
    j: int,
    label: np.ndarray,
    points: np.ndarray,
    num_points: int,
    spec: ChannelSpec,
    cfg: Mapping[str, Any],
):
    """Crop the scene around tree instance ``j`` and build its click-augmented blocks.

    ``points`` has columns ``[xyz | feat]`` (width ``3 + F``). Returns
    ``(test_block, test_plabel, test_click, pos_label)`` or ``None`` if the tree
    is too small.
    """
    data_cfg = cfg["data"]
    clicks_cfg = data_cfg.get("clicks", {}) or {}
    min_obj = int(clicks_cfg.get("min_obj_points", 100))
    block_size = int(data_cfg.get("block_size", 8192))
    F = spec.F

    idx = np.argwhere(label == j)
    if len(idx) < min_obj:
        return None
    idx = np.squeeze(idx)

    seg_label = np.zeros((num_points, 1), dtype=int)
    seg_label[idx] = 1
    pos_label = np.expand_dims(np.arange(0, num_points), axis=-1)
    inst_label = np.zeros((num_points, 1), dtype=int) + j + 1  # 0 means unclassified

    obj_xyz = points[idx, 0:3]
    obj_max = np.max(obj_xyz, axis=0)
    obj_min = np.min(obj_xyz, axis=0)
    delta = _resolve_delta(data_cfg.get("bbox_delta", 0.25))

    cond = (
        (points[:, 0] <= obj_max[0] + delta) & (points[:, 0] >= obj_min[0] - delta)
        & (points[:, 1] <= obj_max[1] + delta) & (points[:, 1] >= obj_min[1] - delta)
        & (points[:, 2] <= obj_max[2] + delta) & (points[:, 2] >= obj_min[2] - delta)
    )

    new_block = points[cond, ...]
    seg_label = seg_label[cond]
    pos_label = pos_label[cond]
    inst_label = inst_label[cond]
    new_block = np.concatenate((new_block, seg_label, pos_label, inst_label), axis=1)

    blocks, _ = scene_divide(new_block, block_size)
    coord_feat = blocks[:, :, 0 : 3 + F]     # [xyz | feat]
    test_plabel = blocks[:, :, 3 + F]        # seg (fg/bg) label
    pos_label = blocks[:, :, [3 + F + 1, 3 + F + 2]]  # (position idx, instance id)

    test_block, test_click = gen_click_center(
        coord_feat,
        test_plabel,
        spec,
        sigma=float(clicks_cfg.get("sigma", 0.01)),
        neg_switch_points=int(clicks_cfg.get("min_obj_points_infer", 1000)),
        min_obj_points=min_obj,
        flag=False,
    )
    return test_block, test_plabel, test_click, pos_label


def load_scene_txt(
    test_path: str,
    spec: ChannelSpec,
    cfg: Mapping[str, Any],
) -> Tuple[Data.TensorDataset, np.ndarray, np.ndarray, np.ndarray]:
    """Load a scene ``.txt`` and build its per-tree click-augmented dataset.

    Returns ``(dataset, clicks, pos_labels, one_shot)`` where ``one_shot`` is the
    raw scene array (kept for whole-scene reassembly).
    """
    data_cfg = cfg["data"]
    cols = list(spec.coord_channels) + list(spec.feat_channels)
    label_channel = int(data_cfg.get("label_channel", 4))

    one_shot = pd.read_csv(test_path, delim_whitespace=True, header=None).values
    points = one_shot[:, cols]
    label = np.squeeze(one_shot[:, label_channel])
    num_points = points.shape[0]

    int_num = int(np.max(label)) if isinstance(label, np.ndarray) else int(max(label))

    test_blocks, test_plabels, pos_labels, test_clicks = [], [], [], []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_block, j, label, points, num_points, spec, cfg)
            for j in range(0, int_num + 1)
        ]
        for future in futures:
            result = future.result()
            if result:
                tb, tp, tc, pl = result
                test_blocks.append(tb)
                test_plabels.append(tp)
                test_clicks.append(tc)
                pos_labels.append(pl)

    test_blocks = np.concatenate(test_blocks, axis=0)
    test_plabels = np.concatenate(test_plabels, axis=0)
    test_clicks = np.concatenate(test_clicks, axis=0)
    pos_labels = np.concatenate(pos_labels, axis=0)

    test_blocks = torch.tensor(test_blocks, dtype=torch.float)
    test_plabels = torch.tensor(test_plabels, dtype=torch.long)
    dataset = Data.TensorDataset(test_blocks, test_plabels)
    return dataset, test_clicks, pos_labels, one_shot
