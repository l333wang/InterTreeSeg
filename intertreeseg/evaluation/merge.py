"""Whole-scene reassembly: merge per-tree binary masks into an instance map.

Migrated from the legacy ``mergelarge`` / ``afterRefine``. The label column is
configurable (was hardcoded ``one_shot[:, 4]``) and the output directory is
passed in by the caller instead of being built from hardcoded ``result/`` paths.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Mapping, Optional

import numpy as np

from ..utils.io_ply import write_ply_xyz_scalar, write_result_h5


def after_refine(pr_label: np.ndarray, fake_label: Mapping[int, list]) -> np.ndarray:
    """Assign still-unclassified points (label 0) to the instance with max fg logit."""
    pr_label = np.squeeze(pr_label)
    idx = np.squeeze(np.argwhere(pr_label == 0))
    for ii in np.atleast_1d(idx):
        best_inst, best_score = -1, -1
        for inst, front_p in fake_label[int(ii)]:
            if best_inst == -1 or front_p > best_score:
                best_inst, best_score = inst, front_p
        pr_label[ii] = best_inst
    return np.expand_dims(pr_label, axis=-1)


def merge_scene(
    one_shot: np.ndarray,
    scene_name: str,
    pred: np.ndarray,
    pos_label: np.ndarray,
    logits: np.ndarray,
    cfg: Mapping[str, Any],
    output_dir: str,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Reassemble per-block predictions into a full-scene instance labeling.

    Writes ``<output_dir>/<scene>.ply`` and ``<output_dir>/<scene>.h5`` and
    returns the scene-level instance IoU.

    Parameters
    ----------
    pred: ``(B, N)`` per-block argmax (foreground/background).
    pos_label: ``(B, N, 2)`` -> (position index into the scene, instance id).
    logits: ``(B, N, C)`` per-block logits (used for force-divide tie-breaks).
    """
    if logger is None:
        logger = logging.getLogger()

    label_channel = int(cfg["data"].get("label_channel", 4))
    force_divide = bool(cfg["inference"].get("force_divide", False))

    points = one_shot[:, 0:3]
    label = np.squeeze(one_shot[:, label_channel])

    pred = np.expand_dims(pred, axis=-1)
    pred2 = np.concatenate((pred, pos_label), axis=-1)  # (B, N, 3): fg/bg, pos, inst
    B, N, C = pred2.shape
    pred2 = pred2.reshape(B * N, C)
    logits = logits.reshape(B * N, -1)

    pr_label = np.zeros((points.shape[0], 1), dtype=int)
    fake_label = defaultdict(list)
    for idx in range(B * N):
        pos = int(pred2[idx, 1])
        if pr_label[pos, 0] == 0 and pred2[idx, 0] != 0:       # foreground
            pr_label[pos, 0] = pred2[idx, 2]
        elif pr_label[pos, 0] == 0 and pred2[idx, 0] == 0:     # unclassified candidate
            fake_label[pos].append([pred2[idx, 2], logits[idx, 1]])

    if force_divide:
        logger.info("Start force divide!")
        pr_label = after_refine(pr_label, fake_label)
    else:
        logger.info("No force divide!")

    scene_name = os.path.splitext(scene_name)[0]
    os.makedirs(output_dir, exist_ok=True)

    merge = np.concatenate((points, pr_label), axis=-1)
    ply_path = os.path.join(output_dir, scene_name + ".ply")
    write_ply_xyz_scalar(merge, ply_path, scalar_name="instance")
    logger.info(f"store PLY: {ply_path}")

    label = np.squeeze(label)
    acc = 1.0 * np.sum(label[:] == (pr_label[:, 0] - 1)) / label.shape[0]
    cond_pred = (pr_label[:, 0]) > 0
    cond_correct = label[:] == (pr_label[:, 0] - 1)
    gt = cond_pred & cond_correct
    iou = 1.0 * np.sum(gt) / np.sum(cond_pred) if np.sum(cond_pred) > 0 else 0.0
    logger.info(f"The scene:{scene_name} final IoU result:{iou:.6f}, final Acc:{acc:.6f}")

    label = np.expand_dims(label, axis=-1)
    result = np.concatenate((points, label, pr_label - 1), axis=-1)  # xyz, gt_inst, pred_inst
    write_result_h5(result, os.path.join(output_dir, scene_name + ".h5"))
    return iou
