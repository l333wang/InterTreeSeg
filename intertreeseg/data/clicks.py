"""Interactive click channels: generation and iterative refinement.

Three entry points (migrated and generalized from the legacy
``Gen_clicks`` / ``Gen_clicks2`` / ``reClick``):

* :func:`gen_clicks` — training. Positive + negative Gaussian click heatmaps.
* :func:`gen_click_center` — inference. A single click near an object center,
  placed in the positive or negative slot depending on object size.
* :func:`reclick` — inference refinement. Adds a correction click to blocks
  whose IoU is below a threshold.

All three keep the canonical block layout ``[xyz | feat | click_0 .. click_{K-1}]``
(see :class:`ChannelSpec`). Click channels live at ``spec.click_start`` onward, so
extra raw feature channels never shift them out from under the code — this fixes
the legacy ``reClick`` which hardcoded click indices 3 and 4 and silently dropped
any raw feature columns.

For the default xyz-only, 2-click configuration the output is byte-identical to
the original pipeline (which is what keeps the released checkpoints usable).
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.utils.data as Data

from .channels import ChannelSpec
from .normalize import normalize_block


def mean_shift(cent: np.ndarray, radius: float, points: np.ndarray) -> np.ndarray:
    """Mean-shift a center toward the local density mode within ``radius``."""
    num1 = 0
    dist = np.linalg.norm(points - cent, ord=2, axis=-1)
    cond = dist < radius
    num2 = np.sum(cond)
    cent_new = cent
    it = 0
    while abs(num2 - num1) > 4:
        num1 = num2
        cent_new = np.sum(points[cond, ...], axis=0) / float(num2)
        dist = np.linalg.norm(points - cent_new, ord=2, axis=-1)
        cond = dist < radius
        num2 = np.sum(cond)
        it += 1
        if it > 10:
            break
    return cent_new


def _gaussian_heat(points_xyz: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian response ``(N, 1)`` of every point to a click ``center``."""
    dist = np.linalg.norm(points_xyz - center, ord=2, axis=-1, keepdims=True)
    return np.exp(-np.square(dist) / (2 * sigma))


def gen_clicks(
    blocks: np.ndarray,
    labels: np.ndarray,
    spec: ChannelSpec,
    sigma: float = 0.01,
    pos_max: int = 16,
    neg_max: int = 16,
    min_obj_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Training-time click generation (positive + negative heatmaps).

    Parameters
    ----------
    blocks:
        ``(B, N, 3+F)`` array with columns ``[xyz | feat]``.
    labels:
        ``(B, N)`` binary foreground/background labels.

    Returns
    -------
    (new_blocks, new_labels):
        ``new_blocks`` has shape ``(B', N, 3+F+2)`` with ``[xyz | feat | pos | neg]``.
        Blocks with fewer than ``min_obj_points`` foreground points are dropped
        (hence ``B' <= B`` and labels are re-collected alongside).
    """
    new_blocks, new_labels = [], []
    for i in range(blocks.shape[0]):
        block = blocks[i, ...]
        la = labels[i, ...]

        cond = la == 1
        if np.sum(cond) < min_obj_points:
            continue
        obj = block[cond, 0:3]

        p_num = random.randint(1, pos_max)
        pos_idx = random.sample(range(obj.shape[0]), p_num)
        pos_clicks = obj[pos_idx, :]
        pch = np.zeros((block.shape[0], 1), dtype=np.float32)
        for j in range(p_num):
            pch = pch + _gaussian_heat(block[:, 0:3], pos_clicks[j, ...], sigma)

        n_num = random.randint(0, neg_max)
        nch = np.zeros((block.shape[0], 1), dtype=np.float32)
        bg_cond = la == 0
        if np.sum(bg_cond) > n_num and n_num > 0:
            bg = block[bg_cond, 0:3]
            neg_idx = random.sample(range(bg.shape[0]), n_num)
            neg_clicks = bg[neg_idx, :]
            for j in range(n_num):
                nch = nch + _gaussian_heat(block[:, 0:3], neg_clicks[j, ...], sigma)

        norm = normalize_block(block, spec)
        new_b = np.concatenate((norm, pch, nch), axis=1)
        new_blocks.append(np.expand_dims(new_b, axis=0))
        new_labels.append(np.expand_dims(la, axis=0))

    new_blocks = np.concatenate(new_blocks, axis=0)
    new_labels = np.concatenate(new_labels, axis=0)
    return new_blocks, new_labels


def gen_click_center(
    blocks: np.ndarray,
    labels: np.ndarray,
    spec: ChannelSpec,
    sigma: float = 0.01,
    rand_index: int = 1,
    neg_switch_points: int = 1000,
    min_obj_points: int = 100,
    flag: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inference-time first-click generation (single click, pos or neg slot).

    Mirrors the legacy ``Gen_clicks2``: the block's xyz are mean-centered, a click
    center is chosen (randomly if ``flag`` else via one of four heuristics), a
    Gaussian heat channel is built, coordinates/features are normalized, and the
    heat is placed in the positive slot for large objects or the negative slot for
    small ones.
    """
    new_blocks, clicks = [], []
    for i in range(blocks.shape[0]):
        block = blocks[i, ...].copy()
        # Mean-center xyz only (legacy centered the xyz-only block it received).
        block[:, 0:3] = block[:, 0:3] - np.mean(block[:, 0:3], axis=0)
        la = labels[i, ...]

        cond = la == 1
        if np.sum(cond) < neg_switch_points:  # too few fg -> click on background
            obj = block[~cond, 0:3]
        else:
            obj = block[cond, 0:3]
        if obj.shape[0] < min_obj_points:
            continue

        scale = np.max(obj, axis=0) - np.min(obj, axis=0)

        if flag:  # fully random
            cent = obj[np.random.randint(obj.shape[0], size=1)]
        else:
            choice = int(np.random.randint(4, size=1)[0])
            if choice == 0:
                cent = np.mean(obj, axis=0)[np.newaxis, :]
                cent = cent + (np.random.rand(3) - 0.5) * 2 * scale * 0.05 * rand_index
                cent = obj[np.argmin(np.linalg.norm(obj - cent, ord=2, axis=-1))]
            elif choice == 1:
                cent = (np.max(obj, axis=0) + np.min(obj, axis=0)) / 2
                cent = cent + (np.random.rand(3) - 0.5) * 2 * scale * 0.1 * rand_index
                cent = obj[np.argmin(np.linalg.norm(obj - cent, ord=2, axis=-1))]
            elif choice == 2:
                cent = (np.max(obj, axis=0) + np.min(obj, axis=0)) / 2
                cent = cent + (np.random.rand(3) - 0.5) * 2 * scale * 0.1 * rand_index
            else:
                radius = np.sum(scale**2) * 0.05
                cent = mean_shift(np.mean(obj, axis=0), radius, obj)

        gaus = _gaussian_heat(block[:, 0:3], cent, sigma)
        norm = normalize_block(block, spec)
        zeros = np.zeros(gaus.shape, dtype=gaus.dtype)
        if np.sum(cond) < min_obj_points:
            new_b = np.concatenate((norm, zeros, gaus), axis=1)  # neg slot
        else:
            new_b = np.concatenate((norm, gaus, zeros), axis=1)  # pos slot

        new_blocks.append(np.expand_dims(new_b, axis=0))
        clicks.append(cent)

    new_blocks = np.concatenate(new_blocks, axis=0)
    clicks = np.concatenate(clicks, axis=0)
    return new_blocks, clicks


def reclick(
    pred: np.ndarray,
    dataset: Data.TensorDataset,
    n_clicks: np.ndarray,
    spec: ChannelSpec,
    thr: float = 0.80,
    sigma: float = 0.01,
):
    """Add a correction click to every block whose object IoU is ``<= thr``.

    Click channels are located dynamically at ``spec.click_start`` and raw
    feature columns between xyz and the clicks are preserved (the legacy version
    hardcoded indices 3/4 and dropped raw features).
    """
    cs = spec.click_start
    new_blocks, clicks = [], []
    for i in range(pred.shape[0]):
        pred_one = pred[i, ...]
        block_one, la_one = dataset[i]
        block_one = np.array(block_one)
        la_one = np.array(la_one)

        points_one = block_one[:, 0:3]
        rawfeat = block_one[:, 3:cs]                    # preserved untouched
        pch = block_one[:, cs : cs + 1].reshape(-1, 1)
        nch = block_one[:, cs + 1 : cs + 2].reshape(-1, 1)

        # object IoU (foreground)
        inter = np.sum(np.multiply(pred_one, la_one))
        union = np.sum(pred_one) + np.sum(la_one) - inter
        iou_obj = inter / float(union)

        if iou_obj <= thr:
            # points that belong to the object but were missed
            miss = np.multiply(-pred_one + 1, la_one)
            # points wrongly classified as object
            wrong = np.multiply(pred_one, -la_one + 1)
            n_clicks[i] = n_clicks[i] + 1

            if np.sum(miss) > np.sum(wrong):
                obj = points_one[miss == 1, ...]
                cent = obj[np.random.randint(obj.shape[0], size=1)]
                pch = pch + _gaussian_heat(points_one, cent, sigma)
            else:
                obj = points_one[wrong == 1, ...]
                cent = obj[np.random.randint(obj.shape[0], size=1)]
                nch = nch + _gaussian_heat(points_one, cent, sigma)

            new_b = np.concatenate((points_one, rawfeat, pch, nch), axis=1)
        else:
            new_b = block_one
            cent = np.zeros((1, 3), dtype=np.float32)

        new_blocks.append(np.expand_dims(new_b, axis=0))
        clicks.append(cent)

    new_blocks = np.concatenate(new_blocks, axis=0)
    clicks = np.concatenate(clicks, axis=0)

    _, label = dataset[:]
    new_blocks = torch.tensor(new_blocks, dtype=torch.float)
    label = torch.tensor(np.array(label), dtype=torch.long)
    new_dataset = Data.TensorDataset(new_blocks, label)
    return new_dataset, n_clicks, clicks
