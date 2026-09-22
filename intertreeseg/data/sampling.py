"""Point up/down-sampling and block-splitting helpers (migrated verbatim).

Shared by the h5 training dataset and the scene inference dataset.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sample(one_shot: np.ndarray, num_point: int = 1024) -> np.ndarray:
    """Up/down-sample each cloud in a batch ``(B, N, C)`` to ``num_point`` points."""
    length = one_shot.shape[1]
    idx = np.arange(length)
    if length >= num_point:  # down-sample
        np.random.shuffle(idx)
        return one_shot[:, idx[0:num_point], ...]
    # up-sample by tiling then filling the remainder
    batch = num_point // length
    els = num_point % length
    rep = np.tile(one_shot, (1, batch, 1))
    return np.concatenate((rep, one_shot[:, idx[0:els], :]), axis=1)


def sample_divide(one_shot: np.ndarray, num_point: int = 4096) -> np.ndarray:
    """Split a batch ``(B, N, C)`` into blocks of ``num_point`` -> ``(B*N//num_point, num_point, C)``."""
    (B1, N1, C1) = one_shot.shape
    num_divide = N1 // num_point
    idx = np.arange(N1)
    np.random.shuffle(idx)
    scene = one_shot[:, idx[0 : num_divide * num_point], :]
    scene = scene.reshape(B1, num_divide, num_point, -1)
    scene = scene.reshape(B1 * num_divide, num_point, -1)
    return scene


def scene_divide(one_shot: np.ndarray, num_point: int = 4096) -> Tuple[np.ndarray, int]:
    """Split a single dense cloud ``(N, C)`` into blocks ``(B, num_point, C)``.

    The final block is padded (by repeating points) so every block has exactly
    ``num_point`` points; returns the blocks and the number of blocks.
    """
    (N1, C1) = one_shot.shape
    idx = np.arange(N1)
    np.random.shuffle(idx)
    one_shot = one_shot[idx, ...]
    num_divide = N1 // num_point
    if num_divide < 1:
        batch = num_point // N1
        els = num_point % N1
        rep = np.tile(one_shot, (batch, 1))
        scene = np.concatenate((rep, one_shot[0:els, :]), axis=0)
        num_divide = 1
    else:
        els = N1 % num_point
        scene = one_shot
        if els > 0:
            pad = one_shot[0 : num_point - els, ...]
            scene = np.concatenate((scene, pad), axis=0)
            num_divide = num_divide + 1
    scene = scene.reshape(num_divide, num_point, C1)
    return scene, num_divide
