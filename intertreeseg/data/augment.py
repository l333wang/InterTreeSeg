"""Point-cloud augmentation.

The primary augmentation used by the training pipeline is a per-sample random
rotation about a single axis (``dataaugment_z``). Unlike the legacy
``dataaugmentZ`` — which applied the rotation to *all* columns and therefore
broke (or corrupted) any non-xyz feature channel — this version rotates only the
xyz coordinates (columns 0:3) and passes feature columns through untouched. For
the default xyz-only case the result is identical to the legacy behavior.

``jitter_point_cloud`` / ``shift_point_cloud`` / ``random_scale_point_cloud`` and
the combined ``dataaugment`` (Rodrigues rotate + scale + jitter) are kept as
optional helpers; they operate on xyz only.
"""

from __future__ import annotations

import numpy as np


def _rotation_matrices(axis: str, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Stack of ``(B, 3, 3)`` rotation matrices about ``axis`` for each (cos, sin)."""
    B = cos.shape[0]
    rot = np.zeros((B, 3, 3), dtype=np.float64)
    if axis == "x":
        rot[:, 0, 0] = 1
        rot[:, 1, 1] = cos
        rot[:, 1, 2] = -sin
        rot[:, 2, 1] = sin
        rot[:, 2, 2] = cos
    elif axis == "y":
        rot[:, 0, 0] = cos
        rot[:, 0, 2] = sin
        rot[:, 1, 1] = 1
        rot[:, 2, 0] = -sin
        rot[:, 2, 2] = cos
    elif axis == "z":
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = -sin
        rot[:, 1, 0] = sin
        rot[:, 1, 1] = cos
        rot[:, 2, 2] = 1
    else:
        raise NotImplementedError("axis must be 'x', 'y', or 'z'.")
    return rot


def dataaugment_z(blocks: np.ndarray, axis: str = "z", center=None) -> np.ndarray:
    """Random per-sample rotation about ``axis``, applied to xyz only.

    Parameters
    ----------
    blocks:
        ``(B, N, 3+F)`` array — columns 0:3 are xyz, the rest are passed through.
    axis:
        Rotation axis ("x" | "y" | "z"). Trees are upright, so "z" is the default.
    center:
        Rotation center; if ``None``, each sample's bbox center (xyz) is used.
    """
    out = blocks.copy()
    xyz = out[..., 0:3]
    B = xyz.shape[0]

    angles = np.random.uniform(-1, 1, size=B) * np.pi
    rot = _rotation_matrices(axis, np.cos(angles), np.sin(angles))

    if center is None:
        mn = xyz.min(axis=1)  # (B, 3)
        mx = xyz.max(axis=1)
        center = (mn + mx) / 2.0
    else:
        center = np.asarray(center)

    centered = xyz - center[:, np.newaxis, :]
    rotated = np.einsum("bij,bnj->bni", rot, centered)
    out[..., 0:3] = rotated + center[:, np.newaxis, :]
    return out


# --- optional helpers (xyz only) ---
def jitter_point_cloud(batch_data: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """Per-point Gaussian jitter, clipped. ``(B, N, C)`` in/out."""
    assert clip > 0
    jitter = np.clip(sigma * np.random.randn(*batch_data.shape), -clip, clip)
    return batch_data + jitter


def shift_point_cloud(batch_data: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
    """Per-cloud random translation. ``(B, N, 3)`` in/out (mutates input)."""
    B = batch_data.shape[0]
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for b in range(B):
        batch_data[b, :, :] += shifts[b, :]
    return batch_data


def random_scale_point_cloud(batch_data: np.ndarray, scale_low: float = 0.8, scale_high: float = 1.25) -> np.ndarray:
    """Per-cloud random isotropic scale. ``(B, N, 3)`` in/out (mutates input)."""
    B = batch_data.shape[0]
    scales = np.random.uniform(scale_low, scale_high, B)
    for b in range(B):
        batch_data[b, :, :] *= scales[b]
    return batch_data
