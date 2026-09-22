"""Per-channel normalization.

Coordinates use the exact legacy per-block, per-axis min-max to ``[0, 1]`` that
the released checkpoints were trained on — do NOT change it or loaded weights
silently degrade. Raw feature channels (intensity, return number, ...) are
normalized **independently per channel** so they are never mixed into the xyz
min-max (which was a latent bug the moment a non-xyz channel was selected).

Click channels are never normalized (they are Gaussian responses that must keep
their native range); they are appended *after* normalization by the clicks
module.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .channels import ChannelSpec, NormSpec


def minmax_block(xyz: np.ndarray) -> np.ndarray:
    """Per-axis min-max of a single block's coordinates to ``[0, 1]``.

    Exactly reproduces the legacy transform::

        block = block - np.min(block, axis=0)
        block = block / (np.max(block, axis=0) - np.min(block, axis=0))

    No epsilon is added (to stay byte-identical to the trained pipeline); a
    degenerate constant axis will therefore produce non-finite values, same as
    before.
    """
    mn = np.min(xyz, axis=0)
    mx = np.max(xyz, axis=0)
    return np.divide(xyz - mn, mx - mn)


def _normalize_one_feat(col: np.ndarray, spec: NormSpec) -> np.ndarray:
    """Normalize a single feature column ``(N,)`` per its spec."""
    if isinstance(spec, tuple):
        name, param = spec
    else:
        name, param = spec, None

    if name == "none":
        return col
    if name == "standardize":
        mean = np.mean(col)
        std = np.std(col)
        return (col - mean) / std if std > 0 else col - mean
    if name == "minmax":
        mn, mx = np.min(col), np.max(col)
        return (col - mn) / (mx - mn) if mx > mn else col - mn
    if name == "scale":
        return col * float(param)
    raise ValueError(f"unknown feat_norm spec: {spec!r}")


def normalize_feats(feats: np.ndarray, spec: ChannelSpec) -> np.ndarray:
    """Normalize each raw feature column of ``feats`` ``(N, F)`` independently."""
    if feats.shape[-1] == 0:
        return feats
    out = feats.astype(np.float32, copy=True)
    for i, norm in enumerate(spec.feat_norm):
        out[:, i] = _normalize_one_feat(out[:, i], norm)
    return out


def normalize_block(block: np.ndarray, spec: ChannelSpec) -> np.ndarray:
    """Normalize a ``(N, 3+F)`` [xyz | feat] block: coord min-max + per-feat norm.

    For ``F == 0`` this is identical to the legacy coordinate min-max, keeping
    the default pipeline byte-compatible.
    """
    xyz = minmax_block(block[:, 0:3])
    if spec.F == 0:
        return xyz.astype(np.float32, copy=False)
    feats = normalize_feats(block[:, 3 : 3 + spec.F], spec)
    return np.concatenate([xyz, feats], axis=1).astype(np.float32, copy=False)
