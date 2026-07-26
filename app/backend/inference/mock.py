"""Geometry-based mock segmentation.

No model / no torch. Selects a tree-like vertical column around the positive
clicks and subtracts the neighbourhood of the negative clicks. Good enough to
drive the full annotation UI end-to-end; replaced by the real PTv3 model behind
the same `InferenceService` interface.
"""
from __future__ import annotations

import time

import numpy as np

from .service import Click, InferenceService, InferResult

# Horizontal capture radius around a positive click (metres). A vertical column
# of this radius reads as a single tree in forestry point clouds.
_POS_RADIUS = 1.5
# Points within this radius of a negative click are pushed out of the mask.
_NEG_RADIUS = 1.0


class MockInferenceService(InferenceService):
    def infer(
        self,
        scene_xyz: np.ndarray,
        candidate_indices: np.ndarray,
        clicks: list[Click],
    ) -> InferResult:
        t0 = time.perf_counter()

        if candidate_indices is None or len(candidate_indices) == 0:
            candidate_indices = np.arange(scene_xyz.shape[0])
        cand = np.ascontiguousarray(candidate_indices)
        pts_xy = scene_xyz[cand, :2]

        pos = np.array([[c.x, c.y] for c in clicks if c.positive], dtype=np.float64)
        neg = np.array([[c.x, c.y] for c in clicks if not c.positive], dtype=np.float64)

        if len(pos) == 0:
            # Nothing positive to grow from → empty mask.
            return InferResult(np.empty(0, dtype=np.int64), (time.perf_counter() - t0) * 1e3)

        # Distance to nearest positive click (horizontal).
        dpos = _min_dist(pts_xy, pos)
        keep = dpos <= _POS_RADIUS

        if len(neg) > 0:
            dneg = _min_dist(pts_xy, neg)
            # Drop points that sit near a negative click and are not clearly
            # closer to a positive one.
            keep &= ~((dneg <= _NEG_RADIUS) & (dneg < dpos))

        mask_indices = cand[keep]
        return InferResult(mask_indices.astype(np.int64), (time.perf_counter() - t0) * 1e3)


def _min_dist(pts_xy: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Min horizontal distance from each point to any center. (P,2),(C,2)->(P,)."""
    # (P, C) distances; C is small (few clicks) so this is cheap.
    diff = pts_xy[:, None, :] - centers[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return d.min(axis=1)
