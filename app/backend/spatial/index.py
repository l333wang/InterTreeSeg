"""Spatial index over a scene's points: nearest-point picking + bbox cropping."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class SpatialIndex:
    """Wraps a KD-tree for click-to-nearest-point and a fast 2D bbox crop.

    xy is indexed for top-view rectangle selection; the full 3D tree handles
    nearest-point picking from a ray hit.
    """

    def __init__(self, xyz: np.ndarray):
        self.xyz = np.ascontiguousarray(xyz[:, :3], dtype=np.float64)
        # Build the KD-tree lazily: loading a scene doesn't need it (bbox crop is
        # a boolean mask), so we avoid a multi-million-point tree build at load.
        self._tree3d: cKDTree | None = None

    def nearest(self, point: np.ndarray) -> int:
        """Return the index of the point nearest to `point` (xyz)."""
        if self._tree3d is None:
            self._tree3d = cKDTree(self.xyz)
        _, idx = self._tree3d.query(np.asarray(point, dtype=np.float64)[:3], k=1)
        return int(idx)

    def crop_bbox_xy(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> np.ndarray:
        """Return indices of points inside the (top-view) rectangle.

        Z is unconstrained by default (full column height), matching the
        "draw a box from the top, capture the whole tree" workflow.
        """
        x, y, z = self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2]
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        if z_min is not None:
            mask &= z >= z_min
        if z_max is not None:
            mask &= z <= z_max
        return np.nonzero(mask)[0]
