"""Derive per-tree geometric attributes from a segmented point mask.

Used both for the single-tree "compute now on commit" path and the batch
"recompute all" path. Returns plain floats; callers decide whether to keep a
value or a user's manual override.
"""
from __future__ import annotations

import numpy as np


def compute_geometry(mask_xyz: np.ndarray) -> dict[str, float]:
    """Compute {height, crown_width, stem_x, stem_y, point_count} for one tree.

    - height: vertical extent of the mask (z_max - z_min). Ground baseline is the
      mask's own lowest point for now; a DTM-based baseline can replace this.
    - crown_width: larger of the horizontal x/y extents.
    - stem_x/stem_y: horizontal centroid of the lowest slice of points (stem base).
    """
    pts = np.asarray(mask_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return {"height": 0.0, "crown_width": 0.0, "stem_x": 0.0, "stem_y": 0.0, "point_count": 0}

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    height = z_max - z_min

    crown_width = float(max(x.max() - x.min(), y.max() - y.min()))

    # Stem base: centroid of the lowest 10% of points by height (>= a few points).
    n = pts.shape[0]
    k = max(1, int(0.1 * n))
    low_idx = np.argpartition(z, k - 1)[:k] if k < n else np.arange(n)
    stem_x = float(x[low_idx].mean())
    stem_y = float(y[low_idx].mean())

    return {
        "height": round(height, 3),
        "crown_width": round(crown_width, 3),
        "stem_x": round(stem_x, 3),
        "stem_y": round(stem_y, 3),
        "point_count": int(n),
    }
