"""Project full-resolution points to screen and test them against a 2D selection.

Lets manual lasso/brush editing act on FULL-resolution points even though the
browser only renders a decimated subset: the frontend sends the camera's
view-projection matrix + viewport, and the backend does the projection here.
"""
from __future__ import annotations

import numpy as np


def _view_proj_matrix(elements: list[float]) -> np.ndarray:
    """three.js Matrix4.elements is column-major; return a 4x4 row/col matrix."""
    e = np.asarray(elements, dtype=np.float64)
    m = np.empty((4, 4), dtype=np.float64)
    for col in range(4):
        for row in range(4):
            m[row, col] = e[col * 4 + row]
    return m


def project_to_screen(xyz: np.ndarray, view_proj: list[float], vw: float, vh: float):
    """Return (sx, sy, visible) pixel coords for each point (container-relative)."""
    m = _view_proj_matrix(view_proj)
    n = xyz.shape[0]
    homo = np.concatenate([xyz, np.ones((n, 1))], axis=1)     # (n,4)
    clip = homo @ m.T                                         # (n,4)
    w = clip[:, 3]
    visible = w > 1e-9
    w_safe = np.where(visible, w, 1.0)
    ndc_x = clip[:, 0] / w_safe
    ndc_y = clip[:, 1] / w_safe
    ndc_z = clip[:, 2] / w_safe
    visible &= (ndc_z >= -1) & (ndc_z <= 1)
    sx = (ndc_x + 1) * 0.5 * vw
    sy = (1 - ndc_y) * 0.5 * vh
    return sx, sy, visible


def select_polygon(xyz, view_proj, vw, vh, polygon: list[list[float]]) -> np.ndarray:
    sx, sy, vis = project_to_screen(xyz, view_proj, vw, vh)
    poly = np.asarray(polygon, dtype=np.float64)
    inside = _point_in_polygon(sx, sy, poly) & vis
    return np.nonzero(inside)[0]


def select_disc(xyz, view_proj, vw, vh, cx: float, cy: float, r: float) -> np.ndarray:
    sx, sy, vis = project_to_screen(xyz, view_proj, vw, vh)
    inside = ((sx - cx) ** 2 + (sy - cy) ** 2 <= r * r) & vis
    return np.nonzero(inside)[0]


def _point_in_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon for many points."""
    n = poly.shape[0]
    inside = np.zeros(px.shape[0], dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside
