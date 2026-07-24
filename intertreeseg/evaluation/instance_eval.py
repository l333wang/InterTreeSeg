"""Read a merged result and report instance-level metrics.

Refactored from the legacy ``Evaluate_treeSeg`` script (which ran on import with
hardcoded Windows paths). Here everything is behind functions taking explicit
paths, so it is importable and scriptable.

Expected result H5 layout (as written by :func:`intertreeseg.evaluation.merge.merge_scene`):
column 3 = ground-truth instance id, column 4 = predicted instance id.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import h5py
import numpy as np

from ..metrics.instance_iou import InstanceMetrics, instance_metrics


def read_points_from_h5(file_path: str, dataset: str = "points") -> np.ndarray:
    """Read the point array from a result H5 file."""
    with h5py.File(file_path, "r") as f:
        return f[dataset][:]


def save_error_ply(points: np.ndarray, file_path: str, gt_col: int = 3, pred_col: int = 4) -> None:
    """Write an error-colored PLY: green = correct, red = wrong (needs open3d)."""
    import open3d as o3d

    gt = points[:, gt_col].astype(int)
    pred = points[:, pred_col].astype(int)
    err = np.where(gt == pred, 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = np.zeros((err.shape[0], 3))
    colors[err == 1] = [1, 0, 0]
    colors[err == 0] = [0, 1, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file_path, pcd)


def evaluate_result_h5(
    file_path: str,
    detection_iou: float = 0.7,
    gt_col: int = 3,
    pred_col: int = 4,
    error_ply: Optional[str] = None,
) -> InstanceMetrics:
    """Evaluate one result H5 and return :class:`InstanceMetrics`."""
    points = read_points_from_h5(file_path)
    gt = points[:, gt_col].astype(int)
    pred = points[:, pred_col].astype(int)
    metrics = instance_metrics(gt, pred, detection_iou=detection_iou)
    if error_ply is not None:
        save_error_ply(points, error_ply, gt_col, pred_col)
    return metrics


def evaluate_result_dir(
    result_dir: str,
    detection_iou: float = 0.7,
    gt_col: int = 3,
    pred_col: int = 4,
) -> dict[str, InstanceMetrics]:
    """Evaluate every ``*.h5`` under ``result_dir``; returns {scene: metrics}."""
    out: dict[str, InstanceMetrics] = {}
    for h5_path in sorted(glob.glob(os.path.join(result_dir, "*.h5"))):
        name = os.path.splitext(os.path.basename(h5_path))[0]
        out[name] = evaluate_result_h5(h5_path, detection_iou, gt_col, pred_col)
    return out
