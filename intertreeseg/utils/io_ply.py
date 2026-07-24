"""Point-cloud I/O: PLY export and result H5 writer (migrated from ``ply_view2``)."""

from __future__ import annotations

import os

import h5py
import numpy as np
from plyfile import PlyData, PlyElement


def write_ply_xyz_scalar(
    pointcloud: np.ndarray,
    filename: str,
    scalar_name: str = "intensity",
) -> None:
    """Write an ``(N, >=4)`` array as a PLY with x, y, z + one scalar field.

    Column 3 (0-indexed) is stored under ``scalar_name`` (legacy default
    "intensity"; for merged instance results this column is the instance label).
    Renamed from the legacy ``ply_view2``; the scalar field name is now a
    parameter instead of being hardcoded to "intensity".
    """
    vertex = np.array(
        [
            (pointcloud[i][0], pointcloud[i][1], pointcloud[i][2], pointcloud[i][3])
            for i in range(pointcloud.shape[0])
        ],
        dtype=[
            ("x", np.dtype("float32")),
            ("y", np.dtype("float32")),
            ("z", np.dtype("float32")),
            (scalar_name, np.dtype("float32")),
        ],
    )
    output_pc = PlyData([PlyElement.describe(vertex, "vertex")])
    output_pc.write(filename)


def write_result_h5(points: np.ndarray, filename: str, dataset: str = "points") -> None:
    """Write a merged result array to a gzip-compressed H5 file (legacy layout)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, "w") as h5_fout:
        h5_fout.create_dataset(
            dataset,
            data=points,
            compression="gzip",
            compression_opts=4,
            dtype="float32",
        )
