"""Point-cloud reader abstraction.

MVP ships `TxtReader` (6-column ASCII). Adding `.las/.laz/.ply` later means
implementing another `PointCloudReader` and registering it in `get_reader`;
callers stay unchanged.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class PointCloudReader(ABC):
    """Reads a point cloud file into an (N, C) float array.

    Convention for the returned array columns: at least x, y, z in columns 0..2.
    Extra columns (intensity/feature, ground-truth labels) are preserved as-is
    so evaluation code can use them later.
    """

    #: file extensions this reader handles, lowercase incl. dot
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def read(self, path: str) -> np.ndarray:  # pragma: no cover - interface
        ...


class TxtReader(PointCloudReader):
    """6-column whitespace-separated ASCII: x y z feat gt_semantic gt_instance.

    Only the first 3 columns are required; any trailing columns are kept.
    """

    extensions = (".txt",)

    def read(self, path: str) -> np.ndarray:
        import pandas as pd

        # pandas is markedly faster than np.loadtxt on large files.
        arr = pd.read_csv(path, sep=r"\s+", header=None).values
        if arr.shape[1] < 3:
            raise ValueError(
                f"{os.path.basename(path)}: expected >=3 columns (x y z), got {arr.shape[1]}"
            )
        return np.ascontiguousarray(arr, dtype=np.float64)


_READERS: list[PointCloudReader] = [TxtReader()]


def get_reader(path: str) -> PointCloudReader:
    ext = os.path.splitext(path)[1].lower()
    for r in _READERS:
        if ext in r.extensions:
            return r
    raise ValueError(f"No reader registered for extension '{ext}'")
