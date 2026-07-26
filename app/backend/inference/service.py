"""InferenceService interface.

The whole UI loop talks to this interface only. MVP uses `MockInferenceService`
(no torch). Swapping in the real promptable PTv3 model later means implementing
this same `infer` signature in a `Ptv3InferenceService` and selecting it via
config.INFERENCE_BACKEND — no call-site changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Click:
    x: float
    y: float
    z: float
    positive: bool  # True = belongs to this tree, False = does not


@dataclass
class InferResult:
    mask_indices: np.ndarray  # indices (into the candidate/scene array) of foreground points
    elapsed_ms: float


class InferenceService(ABC):
    @abstractmethod
    def infer(
        self,
        scene_xyz: np.ndarray,
        candidate_indices: np.ndarray,
        clicks: list[Click],
    ) -> InferResult:
        """Segment one tree.

        Args:
            scene_xyz: (N, 3) all scene point coordinates.
            candidate_indices: indices into scene_xyz defining the bbox crop to
                segment within. If empty, the whole scene is the candidate set.
            clicks: user positive/negative prompt clicks.

        Returns:
            InferResult whose mask_indices are a subset of candidate_indices.
        """
        ...
