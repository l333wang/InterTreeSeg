"""Real promptable PointTransformerV3 inference (in-process, GPU).

Uses the packaged model from the ``intertreeseg`` library. Replicates the trained
interactive inference recipe but driven by *user* clicks instead of
ground-truth-sampled clicks:

  1. Take the bbox crop (candidate_indices) as the working point set, carrying
     each point's original scene index so results map back.
  2. scene_divide into blocks of PTV3_NUM_POINT (8192), padding the tail.
  3. Per block: build the positive/negative click heatmap channels
     pch/nch = sum exp(-|xyz - click|^2 / (2*sigma)) in world metres, then
     normalise xyz to [0,1] (as Gen_clicks does).
  4. Feed {feat:[pch,nch], coord:normalised xyz, batch, grid_size} to the model,
     argmax -> foreground, collect the union of scene indices predicted class 1.

flash_attn is NOT required: the model is built with enable_flash=False, which
selects the repo's plain-softmax attention path (mathematically equivalent, so
the pretrained weights apply unchanged).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .. import config
from .service import Click, InferenceService, InferResult


def _scene_divide(one_shot: np.ndarray, num_point: int):
    """Port of crsnet_dataloader.scene_divide: shuffle + tile into (B,num_point,C)."""
    n, c = one_shot.shape
    idx = np.arange(n)
    np.random.shuffle(idx)
    one_shot = one_shot[idx]
    num_divide = n // num_point
    if num_divide < 1:
        batch = num_point // n
        els = num_point % n
        rep = np.tile(one_shot, (batch, 1))
        scene = np.concatenate((rep, one_shot[:els]), axis=0)
        num_divide = 1
    else:
        els = n % num_point
        scene = one_shot
        if els > 0:
            scene = np.concatenate((scene, one_shot[: num_point - els]), axis=0)
            num_divide += 1
    return scene.reshape(num_divide, num_point, c), num_divide


class Ptv3InferenceService(InferenceService):
    def __init__(self):
        import torch
        from pathlib import Path

        # Use the packaged model from the intertreeseg library (this repo). Ensure
        # the repo root is importable even when the backend is launched from app/
        # without `pip install -e .`.
        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from intertreeseg.models import PointTransformerV3  # noqa: E402

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_point = config.PTV3_NUM_POINT
        self.grid_size = config.PTV3_GRID_SIZE
        self.sigma = config.PTV3_CLICK_SIGMA

        model = PointTransformerV3(
            in_channels=2,
            num_classes=2,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            enc_patch_size=(512, 512, 512, 512, 512),
            dec_patch_size=(512, 512, 512, 512),
            enable_flash=False,  # no flash_attn dependency
        ).to(self.device)
        state = torch.load(config.PTV3_CHECKPOINT, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def infer(
        self,
        scene_xyz: np.ndarray,
        candidate_indices: np.ndarray,
        clicks: list[Click],
    ) -> InferResult:
        torch = self.torch
        t0 = time.perf_counter()

        if candidate_indices is None or len(candidate_indices) == 0:
            candidate_indices = np.arange(scene_xyz.shape[0])
        cand = np.ascontiguousarray(candidate_indices).astype(np.int64)
        if not any(c.positive for c in clicks):
            return InferResult(np.empty(0, np.int64), (time.perf_counter() - t0) * 1e3)

        pts = scene_xyz[cand, :3].astype(np.float64)
        # carry the original scene index as a 4th column through the tiling
        arr = np.concatenate([pts, cand.reshape(-1, 1).astype(np.float64)], axis=1)
        blocks, ndiv = _scene_divide(arr, self.num_point)

        pos = np.array([[c.x, c.y, c.z] for c in clicks if c.positive], dtype=np.float64)
        neg = np.array([[c.x, c.y, c.z] for c in clicks if not c.positive], dtype=np.float64)

        coords = np.empty((ndiv, self.num_point, 3), dtype=np.float32)
        feats = np.empty((ndiv, self.num_point, 2), dtype=np.float32)
        block_idx = blocks[:, :, 3].astype(np.int64)

        two_sigma = 2.0 * self.sigma
        for b in range(ndiv):
            bxyz = blocks[b, :, :3]
            feats[b, :, 0] = _click_heat(bxyz, pos, two_sigma)
            feats[b, :, 1] = _click_heat(bxyz, neg, two_sigma)
            mn = bxyz.min(axis=0)
            rng = bxyz.max(axis=0) - mn
            rng[rng == 0] = 1.0
            coords[b] = ((bxyz - mn) / rng).astype(np.float32)

        total = ndiv * self.num_point
        coord = torch.from_numpy(coords.reshape(total, 3)).to(self.device)
        feat = torch.from_numpy(feats.reshape(total, 2)).to(self.device)
        batch = torch.arange(ndiv, dtype=torch.long, device=self.device).repeat_interleave(
            self.num_point
        )
        data = {"feat": feat, "coord": coord, "batch": batch, "grid_size": self.grid_size}

        with torch.inference_mode():
            logits = self.model(data)
            logits = logits.view(-1, logits.shape[-1])
            pred = logits.argmax(dim=-1).cpu().numpy()

        fg_scene = np.unique(block_idx.reshape(-1)[pred == 1])
        return InferResult(fg_scene.astype(np.int64), (time.perf_counter() - t0) * 1e3)


def _click_heat(bxyz: np.ndarray, centers: np.ndarray, two_sigma: float) -> np.ndarray:
    """sum_j exp(-|bxyz - center_j|^2 / (2*sigma)) over click centers; (N,) float."""
    out = np.zeros(bxyz.shape[0], dtype=np.float64)
    for j in range(centers.shape[0]):
        d2 = ((bxyz - centers[j]) ** 2).sum(axis=1)
        out += np.exp(-d2 / two_sigma)
    return out
