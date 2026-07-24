"""Shared test paths/utilities.

Adds the package root to sys.path so ``python tests/xxx.py`` works without
installing, and locates the released checkpoint (overridable via TREESEG_CKPT).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Released 2-channel checkpoint. Set TREESEG_CKPT to your checkpoint path to run
# the checkpoint-compat tests; otherwise they skip gracefully.
DEFAULT_CKPT = os.path.join(REPO_ROOT, "checkpoints", "model.pth")
CKPT_PATH = os.environ.get("TREESEG_CKPT", DEFAULT_CKPT)
DEFAULT_CFG = os.path.join(REPO_ROOT, "configs", "default.yaml")
MULTI_CFG = os.path.join(REPO_ROOT, "configs", "multichannel_intensity.yaml")
