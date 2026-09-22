"""Global configuration for the annotation backend.

Kept intentionally small; values can later move to env vars / a settings file
without touching call sites.
"""
from __future__ import annotations

import os

# Target render point count: clouds larger than this are evenly sub-sampled to
# EXACTLY this many points for rendering (bounded render cost). Labels, inference
# and export always use the full-resolution cloud, so quality is unaffected.
MAX_RENDER_POINTS = int(os.environ.get("ANNO_MAX_RENDER_POINTS", 1_500_000))

# Which InferenceService implementation to use: "mock" | "ptv3".
# "mock" needs no torch and lets the whole UI loop run end-to-end.
INFERENCE_BACKEND = os.environ.get("ANNO_INFERENCE_BACKEND", "mock")

# Real PTv3 model (used when INFERENCE_BACKEND == "ptv3"). The model code comes
# from the packaged `intertreeseg` library (this repo); only the checkpoint path
# is configurable. Download the checkpoint per checkpoints/README.md.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PTV3_CHECKPOINT = os.environ.get(
    "ANNO_PTV3_CKPT",
    os.path.join(_REPO_ROOT, "checkpoints", "E100_B60_FOR05x4_novali_newmodel512.pth"),
)
PTV3_GRID_SIZE = float(os.environ.get("ANNO_PTV3_GRID", "0.02"))
PTV3_NUM_POINT = int(os.environ.get("ANNO_PTV3_NUMPOINT", "8192"))
PTV3_CLICK_SIGMA = float(os.environ.get("ANNO_PTV3_SIGMA", "0.01"))  # gaussian click width (world metres)

# Column layout of the 6-column .txt point clouds: x y z feat treeID semanticCode
TXT_COLS = {"x": 0, "y": 1, "z": 2, "feat": 3, "gt_instance": 4, "gt_semantic": 5}

# Ground-truth instance-id column (0-based) used for IoU evaluation. In the
# FOR/NIBIO test scenes the per-tree id (treeID) is column 4; column 5 is a
# semantic class code, NOT an instance id. This matches the library's
# configs/default.yaml `label_channel: 4`. Override for other layouts.
GT_INSTANCE_COL = int(os.environ.get("ANNO_GT_INSTANCE_COL", "4"))
GT_IGNORE_ID = int(os.environ.get("ANNO_GT_IGNORE_ID", "0"))  # 0 = ground / unlabeled

# Default tree-species dropdown options (editable in the UI).
DEFAULT_SPECIES = ["Unknown", "Pine", "Spruce", "Birch", "Oak", "Beech", "Larch"]

# Default growth/health status dropdown options.
DEFAULT_HEALTH_STATUS = ["Alive", "Dead-standing", "Leaning", "Diseased", "Broken"]
