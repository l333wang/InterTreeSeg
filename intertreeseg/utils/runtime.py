"""Small runtime helpers shared by the CLI tools (paths, device, seeding)."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RunPaths:
    log_dir: str          # log/<name>
    ckpt_path: str        # log/<name>/checkpoints/<name>.pth
    result_root: str      # result/<name>


def setup_run_paths(log_name: str, base: str = ".", make_dirs: bool = True) -> RunPaths:
    """Compute (and optionally create) the standard log / checkpoint / result paths."""
    log_dir = os.path.join(base, "log", log_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"{log_name}.pth")
    result_root = os.path.join(base, "result", log_name)
    if make_dirs:
        os.makedirs(ckpt_dir, exist_ok=True)
    return RunPaths(log_dir=log_dir, ckpt_path=ckpt_path, result_root=result_root)


def parse_set_overrides(pairs: List[str]) -> dict:
    """Parse ``--set key=value`` pairs into a dotted-override dict."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--set expects key=value, got: {p!r}")
        key, val = p.split("=", 1)
        out[key.strip()] = val.strip()
    return out
