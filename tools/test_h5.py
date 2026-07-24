"""Evaluate a trained checkpoint on an h5 validation set.

Replaces the legacy ``main2.py`` ``onlyTest()`` path (which crashed on an
8-vs-9 tuple unpack — fixed here via the EvalResult dataclass).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from intertreeseg.config import load_config
from intertreeseg.data import build_dataloaders
from intertreeseg.engine import evaluate
from intertreeseg.models import build_model
from intertreeseg.utils.runtime import parse_set_overrides, resolve_device, setup_run_paths


def parse_args():
    p = argparse.ArgumentParser("InterTreeSeg h5 evaluation")
    p.add_argument("--cfg", type=str, default="configs/default.yaml")
    p.add_argument("--log_dir", type=str, default="ptv3", help="run name; checkpoint read from log/<name>/checkpoints")
    p.add_argument("--ckpt", type=str, default=None, help="explicit checkpoint path (overrides --log_dir)")
    p.add_argument("--data_root", type=str, default=".")
    p.add_argument("--set", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg, overrides=parse_set_overrides(args.set))
    device = resolve_device()

    ckpt = args.ckpt or setup_run_paths(args.log_dir, make_dirs=False).ckpt_path
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    print(f"loaded checkpoint: {ckpt}")

    _, val_loader = build_dataloaders(cfg, augment=False, data_root=args.data_root)
    result = evaluate(model, val_loader, nn.CrossEntropyLoss(), device, cfg)
    print(f"test mIoU: {result.miou:.2f}, macc: {result.macc:.2f}, oa: {result.oa:.2f}")
    print(f"iou per class: {result.ious}")


if __name__ == "__main__":
    main()
