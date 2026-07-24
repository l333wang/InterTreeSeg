"""Interactive whole-scene inference over a folder of scene txt files.

Replaces the legacy ``main2.py`` ``testScene()`` path.

Example:
    python tools/infer_scene.py --cfg configs/default.yaml --log_dir my_run \
        --scene_dir /path/to/scenes --n_clicks 5 --force_divide
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from intertreeseg.config import load_config
from intertreeseg.engine import run_scene_folder
from intertreeseg.models import build_model
from intertreeseg.utils.logging import log2file
from intertreeseg.utils.runtime import parse_set_overrides, resolve_device, setup_run_paths


def parse_args():
    p = argparse.ArgumentParser("InterTreeSeg scene inference")
    p.add_argument("--cfg", type=str, default="configs/default.yaml")
    p.add_argument("--log_dir", type=str, default="ptv3", help="run name; checkpoint read from log/<name>/checkpoints")
    p.add_argument("--ckpt", type=str, default=None, help="explicit checkpoint path (overrides --log_dir)")
    p.add_argument("--testname", type=str, default=None, help="scene folder name under data.data_dir")
    p.add_argument("--scene_dir", type=str, default=None, help="explicit scene folder (overrides --testname)")
    p.add_argument("--data_root", type=str, default=".")
    p.add_argument("--n_clicks", type=int, default=None, help="override inference.n_clicks")
    p.add_argument("--grid_size", type=float, default=None, help="override cfg.grid_size")
    p.add_argument("--force_divide", action="store_true", help="set inference.force_divide=true")
    p.add_argument("--set", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    overrides = parse_set_overrides(args.set)
    if args.n_clicks is not None:
        overrides["inference.n_clicks"] = str(args.n_clicks)
    if args.grid_size is not None:
        overrides["grid_size"] = str(args.grid_size)
    if args.force_divide:
        overrides["inference.force_divide"] = "true"
    cfg = load_config(args.cfg, overrides=overrides)

    device = resolve_device()
    paths = setup_run_paths(args.log_dir, make_dirs=False)
    logger = log2file(log_dir="log", sub_dir=args.log_dir)

    ckpt = args.ckpt or paths.ckpt_path
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    logger.info(f"loaded checkpoint: {ckpt}")

    if args.scene_dir is not None:
        scene_dir = args.scene_dir
        tag = os.path.basename(os.path.normpath(scene_dir))
    else:
        if args.testname is None:
            raise SystemExit("provide --testname or --scene_dir")
        data_dir = cfg["data"]["data_dir"]
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(args.data_root, data_dir)
        scene_dir = os.path.join(data_dir, args.testname)
        tag = args.testname

    output_dir = os.path.join(paths.result_root, tag)
    logger.info(f"scene dir: {scene_dir}  ->  output: {output_dir}")
    run_scene_folder(scene_dir, model, cfg, output_dir, device, logger)


if __name__ == "__main__":
    main()
