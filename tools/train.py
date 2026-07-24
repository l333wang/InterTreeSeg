"""Train the interactive tree segmentation model.

Replaces the legacy ``main2.py`` ``main()`` path.

Example (reproducing the released architecture):
    python tools/train.py --cfg configs/default.yaml --log_dir my_run --loss combined
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from intertreeseg.config import load_config, dump_config
from intertreeseg.data import build_dataloaders
from intertreeseg.engine import build_optimizer, build_scheduler, evaluate, train_one_epoch
from intertreeseg.losses import LOSS_CHOICES, build_loss
from intertreeseg.models import build_model
from intertreeseg.utils.logging import log2file
from intertreeseg.utils.runtime import parse_set_overrides, resolve_device, set_seed, setup_run_paths


def parse_args():
    p = argparse.ArgumentParser("InterTreeSeg training")
    p.add_argument("--cfg", type=str, default="configs/default.yaml", help="config YAML")
    p.add_argument("--log_dir", type=str, default="ptv3", help="run name under log/")
    p.add_argument("--loss", type=str, default=None, choices=list(LOSS_CHOICES),
                   help="override cfg.loss")
    p.add_argument("--grid_size", type=float, default=None, help="override cfg.grid_size")
    p.add_argument("--data_root", type=str, default=".", help="prefix for a relative data.data_dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], help="dotted config overrides, e.g. batch_size=32")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = parse_set_overrides(args.set)
    if args.loss is not None:
        overrides["loss"] = args.loss
    if args.grid_size is not None:
        overrides["grid_size"] = str(args.grid_size)
    cfg = load_config(args.cfg, overrides=overrides)

    set_seed(args.seed)
    device = resolve_device()
    paths = setup_run_paths(args.log_dir)
    logger = log2file(log_dir="log", sub_dir=args.log_dir)
    shutil.copy(args.cfg, paths.log_dir)
    dump_config(cfg, os.path.join(paths.log_dir, "resolved_config.yaml"))

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)

    train_loader, val_loader = build_dataloaders(cfg, augment=cfg["data"]["augment"]["enabled"], data_root=args.data_root)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    num_epochs = int(cfg["epoch"])
    miou_best = 0.0
    for epoch in range(num_epochs):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler, cfg, epoch)
        ev = evaluate(model, val_loader, criterion, device, cfg)
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch} LR {lr:.6f} train_miou {tr.miou:.2f} train_loss {tr.loss:.6f} "
            f"val_miou {ev.miou:.2f} val_loss {ev.loss:.6f} best_val_miou {miou_best:.2f}"
        )
        if miou_best < ev.miou:
            torch.save(model.state_dict(), paths.ckpt_path)
            logger.info(
                f"Saved better ckpt @E{epoch}: val_miou {ev.miou:.2f} val_macc {ev.macc:.2f} "
                f"val_oa {ev.oa:.2f}\nious: {ev.ious}"
            )
            miou_best = ev.miou


if __name__ == "__main__":
    main()
