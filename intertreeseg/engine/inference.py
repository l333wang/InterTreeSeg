"""Interactive whole-scene inference (migrated from ``infer_onescene`` / ``testScene``).

For each scene: preprocess into per-tree click-augmented blocks, run a first
segmentation pass, iteratively add correction clicks where IoU is low, then merge
the per-tree masks back into a full-scene instance labeling.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from ..data.channels import ChannelSpec
from ..data.clicks import reclick
from ..data.scene_dataset import load_scene_txt
from ..evaluation.merge import merge_scene
from .tester import evaluate


def infer_one_scene(
    scene_path: str,
    model: nn.Module,
    cfg: Mapping[str, Any],
    output_dir: str,
    device,
    logger: Optional[logging.Logger] = None,
) -> Optional[float]:
    """Run the interactive segmentation + merge pipeline on a single scene file.

    Returns the scene-level instance IoU, or ``None`` on error.
    """
    if logger is None:
        logger = logging.getLogger()

    spec = ChannelSpec.from_cfg(cfg["data"])
    inf_cfg = cfg["inference"]
    thr = float(inf_cfg.get("reclick_thr", 0.99))
    n_clicks = int(inf_cfg.get("n_clicks", 1))
    sigma = float(cfg["data"].get("clicks", {}).get("sigma", 0.01))
    batch_size = int(cfg["batch_size"])
    criterion = nn.CrossEntropyLoss()
    scene_name = os.path.basename(scene_path)

    try:
        t0 = time.time()
        dataset, _clicks, pos_labels, one_shot = load_scene_txt(scene_path, spec, cfg)
        logger.info(f"Data pre-processing time: {time.time() - t0:.4f}s")

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=False
        )
        result = evaluate(model, loader, criterion, device, cfg)
        pred, logits = result.pred, result.logits

        click_num = np.ones((pred.shape[0]), dtype=int)
        with np.printoptions(precision=2, suppress=True):
            logger.info(
                f"[click 1] oa:{result.oa:.2f} macc:{result.macc:.2f} miou:{result.miou:.2f} "
                f"ious:{result.ious} avg_click:{np.mean(click_num):.2f} time:{result.elapsed:.4f}"
            )

        new_test = dataset
        del dataset
        gc.collect()

        for i in range(1, n_clicks):
            new_test, click_num, _clicks = reclick(pred, new_test, click_num, spec, thr, sigma)
            loader = torch.utils.data.DataLoader(
                new_test, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=False
            )
            result = evaluate(model, loader, criterion, device, cfg)
            pred, logits = result.pred, result.logits
            with np.printoptions(precision=2, suppress=True):
                logger.info(
                    f"[click {i + 1}] oa:{result.oa:.2f} macc:{result.macc:.2f} "
                    f"miou:{result.miou:.2f} ious:{result.ious} "
                    f"avg_click:{np.mean(click_num):.2f} time:{result.elapsed:.4f}"
                )

        return merge_scene(one_shot, scene_name, pred, pos_labels, logits, cfg, output_dir, logger)
    except Exception as e:  # noqa: BLE001 - skip a bad scene, keep the batch going
        logger.error(f"Error processing scene {scene_path}: {e}")
        return None


def run_scene_folder(
    scene_dir: str,
    model: nn.Module,
    cfg: Mapping[str, Any],
    output_dir: str,
    device,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Run interactive inference over every scene file in ``scene_dir``."""
    if logger is None:
        logger = logging.getLogger()

    for scene_name in sorted(os.listdir(scene_dir)):
        scene_path = os.path.join(scene_dir, scene_name)
        try:
            infer_one_scene(scene_path, model, cfg, output_dir, device, logger)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except MemoryError:
            logger.error(f"CPU MemoryError for scene {scene_name}; skipping.")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"GPU OOM for scene {scene_name}; skipping.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                logger.error(f"RuntimeError for scene {scene_name}: {e}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Exception for scene {scene_name}: {e}")
