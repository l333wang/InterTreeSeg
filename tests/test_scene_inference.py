"""Tier 6: end-to-end synthetic-scene interactive inference + instance eval (GPU).

Builds a tiny synthetic forest scene, runs the interactive segmentation + merge
pipeline, and checks that ply/h5 outputs and instance metrics are produced. Runs
once for the default (2-channel) config and once for the intensity (3-channel)
config, proving the click channel indices track the number of feature channels.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import CKPT_PATH, DEFAULT_CFG, MULTI_CFG  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from intertreeseg.config import load_config  # noqa: E402
from intertreeseg.data.channels import ChannelSpec  # noqa: E402
from intertreeseg.engine.inference import infer_one_scene  # noqa: E402
from intertreeseg.evaluation.instance_eval import evaluate_result_h5  # noqa: E402
from intertreeseg.models import build_model  # noqa: E402


def _make_scene(path, n_trees=3, pts_per_tree=400):
    """Write a synthetic scene txt: columns x y z intensity treeID label2."""
    rng = np.random.default_rng(0)
    rows = []
    for tid in range(1, n_trees + 1):
        center = np.array([tid * 3.0, 0.0, 0.0])
        xyz = center + rng.normal(0, 0.5, size=(pts_per_tree, 3))
        intensity = rng.uniform(0, 1, size=(pts_per_tree, 1))
        tree_id = np.full((pts_per_tree, 1), tid)
        label2 = np.zeros((pts_per_tree, 1))
        rows.append(np.hstack([xyz, intensity, tree_id, label2]))
    scene = np.vstack(rows)
    np.savetxt(path, scene, fmt="%.5f")


def _run(cfg_path, base, expected_in, load_ckpt):
    if not torch.cuda.is_available():
        print(f"SKIP scene inference ({os.path.basename(cfg_path)}): no CUDA device")
        return
    cfg = load_config(cfg_path, base=base, overrides={"inference.n_clicks": "2"})
    spec = ChannelSpec.from_cfg(cfg["data"])
    assert spec.in_channels == expected_in

    device = torch.device("cuda")
    model = build_model(cfg).to(device)
    if load_ckpt and os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)

    with tempfile.TemporaryDirectory() as tmp:
        scene_path = os.path.join(tmp, "synthetic_scene.txt")
        _make_scene(scene_path)
        out_dir = os.path.join(tmp, "out")
        iou = infer_one_scene(scene_path, model, cfg, out_dir, device)
        assert iou is not None, "inference returned None (pipeline error)"

        ply = os.path.join(out_dir, "synthetic_scene.ply")
        h5 = os.path.join(out_dir, "synthetic_scene.h5")
        assert os.path.exists(ply), "missing ply output"
        assert os.path.exists(h5), "missing h5 output"

        metrics = evaluate_result_h5(h5, detection_iou=cfg["inference"]["detection_iou"])
        print(
            f"scene inference OK ({os.path.basename(cfg_path)}, in={expected_in}): "
            f"scene_iou={iou:.3f} instances={metrics.num_instances} "
            f"detected={metrics.num_detected} mean_iou={metrics.mean_iou:.3f}"
        )


def test_scene_default():
    _run(DEFAULT_CFG, base=None, expected_in=2, load_ckpt=True)


def test_scene_multichannel():
    _run(MULTI_CFG, base=DEFAULT_CFG, expected_in=3, load_ckpt=False)


if __name__ == "__main__":
    test_scene_default()
    test_scene_multichannel()
    print("\nscene inference tests done.")
