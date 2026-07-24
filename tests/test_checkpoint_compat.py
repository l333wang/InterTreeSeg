"""Tier 1 + Tier 5: checkpoint backward-compat and the multi-channel guard (CPU).

* default config builds a model that strict-loads the released 2-channel weights;
* a 3-channel (multichannel) model must NOT strict-load the 2-channel weights.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import CKPT_PATH, DEFAULT_CFG, MULTI_CFG, REPO_ROOT  # noqa: E402

import torch  # noqa: E402

from intertreeseg.config import load_config  # noqa: E402
from intertreeseg.models import build_model  # noqa: E402


def test_default_strict_load():
    cfg = load_config(DEFAULT_CFG)
    model = build_model(cfg)
    assert model.embedding.stem.conv.weight.shape[-1] == 2  # spconv layout (out,k,k,k,in)
    assert model.cls_head[0].out_features == 2
    if not os.path.exists(CKPT_PATH):
        print(f"SKIP strict-load: checkpoint not found at {CKPT_PATH}")
        return
    sd = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(sd, strict=True)  # must not raise
    print("Tier1: default config strict-loads the released checkpoint OK")


def test_multichannel_cannot_load_2ch():
    cfg = load_config(MULTI_CFG, base=DEFAULT_CFG)
    model = build_model(cfg)
    assert model.embedding.stem.conv.weight.shape[-1] == 3
    if not os.path.exists(CKPT_PATH):
        print(f"SKIP guard: checkpoint not found at {CKPT_PATH}")
        return
    sd = torch.load(CKPT_PATH, map_location="cpu")
    raised = False
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        raised = True
    assert raised, "3-channel model should NOT strict-load a 2-channel checkpoint"
    print("Tier5: 3-channel model correctly refuses the 2-channel checkpoint")


if __name__ == "__main__":
    test_default_strict_load()
    test_multichannel_cannot_load_2ch()
    print("\ncheckpoint-compat tests passed.")
