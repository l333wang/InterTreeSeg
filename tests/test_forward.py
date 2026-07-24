"""Tier 3 + Tier 4: tiny GPU forward / one train step (default and multi-channel).

Requires a CUDA device (spconv + flash_attn are effectively CUDA-only). On CPU
these are skipped with a message.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import DEFAULT_CFG, MULTI_CFG  # noqa: E402

import torch  # noqa: E402

from intertreeseg.config import load_config  # noqa: E402
from intertreeseg.data.channels import ChannelSpec  # noqa: E402
from intertreeseg.engine.build import build_data_dict  # noqa: E402
from intertreeseg.models import build_model  # noqa: E402


def _make_batch(spec, B=2, N=1024, device="cuda"):
    """Synthetic block ``(B, N, 3 + in_channels)`` = [xyz | feat(incl clicks)]."""
    coord = torch.rand(B, N, 3)
    feat = torch.rand(B, N, spec.in_channels)
    points = torch.cat([coord, feat], dim=-1)
    labels = (torch.rand(B, N) > 0.5).long()
    return points, labels


def _run_one_step(cfg_path, base=None, expected_in=None):
    if not torch.cuda.is_available():
        print(f"SKIP forward ({cfg_path}): no CUDA device")
        return
    cfg = load_config(cfg_path, base=base)
    device = torch.device("cuda")
    spec = ChannelSpec.from_cfg(cfg["data"])
    if expected_in is not None:
        assert spec.in_channels == expected_in

    model = build_model(cfg).to(device)
    assert model.embedding.stem.conv.weight.shape[-1] == spec.in_channels

    points, labels = _make_batch(spec, device=device)
    data_dict = build_data_dict(points, cfg["grid_size"], device)
    logits = model(data_dict)
    assert logits.shape == (2 * 1024, cfg["model"]["num_classes"])
    assert torch.isfinite(logits).all()

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, cfg["model"]["num_classes"]), labels.view(-1).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    print(f"forward+step OK for {os.path.basename(cfg_path)} "
          f"(in_channels={spec.in_channels}, loss={loss.item():.4f})")


def test_forward_default():
    _run_one_step(DEFAULT_CFG, expected_in=2)


def test_forward_multichannel():
    _run_one_step(MULTI_CFG, base=DEFAULT_CFG, expected_in=3)


if __name__ == "__main__":
    test_forward_default()
    test_forward_multichannel()
    print("\nforward tests done.")
