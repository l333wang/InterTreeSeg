"""Tier 2: channel-contract math and multi-channel feature assembly (CPU/numpy).

Run with ``pytest`` or directly (``python tests/test_channels.py``).
"""

import numpy as np

from intertreeseg.data.channels import ChannelSpec
from intertreeseg.data.normalize import normalize_block
from intertreeseg.data.augment import dataaugment_z
from intertreeseg.data.clicks import gen_clicks, reclick


def test_default_spec_matches_legacy():
    spec = ChannelSpec.from_cfg({"coord_channels": [0, 1, 2], "feat_channels": [], "clicks": {"n_channels": 2}})
    assert spec.F == 0
    assert spec.in_channels == 2
    assert spec.click_start == 3
    assert spec.block_width == 5


def test_intensity_spec():
    spec = ChannelSpec.from_cfg(
        {"coord_channels": [0, 1, 2], "feat_channels": [3], "feat_norm": ["standardize"], "clicks": {"n_channels": 2}}
    )
    assert spec.F == 1
    assert spec.in_channels == 3
    assert spec.click_start == 4
    assert spec.block_width == 6


def test_normalize_isolates_intensity():
    spec = ChannelSpec(coord_channels=(0, 1, 2), feat_channels=(3,), feat_norm=("none",))
    rng = np.random.default_rng(0)
    block = rng.uniform(-5, 5, size=(200, 4)).astype(np.float32)
    intensity_before = block[:, 3].copy()
    out = normalize_block(block, spec)
    # xyz normalized to [0, 1] per axis
    assert out[:, 0:3].min() >= -1e-6 and out[:, 0:3].max() <= 1 + 1e-6
    # intensity with 'none' norm passes through unchanged
    assert np.allclose(out[:, 3], intensity_before)


def test_augment_z_leaves_feats_untouched():
    spec_F = 2
    rng = np.random.default_rng(1)
    blocks = rng.uniform(-3, 3, size=(4, 500, 3 + spec_F)).astype(np.float64)
    feats_before = blocks[:, :, 3:].copy()
    out = dataaugment_z(blocks, axis="z")
    # feature columns are byte-identical after rotation
    assert np.allclose(out[:, :, 3:], feats_before)
    # z coordinate is preserved by a z-rotation (about bbox center) up to translation-back
    # (checking it changed x/y but not the feat block is the key contract)
    assert not np.allclose(out[:, :, 0:2], blocks[:, :, 0:2])


def test_gen_clicks_layout_default():
    spec = ChannelSpec()  # default: F=0, K=2
    rng = np.random.default_rng(2)
    blocks = rng.uniform(0, 1, size=(3, 400, 3)).astype(np.float32)
    labels = (rng.uniform(0, 1, size=(3, 400)) > 0.5).astype(np.int64)
    out_blocks, out_labels = gen_clicks(blocks, labels, spec, sigma=0.01)
    assert out_blocks.shape[-1] == spec.block_width == 5  # [x,y,z,pch,nch]
    assert out_blocks.shape[0] == out_labels.shape[0]


def test_gen_clicks_layout_intensity():
    spec = ChannelSpec(feat_channels=(3,), feat_norm=("none",))
    rng = np.random.default_rng(3)
    blocks = rng.uniform(0, 1, size=(3, 400, 4)).astype(np.float32)  # [x,y,z,intensity]
    labels = (rng.uniform(0, 1, size=(3, 400)) > 0.5).astype(np.int64)
    out_blocks, _ = gen_clicks(blocks, labels, spec, sigma=0.01)
    assert out_blocks.shape[-1] == spec.block_width == 6  # [x,y,z,intensity,pch,nch]


def test_reclick_preserves_feats_and_targets_click_cols():
    import torch
    import torch.utils.data as Data

    spec = ChannelSpec(feat_channels=(3,), feat_norm=("none",))  # click_start = 4
    rng = np.random.default_rng(4)
    N = 300
    # assembled block: [x,y,z,intensity,pch,nch]
    block = rng.uniform(0, 1, size=(1, N, 6)).astype(np.float32)
    intensity_before = block[0, :, 3].copy()
    labels = (rng.uniform(0, 1, size=(1, N)) > 0.5).astype(np.int64)
    dataset = Data.TensorDataset(torch.tensor(block), torch.tensor(labels))
    # a prediction guaranteed to have low IoU so a correction click is added
    pred = np.zeros((1, N), dtype=np.int64)
    new_ds, n_c, clicks = reclick(pred, dataset, np.ones(1, dtype=int), spec, thr=0.99, sigma=0.01)
    new_block = np.array(new_ds[0][0])
    # intensity column (index 3) is preserved bit-for-bit
    assert np.allclose(new_block[:, 3], intensity_before)
    # a click was added (n_c incremented) and layout width preserved
    assert new_block.shape[-1] == spec.block_width == 6
    assert n_c[0] == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} Tier-2 channel tests passed.")
