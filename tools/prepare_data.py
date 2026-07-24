"""Offline data preparation: raw scene txt -> HDF5 training blocks.

Replaces the legacy ``trees_prep_2.py`` ``database()`` path. Each raw scene is a
delimited txt; for every tree instance we crop a bbox (+ ``delta`` buffer), center
xyz, keep the configured channels, sample to a fixed size, and write batched
``.h5`` files with datasets ``data`` (B, N, C) and ``label`` (B, N).

The defaults (``--inst_col 7 --sem_col 6``) match the FORinstance raw layout;
override them for other column conventions.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from tqdm import tqdm


def _sample(one_shot: np.ndarray, num_point: int) -> np.ndarray:
    """Up/down-sample a single cloud ``(N, C)`` to ``num_point`` points."""
    length = one_shot.shape[0]
    idx = np.arange(length)
    if length >= num_point:
        np.random.shuffle(idx)
        return one_shot[idx[0:num_point], :]
    batch = num_point // length
    els = num_point % length
    rep = np.tile(one_shot, (batch, 1))
    return np.concatenate((rep, one_shot[idx[0:els], :]), axis=0)


def _write_h5(data: np.ndarray, label: np.ndarray, filename: str) -> None:
    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data, compression="gzip", compression_opts=4, dtype="float32")
        f.create_dataset("label", data=label, compression="gzip", compression_opts=4, dtype="uint8")


def build_h5_blocks(
    in_dir: str,
    out_prefix: str,
    *,
    delimiter: str = ",",
    store_channels: int = 6,
    inst_col: int = 7,
    sem_col: int = 6,
    delta: float = 1.0,
    sample_size: int = 8192 * 4,
    min_points: int = 100,
    batch_size: int = 20,
) -> int:
    """Convert every scene txt in ``in_dir`` into batched training h5 files.

    ``store_channels`` columns (0..store_channels-1) are kept per point, xyz is
    centered per block, and the per-point semantic label (column ``sem_col``) is
    stored as the ``label`` dataset.
    """
    scene_names = sorted(os.listdir(in_dir))
    num_scene = len(scene_names)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    total_instances = 0
    for h_id in range(num_scene // batch_size + 1):
        bt_start = h_id * batch_size
        bt_end = min(bt_start + batch_size, num_scene)
        if bt_start >= bt_end:
            continue

        total_blocks = []
        for i in tqdm(range(bt_start, bt_end), desc=f"batch {h_id}"):
            one_shot = np.loadtxt(os.path.join(in_dir, scene_names[i]), delimiter=delimiter)
            points = one_shot[:, 0:3]
            inst_label = one_shot[:, inst_col]
            sem_label = one_shot[:, sem_col]

            unique_ids = np.unique(inst_label)
            unique_ids = unique_ids[unique_ids > 0]
            for j in unique_ids:
                idx = np.squeeze(np.argwhere(inst_label == j))
                if np.atleast_1d(idx).shape[0] < min_points:
                    continue
                total_instances += 1

                obj_xyz = points[idx, :]
                obj_max = np.max(obj_xyz, axis=0)
                obj_min = np.min(obj_xyz, axis=0)
                cond = (
                    (points[:, 0] <= obj_max[0] + delta) & (points[:, 0] >= obj_min[0] - delta)
                    & (points[:, 1] <= obj_max[1] + delta) & (points[:, 1] >= obj_min[1] - delta)
                    & (points[:, 2] <= obj_max[2] + delta) & (points[:, 2] >= obj_min[2] - delta)
                )

                new_block = one_shot[cond, :store_channels].astype(np.float64)
                sem_in_block = sem_label[cond].reshape(-1, 1)
                new_block[:, 0:3] = new_block[:, 0:3] - np.mean(new_block[:, 0:3], axis=0)  # center xyz
                new_block = np.concatenate((new_block, sem_in_block), axis=1)
                new_block = _sample(new_block, sample_size)
                total_blocks.append(np.expand_dims(new_block, axis=0))

        if not total_blocks:
            continue
        total_blocks = np.concatenate(total_blocks, axis=0)
        data = total_blocks[:, :, 0:store_channels]
        label = total_blocks[:, :, -1]
        _write_h5(data, label, f"{out_prefix}{h_id}.h5")

    print(f"[Info] total instances processed: {total_instances}")
    return total_instances


def parse_args():
    p = argparse.ArgumentParser("InterTreeSeg data preparation (txt -> h5)")
    p.add_argument("--src", required=True, help="input folder of scene .txt files")
    p.add_argument("--dst", required=True, help="output prefix, e.g. data/trees/train/FOR_train4x")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--store_channels", type=int, default=6)
    p.add_argument("--inst_col", type=int, default=7)
    p.add_argument("--sem_col", type=int, default=6)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--sample_size", type=int, default=8192 * 4)
    p.add_argument("--min_points", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    build_h5_blocks(
        args.src,
        args.dst,
        delimiter=args.delimiter,
        store_channels=args.store_channels,
        inst_col=args.inst_col,
        sem_col=args.sem_col,
        delta=args.delta,
        sample_size=args.sample_size,
        min_points=args.min_points,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
