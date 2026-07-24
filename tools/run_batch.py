"""Batch driver: run scene inference over several test sets sequentially.

Replaces the legacy ``run_tasks.py`` (which hardcoded the test-set list and shell
commands). Test sets and options are passed on the CLI.

Example:
    python tools/run_batch.py --log_dir my_run --testnames setA setB setC --n_clicks 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser("InterTreeSeg batch inference")
    p.add_argument("--cfg", default="configs/default.yaml")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--testnames", nargs="+", required=True, help="scene folder names to run")
    p.add_argument("--n_clicks", type=int, default=5)
    p.add_argument("--grid_size", type=float, default=0.02)
    p.add_argument("--force_divide", action="store_true")
    p.add_argument("--data_root", default=".")
    return p.parse_args()


def main():
    args = parse_args()
    for name in args.testnames:
        cmd = [
            sys.executable, os.path.join(TOOLS_DIR, "infer_scene.py"),
            "--cfg", args.cfg,
            "--log_dir", args.log_dir,
            "--testname", name,
            "--n_clicks", str(args.n_clicks),
            "--grid_size", str(args.grid_size),
            "--data_root", args.data_root,
        ]
        if args.force_divide:
            cmd.append("--force_divide")
        print(f"\n=== running {name} ===\n{' '.join(cmd)}")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
