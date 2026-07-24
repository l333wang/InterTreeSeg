"""Instance-level evaluation over merged result H5 files.

Replaces the standalone legacy ``Evaluate_treeSeg.py`` (which ran on import with
hardcoded absolute paths).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intertreeseg.evaluation import evaluate_result_h5, evaluate_result_dir


def parse_args():
    p = argparse.ArgumentParser("InterTreeSeg instance evaluation")
    p.add_argument("--pred", type=str, required=True, help="result .h5 file OR a directory of them")
    p.add_argument("--detection_iou", type=float, default=0.7)
    p.add_argument("--gt_col", type=int, default=3, help="GT instance-id column in the h5 point array")
    p.add_argument("--pred_col", type=int, default=4, help="predicted instance-id column")
    p.add_argument("--error_ply", type=str, default=None, help="optional error-colored PLY output (single file only)")
    return p.parse_args()


def main():
    args = parse_args()
    if os.path.isdir(args.pred):
        results = evaluate_result_dir(args.pred, args.detection_iou, args.gt_col, args.pred_col)
        detected = total = 0
        for name, m in results.items():
            print(f"\n===== {name} =====")
            print(m.format_report())
            detected += m.num_detected
            total += m.num_instances
        print(f"\n===== SUMMARY over {len(results)} scenes =====")
        print(f"total instances: {total}, detected (IoU>{args.detection_iou}): {detected}")
    else:
        m = evaluate_result_h5(args.pred, args.detection_iou, args.gt_col, args.pred_col, args.error_ply)
        print(m.format_report())


if __name__ == "__main__":
    main()
