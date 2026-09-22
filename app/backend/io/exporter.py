"""Export annotation results: per-point instance labels + a per-tree attribute table."""
from __future__ import annotations

import csv
import io

import numpy as np

_CSV_COLUMNS = [
    "instance_id", "species", "height", "crown_width",
    "stem_x", "stem_y", "dbh", "health_status", "point_count", "notes",
]


def export_points_txt(session) -> bytes:
    """Full-resolution original points (x y z ...) with an appended instance-id
    column. instance id 0 means unlabeled. Every original point is included.
    """
    pts = session.full_points
    labels = session.full_labels.reshape(-1, 1).astype(np.float64)
    merged = np.concatenate([pts, labels], axis=1)
    buf = io.BytesIO()
    np.savetxt(buf, merged, fmt="%.6f")
    return buf.getvalue()


def export_attributes_csv(session) -> bytes:
    """One row per tree; instance_id matches the label column of the points file."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for inst in sorted(session.instances.values(), key=lambda i: i.id):
        row = inst.to_dict()
        row["instance_id"] = inst.id
        writer.writerow(row)
    return out.getvalue().encode("utf-8")


_EVAL_COLUMNS = [
    "instance_id", "clicks", "time_s", "point_count",
    "matched_gt", "gt_point_count", "iou",
]


def export_evaluation_csv(session) -> bytes:
    """Per-tree evaluation (clicks / time / IoU vs GT) + a summary block.

    Columns: instance_id, clicks, time_s, point_count, matched_gt,
    gt_point_count, iou. A trailing '# summary' section lists the session totals.
    """
    ev = session.evaluation()
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=_EVAL_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for rec in ev["records"]:
        writer.writerow(rec)
    out.write("\n# summary\n")
    for k, v in ev["summary"].items():
        out.write(f"{k},{'' if v is None else v}\n")
    out.write(f"source,{session.source_path}\n")
    return out.getvalue().encode("utf-8")
