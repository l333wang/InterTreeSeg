"""Instance-level segmentation metrics (migrated from ``Evaluate_treeSeg``).

Given per-point ground-truth and predicted instance IDs, computes per-instance
IoU, a detection count at an IoU threshold (default 0.7), and macro / weighted
precision, recall, F1 and IoU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class InstanceMetrics:
    instance_ids: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    iou: np.ndarray
    support: np.ndarray
    accuracy: float
    detection_iou: float
    num_instances: int
    num_detected: int          # instances with IoU > detection_iou
    mean_iou: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_iou: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    weighted_iou: float

    def format_report(self) -> str:
        lines = ["Classification Report with IoU:"]
        header = f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'IoU':<10} {'Support':<10}"
        lines.append(header)
        for i, inst in enumerate(self.instance_ids):
            lines.append(
                f"{int(inst):<10} {self.precision[i]:<10.3f} {self.recall[i]:<10.3f} "
                f"{self.f1[i]:<10.3f} {self.iou[i]:<10.3f} {int(self.support[i]):<10}"
            )
        lines += [
            "",
            "Macro Average:",
            f"{'Precision':<10} {self.macro_precision:.3f}",
            f"{'Recall':<10} {self.macro_recall:.3f}",
            f"{'F1-score':<10} {self.macro_f1:.3f}",
            f"{'IoU':<10} {self.macro_iou:.3f}",
            "",
            "Weighted Average:",
            f"{'Precision':<10} {self.weighted_precision:.3f}",
            f"{'Recall':<10} {self.weighted_recall:.3f}",
            f"{'F1-score':<10} {self.weighted_f1:.3f}",
            f"{'IoU':<10} {self.weighted_iou:.3f}",
            "",
            f"Overall Accuracy: {self.accuracy:.3f}",
            f"Total instances: {self.num_instances}",
            f"Instances with IoU > {self.detection_iou}: {self.num_detected}",
            f"Mean IoU: {self.mean_iou:.4f}",
        ]
        return "\n".join(lines)


def instance_metrics(gt: np.ndarray, pred: np.ndarray, detection_iou: float = 0.7) -> InstanceMetrics:
    """Compute instance-level metrics from per-point GT / predicted instance IDs."""
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(gt, pred, average=None, zero_division=0)
    accuracy = accuracy_score(gt, pred)

    unique_instances = np.unique(gt)
    iou_scores: List[float] = []
    for instance_id in unique_instances:
        mask = gt == instance_id
        inter = np.sum(gt[mask] == pred[mask])
        union = np.sum(mask) + np.sum(mask) - inter  # per-point agreement IoU (legacy definition)
        iou_scores.append(inter / union if union != 0 else 0.0)
    iou_scores = np.array(iou_scores)

    total_support = np.sum(support) if np.sum(support) > 0 else 1
    # per-class arrays from sklearn cover labels present in gt or pred; align to gt instances
    # for the weighted-by-support IoU we use gt-instance support.
    gt_support = np.array([np.sum(gt == i) for i in unique_instances])
    total_gt_support = np.sum(gt_support) if np.sum(gt_support) > 0 else 1

    return InstanceMetrics(
        instance_ids=unique_instances,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou_scores,
        support=support,
        accuracy=float(accuracy),
        detection_iou=detection_iou,
        num_instances=len(unique_instances),
        num_detected=int(np.sum(iou_scores > detection_iou)),
        mean_iou=float(np.mean(iou_scores)) if len(iou_scores) else 0.0,
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        macro_iou=float(np.mean(iou_scores)) if len(iou_scores) else 0.0,
        weighted_precision=float(np.sum(precision * support) / total_support),
        weighted_recall=float(np.sum(recall * support) / total_support),
        weighted_f1=float(np.sum(f1 * support) / total_support),
        weighted_iou=float(np.sum(iou_scores * gt_support) / total_gt_support) if len(iou_scores) else 0.0,
    )
