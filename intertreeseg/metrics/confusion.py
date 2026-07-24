"""Confusion matrix and running-average meter (migrated; dead SegMetric dropped)."""

from __future__ import annotations

import logging

import torch


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMatrix:
    """Accumulate a confusion matrix for a classification task.

    ``ignore_index`` only supports an index ``< 0`` or ``> num_classes``.
    """

    def __init__(self, num_classes, ignore_index=None):
        self.value = 0
        self.num_classes = num_classes
        self.virtual_num_classes = num_classes + 1 if ignore_index is not None else num_classes
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, true):
        """Update the confusion matrix with the given predictions."""
        true = true.flatten()
        pred = pred.flatten()
        if self.ignore_index is not None:
            if (true == self.ignore_index).sum() > 0:
                pred[true == self.ignore_index] = self.virtual_num_classes - 1
                true[true == self.ignore_index] = self.virtual_num_classes - 1
        unique_mapping = true.flatten() * self.virtual_num_classes + pred.flatten()
        bins = torch.bincount(unique_mapping, minlength=self.virtual_num_classes**2)
        self.value += bins.view(self.virtual_num_classes, self.virtual_num_classes)[
            : self.num_classes, : self.num_classes
        ]

    def reset(self):
        self.value = 0

    @property
    def tp(self):
        return self.value.diag()

    @property
    def actual(self):
        return self.value.sum(dim=1)

    @property
    def predicted(self):
        return self.value.sum(dim=0)

    @property
    def fn(self):
        return self.actual - self.tp

    @property
    def fp(self):
        return self.predicted - self.tp

    @property
    def tn(self):
        actual = self.actual
        predicted = self.predicted
        return actual.sum() + self.tp - (actual + predicted)

    @property
    def count(self):
        return self.value.sum(dim=1)

    @property
    def frequency(self):
        count = self.value.sum(dim=1)
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        return self.value.sum()

    @property
    def overall_accuray(self):  # noqa: N802 (legacy spelling kept for compatibility)
        return self.tp.sum() / self.total

    @property
    def union(self):
        return self.value.sum(dim=0) + self.value.sum(dim=1) - self.value.diag()

    def all_acc(self):
        return self.cal_acc(self.tp, self.count)

    @staticmethod
    def cal_acc(tp, count):
        acc_per_cls = tp / count.clamp(min=1) * 100
        over_all_acc = tp.sum() / count.sum() * 100
        macc = torch.mean(acc_per_cls)
        return macc.item(), over_all_acc.item(), acc_per_cls.cpu().numpy()

    @staticmethod
    def print_acc(accs):
        out = "\n    Class  " + "   Acc  "
        for i, values in enumerate(accs):
            out += "\n" + str(i).rjust(8) + f"{values.item():.2f}".rjust(8)
        out += "\n" + "-" * 20
        out += "\n" + "   Mean  " + f"{torch.mean(accs).item():.2f}".rjust(8)
        logging.info(out)

    def all_metrics(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        iou_per_cls = tp / (tp + fp + fn).clamp(min=1) * 100
        acc_per_cls = tp / self.count.clamp(min=1) * 100
        over_all_acc = tp.sum() / self.total * 100
        miou = torch.mean(iou_per_cls)
        macc = torch.mean(acc_per_cls)
        return (
            miou.item(),
            macc.item(),
            over_all_acc.item(),
            iou_per_cls.cpu().numpy(),
            acc_per_cls.cpu().numpy(),
        )
