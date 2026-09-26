"""Challenge metric: macro F0.5 over Source-1 entities (singletons included)."""

import numpy as np


def macro_f05(q, is_true, pred, gt_count):
    """q: entity index per candidate pair; pred: predicted-match mask per pair;
    gt_count: number of true matches per entity (from the ground truth, so true
    matches missed by blocking still count as recall losses)."""
    n = len(gt_count)
    tp = np.bincount(q[pred & is_true], minlength=n)
    fp = np.bincount(q[pred & ~is_true], minlength=n)
    npred = tp + fp
    p = np.divide(tp, npred, out=np.zeros(n), where=npred > 0)
    r = np.divide(tp, gt_count, out=np.zeros(n), where=gt_count > 0)
    f = np.divide(1.25 * p * r, 0.25 * p + r, out=np.zeros(n), where=(p + r) > 0)
    f = np.where(gt_count == 0, (npred == 0).astype(float), f)
    return f


def best_threshold(q, is_true, prob, gt_count, grid=None):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best = (-1.0, 0.5)
    for t in grid:
        f = macro_f05(q, is_true, prob >= t, gt_count).mean()
        if f > best[0]:
            best = (float(f), float(t))
    return best
