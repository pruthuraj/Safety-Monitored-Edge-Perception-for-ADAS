"""Confidence calibration for detection outputs (SR-01, Week 4, EXP-006).

Pipeline: match detections to ground truth (greedy one-to-one, same class,
IoU >= 0.50) -> binary correctness labels -> fit one scalar temperature T on
detection confidences (calibration-fit subset, kitti-val only) -> report ECE
before/after on the calibration-report subset with 15 equal-width bins.

Temperature scaling operates in logit space: conf -> logit -> logit/T ->
sigmoid. T is fitted by minimizing binary NLL; T=1 leaves confidences
unchanged, T>1 softens (overconfidence correction).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-7


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boxes in [x1, y1, x2, y2] pixel coordinates."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_detections(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_confs: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_thr: float = 0.50,
) -> np.ndarray:
    """Greedy one-to-one matching of predictions to ground truth.

    Predictions are visited in descending confidence order; a prediction is
    correct (True) iff it matches an unclaimed GT box of the same class with
    IoU >= iou_thr. Duplicates of an already-claimed GT are incorrect.
    Returns a boolean correctness array aligned with the input order.
    """
    n = len(pred_confs)
    correct = np.zeros(n, dtype=bool)
    claimed = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-np.asarray(pred_confs, dtype=float)):
        best_iou, best_j = 0.0, -1
        for j in range(len(gt_boxes)):
            if claimed[j] or gt_classes[j] != pred_classes[i]:
                continue
            iou = iou_xyxy(np.asarray(pred_boxes[i], float), np.asarray(gt_boxes[j], float))
            if iou >= iou_thr and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            correct[i] = True
            claimed[best_j] = True
    return correct


def reliability_bins(confs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Per-bin stats over equal-width confidence bins on (0, 1]."""
    confs = np.asarray(confs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (confs > lo) & (confs <= hi) if k > 0 else (confs >= lo) & (confs <= hi)
        count = int(mask.sum())
        bins.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": count,
                "mean_confidence": float(confs[mask].mean()) if count else 0.0,
                "accuracy": float(correct[mask].mean()) if count else 0.0,
            }
        )
    return bins


def expected_calibration_error(confs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE: sum over bins of (bin_count/N) * |accuracy - mean_confidence|."""
    n = len(confs)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in reliability_bins(confs, correct, n_bins):
        if b["count"]:
            ece += (b["count"] / n) * abs(b["accuracy"] - b["mean_confidence"])
    return float(ece)


def _logit(confs: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(confs, dtype=float), EPS, 1.0 - EPS)
    return np.log(c / (1.0 - c))


def apply_temperature(confs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale confidences through logit space with scalar temperature."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return 1.0 / (1.0 + np.exp(-_logit(confs) / temperature))


def _nll(confs: np.ndarray, correct: np.ndarray, temperature: float) -> float:
    p = np.clip(apply_temperature(confs, temperature), EPS, 1.0 - EPS)
    y = np.asarray(correct, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(
    confs: np.ndarray,
    correct: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 10.0,
    iters: int = 100,
) -> float:
    """Fit scalar temperature by golden-section search on binary NLL."""
    if len(confs) == 0:
        return 1.0
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = t_min, t_max
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _nll(confs, correct, c), _nll(confs, correct, d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _nll(confs, correct, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _nll(confs, correct, d)
    return float((a + b) / 2.0)
