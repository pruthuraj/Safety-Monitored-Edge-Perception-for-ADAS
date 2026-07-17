"""Detection-quality metrics over a frame subset (Week 5, risk-coverage).

Standalone AP implementation (VOC-style all-point interpolation) so that
accepted-frame subsets from the risk-coverage sweep can be scored without
re-running the ultralytics validator. Matching reuses the greedy one-to-one
same-class matcher from calibration (conf-descending, IoU >= thr).

Frame dict schema: {boxes, classes, confs, gt_boxes, gt_classes} — pixel
xyxy numpy arrays, aligned classes/confs.

Classes with zero ground-truth instances in the subset are excluded from
the mean (standard practice); no-detection frames contribute FN only.
"""

from __future__ import annotations

import numpy as np

from src.monitor.calibration import match_detections

IOU_RANGE = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


def _tp_flags(frames: list[dict], iou_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Concatenate per-frame greedy matches -> (confs, classes, tp), gt counts per class."""
    confs, classes, tp = [], [], []
    n_gt: dict[int, int] = {}
    for fr in frames:
        ok = match_detections(
            fr["boxes"], fr["classes"], fr["confs"], fr["gt_boxes"], fr["gt_classes"], iou_thr
        )
        confs.append(np.asarray(fr["confs"], dtype=float))
        classes.append(np.asarray(fr["classes"], dtype=int))
        tp.append(ok)
        for c in np.asarray(fr["gt_classes"], dtype=int):
            n_gt[int(c)] = n_gt.get(int(c), 0) + 1
    if confs:
        return np.concatenate(confs), np.concatenate(classes), np.concatenate(tp), n_gt
    return np.array([]), np.array([], dtype=int), np.array([], dtype=bool), n_gt


def average_precision(frames: list[dict], iou_thr: float = 0.50) -> dict[int, float]:
    """Per-class AP at one IoU threshold (all-point interpolation)."""
    confs, classes, tp, n_gt = _tp_flags(frames, iou_thr)
    aps: dict[int, float] = {}
    for c, total in n_gt.items():
        mask = classes == c
        if not mask.any():
            aps[c] = 0.0
            continue
        order = np.argsort(-confs[mask])
        tp_c = tp[mask][order]
        cum_tp = np.cumsum(tp_c)
        cum_fp = np.cumsum(~tp_c)
        recall = cum_tp / total
        precision = cum_tp / (cum_tp + cum_fp)
        # monotone precision envelope, then integrate over recall deltas
        prec_env = np.maximum.accumulate(precision[::-1])[::-1]
        r_prev = np.concatenate([[0.0], recall[:-1]])
        aps[c] = float(np.sum((recall - r_prev) * prec_env))
    return aps


def mean_ap(frames: list[dict], iou_thr: float = 0.50) -> float:
    aps = average_precision(frames, iou_thr)
    return float(np.mean(list(aps.values()))) if aps else 0.0


def mean_ap_50_95(frames: list[dict]) -> float:
    return float(np.mean([mean_ap(frames, t) for t in IOU_RANGE]))


def precision_recall(frames: list[dict], conf_thr: float = 0.25, iou_thr: float = 0.50) -> tuple[float, float]:
    """Micro-averaged precision/recall at a fixed confidence threshold."""
    filtered = []
    for fr in frames:
        keep = np.asarray(fr["confs"], dtype=float) >= conf_thr
        filtered.append(
            {
                "boxes": np.asarray(fr["boxes"], dtype=float)[keep],
                "classes": np.asarray(fr["classes"], dtype=int)[keep],
                "confs": np.asarray(fr["confs"], dtype=float)[keep],
                "gt_boxes": fr["gt_boxes"],
                "gt_classes": fr["gt_classes"],
            }
        )
    _, _, tp, n_gt = _tp_flags(filtered, iou_thr)
    total_gt = sum(n_gt.values())
    n_tp = int(tp.sum())
    precision = n_tp / len(tp) if len(tp) else 0.0
    recall = n_tp / total_gt if total_gt else 0.0
    return float(precision), float(recall)
