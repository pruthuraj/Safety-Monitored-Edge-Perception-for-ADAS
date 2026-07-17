"""Monitor score thresholds (Week 5, EXP-008; feeds SR-03/SR-04 gating).

Q95/Q99 thresholds are computed from KITTI validation monitor scores ONLY
(calibration-report subset) — never from BDD slices and never from
kitti-test. Threshold hygiene per docs/dataset_splits.md rule 2: everything
except kitti-val is report-only.

A frame is treated as suspect when its score EXCEEDS a threshold (higher
score = more OOD-like); Q95 marks the DEGRADED candidate boundary, Q99 the
FAIL_SAFE_REQUEST candidate boundary (state machine lands in Week 6).
"""

from __future__ import annotations

import numpy as np

QUANTILES = {"q95": 0.95, "q99": 0.99}


def quantile_thresholds(scores: np.ndarray, quantiles: dict[str, float] = QUANTILES) -> dict:
    """Named quantile thresholds of an ID score sample (linear interpolation)."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("cannot compute thresholds from empty score array")
    return {name: float(np.quantile(scores, q)) for name, q in quantiles.items()}


def coverage_at(scores: np.ndarray, threshold: float) -> float:
    """Fraction of frames accepted (score <= threshold)."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("cannot compute coverage from empty score array")
    return float(np.mean(scores <= threshold))
