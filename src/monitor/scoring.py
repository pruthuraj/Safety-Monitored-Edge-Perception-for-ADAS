"""Frame-level OOD scoring for detection outputs (SR-02, Week 4, EXP-007).

Two scores, both computed from post-NMS detection confidences (higher score
= more OOD-like):

- max-confidence baseline: 1 - max(conf); no detections -> 1.0.
- energy-style score: negative logsumexp over the logits of the top-k
  detection confidences. NOTE: this is a practical post-NMS proxy computed
  from final detection confidences — it is NOT raw-logit energy over the
  classification head and NOT backbone-feature Mahalanobis. Documented as
  such per the Week 4 plan; deeper scores remain stretch work.

Metrics: AUROC (rank-based, ties handled) and FPR@95 (false-positive rate on
ID at the threshold capturing 95% of OOD as positive).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-7


def max_conf_score(confs: np.ndarray) -> float:
    """1 - max detection confidence; empty detections -> 1.0 (max OOD)."""
    confs = np.asarray(confs, dtype=float)
    if confs.size == 0:
        return 1.0
    return float(1.0 - confs.max())


def energy_score(confs: np.ndarray, top_k: int = 10) -> float:
    """Energy-style proxy: -logsumexp(logit(conf)) over top-k detections.

    Confident detections produce large positive logits -> low (negative)
    energy. Empty detections carry no evidence of in-distribution input and
    map to the score of a single chance-level (0.5) detection: 0.0, the
    upper bound of the detection-backed range.
    """
    confs = np.asarray(confs, dtype=float)
    if confs.size == 0:
        return 0.0
    top = np.sort(confs)[-top_k:]
    c = np.clip(top, EPS, 1.0 - EPS)
    logits = np.log(c / (1.0 - c))
    m = logits.max()
    return float(-(m + np.log(np.sum(np.exp(logits - m)))))


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC for OOD detection: P(ood_score > id_score), ties count 0.5.

    Rank-based (Mann-Whitney U), OOD as the positive class.
    """
    id_scores = np.asarray(id_scores, dtype=float)
    ood_scores = np.asarray(ood_scores, dtype=float)
    if id_scores.size == 0 or ood_scores.size == 0:
        raise ValueError("both ID and OOD score arrays must be non-empty")
    all_scores = np.concatenate([id_scores, ood_scores])
    order = all_scores.argsort(kind="mergesort")
    ranks = np.empty(len(all_scores), dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # average ranks over ties
    for v in np.unique(all_scores):
        mask = all_scores == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r_ood = ranks[len(id_scores):].sum()
    n_o, n_i = len(ood_scores), len(id_scores)
    u = r_ood - n_o * (n_o + 1) / 2.0
    return float(u / (n_o * n_i))


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, tpr: float = 0.95) -> float:
    """FPR on ID scores at the score threshold achieving `tpr` on OOD.

    Threshold = (1 - tpr) quantile of OOD scores (lower interpolation, so at
    least tpr of OOD scores are >= threshold). FPR = fraction of ID scores
    >= threshold.
    """
    id_scores = np.asarray(id_scores, dtype=float)
    ood_scores = np.asarray(ood_scores, dtype=float)
    if id_scores.size == 0 or ood_scores.size == 0:
        raise ValueError("both ID and OOD score arrays must be non-empty")
    thr = np.quantile(ood_scores, 1.0 - tpr, method="lower")
    return float(np.mean(id_scores >= thr))
