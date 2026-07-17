"""Week 4 monitor tests (see 'week 4 plan.md' Test Plan): calibration, OOD scoring, BDD slices."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.bdd100k_slices import SLICE_RULES, build_slices, matches
from src.monitor.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    iou_xyxy,
    match_detections,
    reliability_bins,
)
from src.monitor.scoring import auroc, energy_score, fpr_at_tpr, max_conf_score


# --- ECE ---------------------------------------------------------------------

def test_ece_perfectly_calibrated_is_zero():
    # bin (0.6, 0.667]: all confidences 0.65, accuracy 0.65 -> ECE 0
    confs = np.full(100, 0.65)
    correct = np.zeros(100, dtype=bool)
    correct[:65] = True
    assert expected_calibration_error(confs, correct, n_bins=15) == pytest.approx(0.0)


def test_ece_known_value_two_bins():
    # bin A: 10 dets at conf 0.9, accuracy 0.5 -> gap 0.4, weight 10/20
    # bin B: 10 dets at conf 0.1, accuracy 0.1 -> gap 0.0, weight 10/20
    confs = np.array([0.9] * 10 + [0.1] * 10)
    correct = np.array([True] * 5 + [False] * 5 + [True] * 1 + [False] * 9)
    assert expected_calibration_error(confs, correct, n_bins=10) == pytest.approx(0.2)


def test_ece_empty_input_is_zero():
    assert expected_calibration_error(np.array([]), np.array([], dtype=bool)) == 0.0


def test_reliability_bins_count_and_coverage():
    confs = np.linspace(0.01, 0.99, 200)
    correct = np.ones(200, dtype=bool)
    bins = reliability_bins(confs, correct, n_bins=15)
    assert len(bins) == 15
    assert sum(b["count"] for b in bins) == 200


# --- temperature scaling -----------------------------------------------------

def test_temperature_identity_at_one():
    confs = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(apply_temperature(confs, 1.0), confs, atol=1e-9)


def test_temperature_preserves_order_and_softens():
    confs = np.array([0.2, 0.6, 0.95])
    scaled = apply_temperature(confs, 2.0)
    # monotonic: order preserved
    assert np.all(np.diff(scaled) > 0)
    # T>1 pulls toward 0.5
    assert scaled[2] < confs[2] and scaled[0] > confs[0]


def test_temperature_rejects_nonpositive():
    with pytest.raises(ValueError):
        apply_temperature(np.array([0.5]), 0.0)


def test_fit_temperature_recovers_overconfidence():
    # correctness generated at true probability = conf softened by T=2:
    # fitted T should land near 2, and definitely above 1
    rng = np.random.default_rng(42)
    confs = rng.uniform(0.05, 0.99, size=5000)
    true_p = apply_temperature(confs, 2.0)
    correct = rng.uniform(size=5000) < true_p
    t = fit_temperature(confs, correct)
    assert 1.5 < t < 2.5


def test_fit_temperature_empty_returns_identity():
    assert fit_temperature(np.array([]), np.array([], dtype=bool)) == 1.0


# --- IoU matching ------------------------------------------------------------

GT_BOX = np.array([[10.0, 10.0, 50.0, 50.0]])
GT_CLS = np.array([0])


def test_iou_identical_boxes_is_one():
    assert iou_xyxy(GT_BOX[0], GT_BOX[0]) == pytest.approx(1.0)


def test_matching_true_positive():
    ok = match_detections(GT_BOX, np.array([0]), np.array([0.9]), GT_BOX, GT_CLS)
    assert ok.tolist() == [True]


def test_matching_wrong_class_is_fp():
    ok = match_detections(GT_BOX, np.array([1]), np.array([0.9]), GT_BOX, GT_CLS)
    assert ok.tolist() == [False]


def test_matching_low_iou_is_fp():
    far = np.array([[200.0, 200.0, 240.0, 240.0]])
    ok = match_detections(far, np.array([0]), np.array([0.9]), GT_BOX, GT_CLS)
    assert ok.tolist() == [False]


def test_matching_duplicate_claims_gt_once_highest_conf_wins():
    preds = np.vstack([GT_BOX[0], GT_BOX[0] + 1.0])
    ok = match_detections(preds, np.array([0, 0]), np.array([0.6, 0.9]), GT_BOX, GT_CLS)
    # higher-conf second prediction claims the GT; first becomes duplicate FP
    assert ok.tolist() == [False, True]


def test_matching_no_predictions():
    ok = match_detections(
        np.zeros((0, 4)), np.array([], dtype=int), np.array([]), GT_BOX, GT_CLS
    )
    assert ok.shape == (0,)


# --- OOD scorers -------------------------------------------------------------

def test_max_conf_score_normal_and_empty():
    assert max_conf_score(np.array([0.3, 0.8])) == pytest.approx(0.2)
    assert max_conf_score(np.array([])) == 1.0


def test_energy_score_confident_lower_than_weak():
    confident = energy_score(np.array([0.95, 0.9, 0.85]))
    weak = energy_score(np.array([0.10, 0.08]))
    assert confident < weak


def test_energy_score_empty_is_upper_bound_of_detection_range():
    # no detections -> 0.0, above any frame holding a confident detection
    assert energy_score(np.array([])) == 0.0
    assert energy_score(np.array([0.9])) < 0.0


def test_energy_score_uses_top_k_only():
    strong = np.array([0.9] * 5)
    padded = np.concatenate([strong, np.full(50, 0.06)])
    assert energy_score(padded, top_k=5) == pytest.approx(energy_score(strong, top_k=5))


# --- AUROC / FPR@95 ----------------------------------------------------------

def test_auroc_perfect_separation():
    assert auroc(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0


def test_auroc_chance_for_identical_distributions():
    s = np.array([0.5] * 10)
    assert auroc(s, s) == pytest.approx(0.5)


def test_fpr_at_95_perfect_separation_is_zero():
    id_s = np.linspace(0.0, 0.4, 100)
    ood_s = np.linspace(0.6, 1.0, 100)
    assert fpr_at_tpr(id_s, ood_s, 0.95) == 0.0


def test_fpr_at_95_total_overlap_is_high():
    s = np.linspace(0.0, 1.0, 100)
    assert fpr_at_tpr(s, s, 0.95) > 0.9


# --- BDD slice builder -------------------------------------------------------

def frame(name, weather="clear", timeofday="daytime"):
    return {"name": name, "attributes": {"weather": weather, "timeofday": timeofday}}


def test_matches_filters_by_attributes():
    assert matches(frame("a.jpg", "rainy", "night"), {"weather": "rainy"})
    assert not matches(frame("a.jpg", "clear", "night"), {"weather": "rainy"})
    assert matches(frame("a.jpg", "clear", "daytime"), SLICE_RULES["bdd-clear-day"])
    assert not matches(frame("a.jpg", "rainy", "daytime"), SLICE_RULES["bdd-clear-day"])
    assert not matches({"name": "no-attrs.jpg"}, {"weather": "rainy"})


def test_build_slices_filters_and_caps():
    frames = (
        [frame(f"day{i}.jpg") for i in range(30)]
        + [frame(f"night{i}.jpg", "clear", "night") for i in range(30)]
        + [frame(f"rain{i}.jpg", "rainy", "night") for i in range(5)]
    )
    slices, available = build_slices(frames, seed=42, target=10)
    assert len(slices["bdd-clear-day"]) == 10 and available["bdd-clear-day"] == 30
    assert len(slices["bdd-night"]) == 10 and available["bdd-night"] == 35  # rain-night counts
    assert slices["bdd-rain"] == sorted(f"rain{i}.jpg" for i in range(5))  # under target: all
    assert slices["bdd-fog"] == [] and available["bdd-fog"] == 0


def test_build_slices_deterministic():
    frames = [frame(f"d{i}.jpg") for i in range(100)]
    s1, _ = build_slices(frames, seed=42, target=20)
    s2, _ = build_slices(frames, seed=42, target=20)
    assert s1 == s2
