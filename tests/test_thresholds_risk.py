"""Week 5 tests (see 'week 5 paln.md' Test Plan): thresholds, detection metrics,
risk-coverage behavior, BDD manifest integrity, output schemas."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitor.detection_metrics import (
    mean_ap,
    mean_ap_50_95,
    precision_recall,
)
from src.monitor.thresholds import coverage_at, quantile_thresholds

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "configs" / "splits"
RESULTS = REPO / "results"


# --- quantile thresholds -----------------------------------------------------

def test_quantile_thresholds_known_values():
    scores = np.arange(1, 101, dtype=float)  # 1..100
    t = quantile_thresholds(scores)
    assert set(t) == {"q95", "q99"}
    assert t["q95"] == pytest.approx(np.quantile(scores, 0.95))
    assert t["q99"] == pytest.approx(np.quantile(scores, 0.99))
    assert t["q95"] < t["q99"]


def test_quantile_thresholds_empty_raises():
    with pytest.raises(ValueError):
        quantile_thresholds(np.array([]))


def test_coverage_monotone_in_threshold():
    scores = np.random.default_rng(42).uniform(0, 1, 500)
    thresholds = np.linspace(0, 1, 20)
    covs = [coverage_at(scores, t) for t in thresholds]
    assert all(a <= b for a, b in zip(covs, covs[1:]))  # tighter threshold -> less coverage
    assert coverage_at(scores, 1.0) == 1.0
    assert coverage_at(scores, -0.1) == 0.0


# --- detection metrics -------------------------------------------------------

def perfect_frame():
    boxes = np.array([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 150.0, 160.0]])
    classes = np.array([0, 1])
    return {
        "boxes": boxes,
        "classes": classes,
        "confs": np.array([0.9, 0.8]),
        "gt_boxes": boxes.copy(),
        "gt_classes": classes.copy(),
    }


def empty_pred_frame():
    return {
        "boxes": np.zeros((0, 4)),
        "classes": np.array([], dtype=int),
        "confs": np.array([]),
        "gt_boxes": np.array([[10.0, 10.0, 50.0, 50.0]]),
        "gt_classes": np.array([0]),
    }


def test_perfect_predictions_ap_one():
    frames = [perfect_frame(), perfect_frame()]
    assert mean_ap(frames, 0.50) == pytest.approx(1.0)
    assert mean_ap_50_95(frames) == pytest.approx(1.0)
    p, r = precision_recall(frames)
    assert p == 1.0 and r == 1.0


def test_no_predictions_ap_zero_no_crash():
    frames = [empty_pred_frame()]
    assert mean_ap(frames, 0.50) == 0.0
    p, r = precision_recall(frames)
    assert p == 0.0 and r == 0.0


def test_false_positive_lowers_precision_not_recall():
    fr = perfect_frame()
    fr["boxes"] = np.vstack([fr["boxes"], [[300.0, 300.0, 340.0, 340.0]]])
    fr["classes"] = np.append(fr["classes"], 0)
    fr["confs"] = np.append(fr["confs"], 0.7)
    p, r = precision_recall([fr])
    assert r == 1.0 and p == pytest.approx(2 / 3)


def test_ap_excludes_classes_without_gt():
    fr = perfect_frame()
    fr["gt_boxes"] = fr["gt_boxes"][:1]
    fr["gt_classes"] = fr["gt_classes"][:1]  # class 1 has no GT -> excluded from mean
    aps_mean = mean_ap([fr], 0.50)
    assert aps_mean == pytest.approx(1.0)  # class 0 perfect; class 1 not averaged


# --- BDD manifest integrity --------------------------------------------------

BDD_MANIFEST = SPLITS / "bdd_manifest.json"


@pytest.mark.skipif(not BDD_MANIFEST.exists(), reason="BDD slices not generated yet")
def test_bdd_manifest_has_all_slices_and_hashes():
    import hashlib

    m = json.loads(BDD_MANIFEST.read_text())
    expected = {"bdd-clear-day", "bdd-night", "bdd-rain", "bdd-fog"}
    assert set(m["counts"]) == expected
    assert set(m["sha256"]) == expected
    assert m["seed"] == 42
    for name in expected:
        f = SPLITS / f"{name}.txt"
        assert f.exists()
        assert hashlib.sha256(f.read_bytes()).hexdigest() == m["sha256"][name]
        assert m["counts"][name] == len(f.read_text().split())


# --- output schemas ----------------------------------------------------------

@pytest.mark.skipif(
    not (RESULTS / "monitor_thresholds.json").exists(), reason="thresholds not generated yet"
)
def test_monitor_thresholds_schema():
    t = json.loads((RESULTS / "monitor_thresholds.json").read_text())
    for key in ("thresholds", "primary_method", "primary_justification", "seed", "policy"):
        assert key in t
    for m in ("max_conf_score", "energy_score"):
        assert set(t["thresholds"][m]) == {"q95", "q99"}
        assert t["thresholds"][m]["q95"] <= t["thresholds"][m]["q99"]
    assert t["primary_method"] in ("max_conf_score", "energy_score")
    assert "kitti-val" in t["source_scores"]  # KITTI-only provenance recorded


@pytest.mark.skipif(
    not (RESULTS / "risk_coverage.csv").exists(), reason="risk-coverage not generated yet"
)
def test_risk_coverage_schema_and_monotonic_coverage():
    import csv

    with (RESULTS / "risk_coverage.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {
        "slice", "method", "threshold_label", "threshold", "n_frames", "n_accepted",
        "accepted_fraction", "rejected_fraction", "accepted_mAP50", "accepted_mAP50_95",
        "accepted_precision", "accepted_recall",
    }
    assert required <= set(rows[0])
    for m in ("max_conf_score", "energy_score"):
        kv = [
            (float(r["threshold"]), float(r["accepted_fraction"]))
            for r in rows
            if r["slice"] == "kitti-val-report" and r["method"] == m
        ]
        kv.sort()
        covs = [c for _, c in kv]
        assert all(a <= b + 1e-9 for a, b in zip(covs, covs[1:]))  # coverage monotone in threshold
    # BDD rows carry coverage only — detection metrics must be blank
    bdd = [r for r in rows if r["slice"].startswith("bdd-")]
    assert bdd and all(r["accepted_mAP50"] == "" for r in bdd)
