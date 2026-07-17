"""Week 7 fault-injection tests (see 'week 7 plan.md' Test Plan)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.corruptions import (
    CORRUPTIONS,
    SEVERITIES,
    apply_corruption,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def sample_image():
    rng = np.random.default_rng(7)
    return rng.integers(40, 200, (128, 256, 3), dtype=np.uint8)


# --- unit: corruption functions ----------------------------------------------

@pytest.mark.parametrize("corruption", CORRUPTIONS)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_corruptions_preserve_shape_and_dtype(corruption, severity):
    img = sample_image()
    out = apply_corruption(img, corruption, severity, frame_id="000123")
    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_severity_monotonically_increases_magnitude(corruption):
    img = sample_image()
    mags = [
        float(np.mean(np.abs(
            apply_corruption(img, corruption, s, frame_id="000123").astype(np.int16)
            - img.astype(np.int16)
        )))
        for s in SEVERITIES
    ]
    assert mags[0] < mags[1] < mags[2], f"{corruption}: {mags}"


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_same_frame_id_is_bit_identical(corruption):
    img = sample_image()
    a = apply_corruption(img, corruption, "medium", frame_id="000042")
    b = apply_corruption(img, corruption, "medium", frame_id="000042")
    np.testing.assert_array_equal(a, b)


def test_stochastic_corruptions_differ_across_frame_ids():
    img = sample_image()
    a = apply_corruption(img, "gaussian_noise", "medium", frame_id="000001")
    b = apply_corruption(img, "gaussian_noise", "medium", frame_id="000002")
    assert not np.array_equal(a, b)


def test_invalid_corruption_and_severity_raise():
    img = sample_image()
    with pytest.raises(ValueError, match="unknown corruption"):
        apply_corruption(img, "sharknado", "low")
    with pytest.raises(ValueError, match="unknown severity"):
        apply_corruption(img, "fog", "apocalyptic")
    with pytest.raises(ValueError, match="uint8"):
        apply_corruption(img.astype(np.float32), "fog", "low")


# --- integration: smoke run ---------------------------------------------------

WEIGHTS = REPO / "runs" / "detect" / "baseline" / "weights" / "best.pt"
TEST_IMAGES = REPO / "data" / "processed" / "kitti_yolo" / "images" / "test"
THRESHOLDS = REPO / "results" / "monitor_thresholds.json"
smoke_ready = WEIGHTS.exists() and TEST_IMAGES.exists() and THRESHOLDS.exists()


@pytest.mark.skipif(not smoke_ready, reason="weights/test data/thresholds absent")
def test_fault_injection_smoke_outputs():
    cmd = [
        sys.executable, "scripts/run_fault_injection.py",
        "--n", "3", "--corruptions", "fog", "motion_blur", "--severities", "low", "high",
        "--weights", str(WEIGHTS), "--out-prefix", "fault_smoke",
    ]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]

    with (RESULTS / "fault_smoke_metrics.csv").open(newline="") as f:
        metrics = list(csv.DictReader(f))
    assert len(metrics) == 1 + 2 * 2  # clean + 2 corruptions x 2 severities
    required = {
        "corruption", "severity", "mAP50", "mAP50_95", "precision", "recall", "mean_score",
        "max_score", "nominal_fraction", "degraded_fraction", "fail_safe_fraction",
        "first_transition_frame", "latency_ms_p50", "latency_ms_p95", "within_budget_p95",
    }
    assert required <= set(metrics[0])
    assert all(m["mAP50"] != "" for m in metrics)  # non-empty metrics
    assert all(m["latency_ms_p95"] != "" for m in metrics)  # latency computed vs budget

    with (RESULTS / "fault_smoke_monitor_log.csv").open(newline="") as f:
        log = list(csv.DictReader(f))
    assert len(log) == 5 * 3  # one row per processed corrupted/clean frame

    summary = json.loads((RESULTS / "fault_smoke_summary.json").read_text())
    for key in ("subset_sha256", "thresholds", "backend", "headline"):
        assert key in summary
    # cleanup smoke artifacts so they are not mistaken for EXP-010 evidence
    for name in (
        "fault_smoke_metrics.csv",
        "fault_smoke_monitor_log.csv",
        "fault_smoke_summary.json",
        "fault_smoke_curves.png",
    ):
        (RESULTS / name).unlink(missing_ok=True)
