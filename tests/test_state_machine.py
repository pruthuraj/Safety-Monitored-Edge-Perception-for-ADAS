"""Week 6 gating tests (see 'week 6 plan.md' Test Plan): state machine unit
tests + runtime monitor integration (GPU tests skip when weights absent)."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitor.state_machine import (
    DEGRADED,
    FAIL_SAFE_REQUEST,
    NOMINAL,
    MonitorStateMachine,
)

REPO = Path(__file__).resolve().parents[1]
Q95, Q99 = 0.2, 0.4
MID = 0.3   # breaches Q95 only
HOT = 0.5   # breaches both
OK = 0.1    # clean


def machine():
    return MonitorStateMachine(Q95, Q99)


def run(m, scores):
    for s in scores:
        t = m.step(s)
    return t


# --- unit transitions --------------------------------------------------------

def test_single_q95_spike_does_not_change_state():
    m = machine()
    run(m, [OK, MID, OK])
    assert m.state == NOMINAL


def test_two_q95_breaches_not_enough():
    m = machine()
    run(m, [MID, MID, OK])
    assert m.state == NOMINAL


def test_three_consecutive_q95_enter_degraded():
    m = machine()
    t = run(m, [MID, MID, MID])
    assert m.state == DEGRADED
    assert t.reason == "q95_breach_x3"


def test_two_consecutive_q99_enter_fail_safe_from_nominal():
    m = machine()
    t = run(m, [HOT, HOT])
    assert m.state == FAIL_SAFE_REQUEST
    assert t.reason == "q99_breach_x2"


def test_q99_breach_from_degraded_enters_fail_safe():
    m = machine()
    run(m, [MID, MID, MID])  # DEGRADED
    t = run(m, [HOT, HOT])
    assert m.state == FAIL_SAFE_REQUEST
    assert t.reason == "q99_breach_x2"


def test_ten_degraded_frames_escalate():
    m = machine()
    run(m, [MID, MID, MID])  # enter DEGRADED
    t = run(m, [MID] * 10)   # persist without recovery
    assert m.state == FAIL_SAFE_REQUEST
    assert t.reason == "degraded_persist_10"


def test_five_clean_recover_degraded_to_nominal():
    m = machine()
    run(m, [MID, MID, MID])
    run(m, [OK] * 4)
    assert m.state == DEGRADED  # 4 not enough
    t = run(m, [OK])
    assert m.state == NOMINAL
    assert t.reason == "recovered_clean_x5"


def test_five_clean_recover_fail_safe_one_level_only():
    m = machine()
    run(m, [HOT, HOT])
    t = run(m, [OK] * 5)
    assert m.state == DEGRADED  # one level, not straight to NOMINAL
    assert t.reason == "recovered_clean_x5"
    run(m, [OK] * 5)
    assert m.state == NOMINAL


def test_breach_is_strict_greater_than():
    m = machine()
    run(m, [Q95] * 5)  # equality = clean
    assert m.state == NOMINAL
    m2 = machine()
    run(m2, [HOT, Q99])  # second frame equality resets Q99 streak
    assert m2.state == NOMINAL


def test_interrupted_q95_streak_resets():
    m = machine()
    run(m, [MID, MID, OK, MID, MID])
    assert m.state == NOMINAL


def test_q95_q99_ordering_enforced():
    with pytest.raises(ValueError):
        MonitorStateMachine(0.5, 0.4)


# --- gating scenario replay (script parity) ----------------------------------

def test_gating_scenarios_replay_all_pass():
    from scripts.run_demo import gating_scenarios

    for sc in gating_scenarios(Q95, Q99):
        m = machine()
        for s in sc["scores"]:
            m.step(s)
        assert m.state == sc["expected_final"], sc["scenario"]


# --- integration (GPU + data; skipped when unavailable) ----------------------

WEIGHTS = REPO / "runs" / "detect" / "baseline" / "weights" / "best.pt"
KITTI_VAL = REPO / "data" / "processed" / "kitti_yolo" / "images" / "val"
NIGHT_SLICE = REPO / "configs" / "splits" / "bdd-night.txt"
BDD_IMAGES = REPO / "data" / "raw" / "bdd100k" / "images" / "100k" / "val"
THRESHOLDS = REPO / "results" / "monitor_thresholds.json"

runtime_ready = WEIGHTS.exists() and THRESHOLDS.exists()


@pytest.mark.skipif(not (runtime_ready and KITTI_VAL.exists()), reason="weights/thresholds/data absent")
def test_runtime_monitor_kitti_sequence_log_schema():
    from src.monitor.runtime import LOG_COLUMNS, RuntimeMonitor

    monitor = RuntimeMonitor(WEIGHTS)
    images = sorted(KITTI_VAL.glob("*.png"))[:8]
    rows = [monitor.process(p).row for p in images]
    assert len(rows) == 8  # one row per processed frame
    for r in rows:
        assert list(r) == LOG_COLUMNS
        assert r["state_after"] in (NOMINAL, DEGRADED, FAIL_SAFE_REQUEST)
        assert r["latency_ms"] > 0
    assert [r["frame_id"] for r in rows] == list(range(1, 9))


@pytest.mark.skipif(
    not (runtime_ready and NIGHT_SLICE.exists() and BDD_IMAGES.exists()),
    reason="weights/thresholds/BDD absent",
)
def test_runtime_monitor_bdd_night_degrades():
    from src.monitor.runtime import RuntimeMonitor

    monitor = RuntimeMonitor(WEIGHTS)
    images = [BDD_IMAGES / n for n in NIGHT_SLICE.read_text().split()[:15]]
    states = [monitor.process(p).row["state_after"] for p in images]
    assert any(s in (DEGRADED, FAIL_SAFE_REQUEST) for s in states)  # night must trip the monitor


@pytest.mark.skipif(not (REPO / "results" / "runtime_monitor_log.csv").exists(), reason="EXP-009 not run yet")
def test_runtime_log_one_row_per_frame_and_latency_summary():
    with (REPO / "results" / "runtime_monitor_log.csv").open(newline="") as f:
        log = list(csv.DictReader(f))
    ids = [int(r["frame_id"]) for r in log]
    assert ids == list(range(1, len(log) + 1))
    with (REPO / "results" / "monitor_latency_metrics.csv").open(newline="") as f:
        summary = list(csv.DictReader(f))[0]
    for col in ("latency_ms_p50", "latency_ms_p95", "fps_mean", "backend", "n_frames", "threshold_source"):
        assert summary[col] != ""
    assert int(summary["n_frames"]) >= 300
