"""run_demo: runtime gating evidence + annotated demo video (Week 6, EXP-009).

Default run produces, in order:
  1. results/gating_tests.csv            deterministic state-machine scenario replay (SR-03/SR-04)
  2. results/runtime_monitor_log.csv     per-frame machine-readable log, 300 KITTI val frames (SR-05)
     results/monitor_log_check.csv       log completeness summary (SR-05)
     results/monitor_latency_metrics.csv perception+monitor+logging latency p50/p95 (SR-06)
  3. demo/monitor_overlay.mp4            annotated KITTI + BDD night/rain/fog sequence
                                         (illustrative evidence only — never a metric source)

Thresholds are loaded frozen from results/monitor_thresholds.json (EXP-008);
never recomputed here. Backend: TensorRT FP16 engine if present, else PyTorch.

Usage:
    python scripts/run_demo.py                # all three stages
    python scripts/run_demo.py --gating-only
    python scripts/run_demo.py --latency-only
    python scripts/run_demo.py --demo-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset.kitti_classes import CLASS_NAMES
from src.monitor.runtime import LOG_COLUMNS, RuntimeMonitor
from src.monitor.state_machine import (
    DEGRADED,
    FAIL_SAFE_REQUEST,
    MonitorStateMachine,
    NOMINAL,
)

RESULTS = REPO / "results"
DEMO_DIR = REPO / "demo"
KITTI_IMAGES = REPO / "data" / "processed" / "kitti_yolo" / "images" / "val"
BDD_IMAGES = REPO / "data" / "raw" / "bdd100k" / "images" / "100k" / "val"
SPLITS = REPO / "configs" / "splits"

LATENCY_N_FRAMES = 300
DEMO_SEG_FRAMES = 40  # per segment; fog capped by availability (13)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --- stage 1: gating scenario replay ----------------------------------------

def gating_scenarios(q95: float, q99: float) -> list[dict]:
    mid = (q95 + q99) / 2.0  # breaches Q95 only
    hot = q99 + 0.1          # breaches both
    ok = q95 - 0.05          # clean
    return [
        {
            "scenario": "single_q95_spike_no_transition",
            "scores": [ok, mid, ok, ok],
            "expected_final": NOMINAL,
        },
        {
            "scenario": "three_q95_breaches_enter_degraded",
            "scores": [ok, mid, mid, mid],
            "expected_final": DEGRADED,
        },
        {
            "scenario": "two_q99_breaches_enter_fail_safe",
            "scores": [ok, hot, hot],
            "expected_final": FAIL_SAFE_REQUEST,
        },
        {
            "scenario": "ten_degraded_frames_escalate",
            "scores": [mid, mid, mid] + [mid] * 10,
            "expected_final": FAIL_SAFE_REQUEST,
        },
        {
            "scenario": "five_clean_recover_degraded_to_nominal",
            "scores": [mid, mid, mid] + [ok] * 5,
            "expected_final": NOMINAL,
        },
        {
            "scenario": "five_clean_recover_fail_safe_to_degraded",
            "scores": [hot, hot] + [ok] * 5,
            "expected_final": DEGRADED,
        },
        {
            "scenario": "threshold_equality_is_clean",
            "scores": [q95, q95, q95, q95],
            "expected_final": NOMINAL,
        },
    ]


def run_gating_tests(q95: float, q99: float) -> list[dict]:
    rows = []
    for sc in gating_scenarios(q95, q99):
        machine = MonitorStateMachine(q95, q99)
        for s in sc["scores"]:
            machine.step(s)
        rows.append(
            {
                "date": date.today().isoformat(),
                "experiment": "EXP-009",
                "scenario": sc["scenario"],
                "n_frames": len(sc["scores"]),
                "q95": q95,
                "q99": q99,
                "expected_final": sc["expected_final"],
                "actual_final": machine.state,
                "pass": machine.state == sc["expected_final"],
            }
        )
    write_rows(RESULTS / "gating_tests.csv", rows)
    n_pass = sum(r["pass"] for r in rows)
    print(f"gating scenarios: {n_pass}/{len(rows)} pass")
    if n_pass != len(rows):
        raise SystemExit("gating scenario replay failed — see results/gating_tests.csv")
    return rows


# --- stage 2: latency + logging over KITTI val -------------------------------

def latency_frames() -> list[Path]:
    from scripts.evaluate_monitor import calib_subsets

    _, report_ids = calib_subsets()
    return [KITTI_IMAGES / f"{i}.png" for i in sorted(report_ids)[:LATENCY_N_FRAMES]]


def run_latency(weights: Path | None) -> dict:
    monitor = RuntimeMonitor(weights)
    frames = latency_frames()
    print(f"latency run: {len(frames)} KITTI val frames, backend={monitor.backend}")
    # warmup outside measurement
    for p in frames[:10]:
        monitor.model.predict(p, imgsz=monitor.imgsz, conf=monitor.conf_min, verbose=False)

    log_rows = [monitor.process(p).row for p in frames]
    # canonical log = default backend; explicit overrides get a suffixed file
    log_name = (
        "runtime_monitor_log.csv"
        if weights is None
        else f"runtime_monitor_log_{monitor.backend}.csv"
    )
    write_rows(RESULTS / log_name, log_rows)

    # SR-05 completeness check
    missing = {c: sum(1 for r in log_rows if r[c] == "" and c != "max_confidence") for c in LOG_COLUMNS}
    empty_maxconf = sum(1 for r in log_rows if r["max_confidence"] == "")
    check_rows = [
        {
            "check": "one_row_per_frame",
            "expected": len(frames),
            "actual": len(log_rows),
            "pass": len(log_rows) == len(frames),
        },
        *[
            {"check": f"no_missing_{c}", "expected": 0, "actual": n, "pass": n == 0}
            for c, n in missing.items()
        ],
        {
            "check": "max_confidence_empty_only_for_zero_detections",
            "expected": sum(1 for r in log_rows if r["n_detections"] == 0),
            "actual": empty_maxconf,
            "pass": empty_maxconf == sum(1 for r in log_rows if r["n_detections"] == 0),
        },
    ]
    write_rows(RESULTS / "monitor_log_check.csv", check_rows)
    if not all(r["pass"] for r in check_rows):
        raise SystemExit("log completeness check failed — see results/monitor_log_check.csv")

    lat = np.array([r["latency_ms"] for r in log_rows])
    summary = {
        "date": date.today().isoformat(),
        "experiment": "EXP-009",
        "backend": monitor.backend,
        "weights": str(monitor.weights.relative_to(REPO)),
        "n_frames": len(log_rows),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
        "fps_mean": round(1000.0 / float(lat.mean()), 1),
        "budget_ms": 40.0,
        "within_budget_p95": bool(np.percentile(lat, 95) < 40.0),
        "scope": "predict + max_conf_score + state machine + log row (SR-06)",
        "threshold_source": "results/monitor_thresholds.json (EXP-008, kitti-val only)",
        "command": " ".join(sys.argv),
    }
    # one row per backend, appended (rerunning a backend adds a dated row)
    lat_path = RESULTS / "monitor_latency_metrics.csv"
    exists = lat_path.exists()
    with lat_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if not exists:
            w.writeheader()
        w.writerow(summary)
    print(json.dumps(summary, indent=2))
    return summary


# --- stage 3: demo video ------------------------------------------------------

def demo_sequence() -> list[tuple[str, Path]]:
    """Deterministic (segment, image) list: KITTI then night/rain/fog."""
    from scripts.evaluate_monitor import calib_subsets

    _, report_ids = calib_subsets()
    seq = [("kitti-val", KITTI_IMAGES / f"{i}.png") for i in sorted(report_ids)[:DEMO_SEG_FRAMES]]
    for name in ("bdd-night", "bdd-rain", "bdd-fog"):
        images = (SPLITS / f"{name}.txt").read_text().split()[:DEMO_SEG_FRAMES]
        seq.extend((name, BDD_IMAGES / n) for n in images)
    return seq


STATE_COLORS = {NOMINAL: (80, 200, 80), DEGRADED: (0, 165, 255), FAIL_SAFE_REQUEST: (60, 60, 255)}
CANVAS_W, CANVAS_H = 1280, 720


def render_frame(cv2, img, fr, segment: str):
    h, w = img.shape[:2]
    scale = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    ox, oy = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized

    for box, cls, conf in zip(fr.boxes, fr.classes, fr.confs):
        x1, y1, x2, y2 = (box * scale).astype(int) + np.array([ox, oy, ox, oy])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 200, 80), 2)
        cv2.putText(
            canvas,
            f"{CLASS_NAMES[cls]} {conf:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 200, 80),
            1,
        )

    row = fr.row
    state = row["state_after"]
    color = STATE_COLORS[state]
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 64), (25, 25, 25), -1)
    cv2.rectangle(canvas, (8, 8), (330, 56), color, -1)
    cv2.putText(canvas, state, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    info = (
        f"{segment} | score {row['monitor_score']:.3f} "
        f"(Q95 {row['q95_threshold']:.3f} / Q99 {row['q99_threshold']:.3f}) | "
        f"{row['latency_ms']:.1f} ms | {row['backend']}"
    )
    cv2.putText(canvas, info, (344, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    if row["transition_reason"] != "no_change":
        cv2.putText(
            canvas, f"transition: {row['transition_reason']}", (344, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
        )
    return canvas


def run_demo_video(weights: Path | None) -> Path:
    import cv2

    monitor = RuntimeMonitor(weights)  # fresh state machine for the demo
    seq = demo_sequence()
    print(f"demo: {len(seq)} frames")
    DEMO_DIR.mkdir(exist_ok=True)
    out_path = DEMO_DIR / "monitor_overlay.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (CANVAS_W, CANVAS_H))
    for segment, path in seq:
        fr = monitor.process(path)
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"unreadable demo frame {path}")
        writer.write(render_frame(cv2, img, fr, segment))
    writer.release()
    print(f"wrote {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gating-only", action="store_true")
    ap.add_argument("--latency-only", action="store_true")
    ap.add_argument("--demo-only", action="store_true")
    ap.add_argument("--weights", type=Path, default=None, help="override backend weights")
    args = ap.parse_args()
    run_all = not (args.gating_only or args.latency_only or args.demo_only)

    from src.monitor.runtime import load_frozen_thresholds

    q95, q99, method = load_frozen_thresholds()
    print(f"frozen thresholds ({method}): Q95={q95} Q99={q99}")

    if run_all or args.gating_only:
        run_gating_tests(q95, q99)
    if run_all or args.latency_only:
        run_latency(args.weights)
    if run_all or args.demo_only:
        run_demo_video(args.weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
