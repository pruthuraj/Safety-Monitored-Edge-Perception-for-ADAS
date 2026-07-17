"""build_final_demo: assemble the 60-90 s portfolio demo (Week 10, EXP-013).

Concatenates static title / metric / limitation / closing cards around the
existing annotated overlay clip (demo/monitor_overlay.mp4). All card numbers
are read from results/report_summary.json — none are hand-typed. Silent by
default; narration text lives in demo/demo_script.md. Illustrative artifact,
never a metric source.

Layout (24 fps output, 1280x720): title 5 s -> overlay clip (held) ~44 s ->
metric card 8 s -> limitation card 10 s -> closing 5 s ≈ 72 s.

Usage:
    python scripts/build_final_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OVERLAY = REPO / "demo" / "monitor_overlay.mp4"
SUMMARY = REPO / "results" / "report_summary.json"
OUT = REPO / "demo" / "final_demo.mp4"

W, H, FPS = 1280, 720, 24
BG = (22, 22, 26)
FG = (235, 235, 235)
MUTED = (150, 150, 155)
ACCENT = (255, 190, 70)   # BGR amber
RED = (70, 70, 235)
GREEN = (90, 200, 120)


def _put(img, text, org, scale, color, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def card(lines: list[tuple], seconds: float, accent_bar=ACCENT) -> list[np.ndarray]:
    """lines: list of (text, y, scale, color, thickness)."""
    frame = np.full((H, W, 3), BG, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (14, H), accent_bar, -1)
    for text, y, scale, color, thick in lines:
        _put(frame, text, (70, y), scale, color, thick)
    return [frame.copy() for _ in range(int(seconds * FPS))]


def overlay_frames(hold: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(OVERLAY))
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        out.extend(img for _ in range(hold))
    cap.release()
    if not out:
        raise SystemExit(f"no frames read from {OVERLAY}")
    return out


def build() -> Path:
    s = json.loads(SUMMARY.read_text())
    base = s["baseline"]
    lat = s["latency_full_loop"]["tensorrt_fp16"]
    fog = s["fault_injection"]["fog_high"]
    low = s["fault_injection"]["low_light_high"]
    night_auroc = s["ood_max_conf"]["bdd-night"]["auroc"]

    frames: list[np.ndarray] = []

    frames += card([
        ("Safety-Monitored Edge Perception", 250, 1.4, FG, 3),
        ("for ADAS", 320, 1.4, FG, 3),
        ("YOLOv8n + TensorRT FP16  |  calibrated OOD monitor  |  fail-safe gating", 400, 0.7, MUTED, 1),
        ("STPA/HARA -> SR-01..06 -> GSN safety case", 440, 0.7, MUTED, 1),
    ], 5)

    frames += overlay_frames(hold=8)

    frames += card([
        ("Key results  (within the defined KITTI-like ODD)", 130, 0.95, ACCENT, 2),
        (f"Detector      mAP50 {base['trt_fp16_mAP50']:.3f}  (TensorRT FP16)", 240, 0.85, FG, 2),
        (f"Latency       p95 {lat['p95']:.1f} ms full loop  vs  40 ms budget", 300, 0.85, FG, 2),
        (f"OOD (night)   AUROC {night_auroc:.3f}", 360, 0.85, FG, 2),
        (f"Fog (severe)  {int(fog['nonnominal_fraction']*100)}% of frames flagged as mAP50 -> {fog['mAP50']:.3f}", 420, 0.85, GREEN, 2),
        ("Every number traces to results/report_summary.json", 500, 0.6, MUTED, 1),
    ], 8)

    frames += card([
        ("Documented residual risk  (SOTIF)", 130, 0.95, RED, 2),
        (f"Severe low-light removes {int((base['pytorch_mAP50']-low['mAP50'])/base['pytorch_mAP50']*100)}% of mAP50", 250, 0.85, FG, 2),
        (f"(mAP50 {low['mAP50']:.3f})  yet the monitor stays {low['nominal_fraction']*100:.1f}% NOMINAL.", 310, 0.85, FG, 2),
        ("Frame-level max-confidence cannot see silent recall erosion.", 390, 0.75, MUTED, 1),
        ("Carried as a known-unsafe SOTIF residual, not hidden.", 440, 0.75, MUTED, 1),
        ("Future work: detection-count plausibility, temporal checks, Mahalanobis.", 500, 0.65, MUTED, 1),
    ], 10)

    frames += card([
        ("Not certified, not ISO-compliant, not proven safe.", 300, 0.85, FG, 2),
        ("An evidence-backed safety argument, bounded to its ODD.", 360, 0.75, MUTED, 1),
        ("github.com/pruthuraj/Safety-Monitored-Edge-Perception-for-ADAS", 440, 0.6, ACCENT, 1),
    ], 5)

    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()

    duration = len(frames) / FPS
    if not 60 <= duration <= 90:
        raise SystemExit(f"demo duration {duration:.1f}s outside 60-90s window")
    return OUT, duration, len(frames)


def main() -> int:
    out, duration, n = build()
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB, {n} frames, {duration:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
