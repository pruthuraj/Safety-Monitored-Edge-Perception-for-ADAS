"""run_fault_injection: monitor response under synthetic degradation (Week 7, EXP-010).

Runs a deterministic 300-frame subset of kitti-test (seed 42) through the
runtime monitor: once clean (reference) and once per corruption x severity
(5 x 3). Corruptions are applied in memory — no corrupted dataset is
written. KITTI test is REPORT-ONLY here: thresholds and monitor selection
were frozen in EXP-008 from kitti-val; nothing is tuned in this script.

Outputs:
  results/fault_injection_metrics.csv      per run: detection + monitor + latency metrics
  results/fault_injection_monitor_log.csv  per frame: score/state/transition
  results/fault_injection_summary.json     run config, subset hash, headline findings
  results/fault_injection_curves.png       severity vs mAP / rejection / fail-safe

Usage:
    python scripts/run_fault_injection.py            # full 300-frame run
    python scripts/run_fault_injection.py --n 3 --corruptions fog motion_blur  # smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset.corruptions import CORRUPTIONS, SEVERITIES, apply_corruption
from src.monitor.detection_metrics import mean_ap, mean_ap_50_95, precision_recall
from src.monitor.runtime import RuntimeMonitor
from src.monitor.state_machine import DEGRADED, FAIL_SAFE_REQUEST, NOMINAL

RESULTS = REPO / "results"
TEST_SPLIT = REPO / "configs" / "splits" / "test.txt"
TEST_IMAGES = REPO / "data" / "processed" / "kitti_yolo" / "images" / "test"
TEST_LABELS = REPO / "data" / "processed" / "kitti_yolo" / "labels" / "test"

SEED = 42
N_FRAMES = 300


def subset_ids(n: int) -> list[str]:
    ids = sorted(TEST_SPLIT.read_text().split())
    rng = random.Random(SEED)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[:n])


def load_gt(image_id: str, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    label = TEST_LABELS / f"{image_id}.txt"
    boxes, classes = [], []
    if label.exists():
        for line in label.read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            boxes.append(
                [(cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h]
            )
            classes.append(c)
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(classes, dtype=int)


def run_condition(
    ids: list[str], corruption: str | None, severity: str | None, weights: Path | None
) -> tuple[dict, list[dict]]:
    """One monitor pass over the subset under a single corruption condition."""
    import cv2

    monitor = RuntimeMonitor(weights)  # fresh state machine per condition
    tag = f"{corruption}:{severity}" if corruption else "clean"
    frames, log_rows = [], []
    for i in ids:
        img = cv2.imread(str(TEST_IMAGES / f"{i}.png"))
        if img is None:
            raise SystemExit(f"unreadable test image {i}")
        if corruption:
            img = apply_corruption(img, corruption, severity, frame_id=i)
        fr = monitor.process(img, label=f"{tag}/{i}")
        row = dict(fr.row)
        row["corruption"] = corruption or "none"
        row["severity"] = severity or "none"
        log_rows.append(row)
        gt_boxes, gt_classes = load_gt(i, img.shape[1], img.shape[0])
        frames.append(
            {
                "boxes": fr.boxes,
                "classes": fr.classes,
                "confs": fr.confs,
                "gt_boxes": gt_boxes,
                "gt_classes": gt_classes,
            }
        )

    scores = np.array([r["monitor_score"] for r in log_rows])
    states = [r["state_after"] for r in log_rows]
    lat = np.array([r["latency_ms"] for r in log_rows])
    first_transition = next(
        (r["frame_id"] for r in log_rows if r["transition_reason"] != "no_change"), ""
    )
    p, rec = precision_recall(frames)
    metrics = {
        "date": date.today().isoformat(),
        "experiment": "EXP-010",
        "corruption": corruption or "none",
        "severity": severity or "none",
        "n_frames": len(ids),
        "backend": monitor.backend,
        "mAP50": round(mean_ap(frames), 4),
        "mAP50_95": round(mean_ap_50_95(frames), 4),
        "precision": round(p, 4),
        "recall": round(rec, 4),
        "mean_score": round(float(scores.mean()), 4),
        "max_score": round(float(scores.max()), 4),
        "nominal_fraction": round(states.count(NOMINAL) / len(states), 4),
        "degraded_fraction": round(states.count(DEGRADED) / len(states), 4),
        "fail_safe_fraction": round(states.count(FAIL_SAFE_REQUEST) / len(states), 4),
        "first_transition_frame": first_transition,
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
        "within_budget_p95": bool(np.percentile(lat, 95) < 40.0),
    }
    print(
        f"{tag:24s} mAP50 {metrics['mAP50']:.3f}  degraded {metrics['degraded_fraction']:.2f}  "
        f"fail-safe {metrics['fail_safe_fraction']:.2f}  p95 {metrics['latency_ms_p95']:.1f} ms"
    )
    return metrics, log_rows


def plot_curves(metrics_rows: list[dict], prefix: str = "fault_injection") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sev_x = {"none": 0, "low": 1, "medium": 2, "high": 3}
    clean = next(r for r in metrics_rows if r["corruption"] == "none")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, ylabel in (
        (axes[0], "mAP50", "mAP50 (kitti-test subset)"),
        (axes[1], "nominal_fraction", "accepted (NOMINAL) fraction"),
        (axes[2], "fail_safe_fraction", "FAIL_SAFE_REQUEST fraction"),
    ):
        for c in CORRUPTIONS:
            xs, ys = [0], [clean[key]]
            for s in SEVERITIES:
                r = next(
                    (m for m in metrics_rows if m["corruption"] == c and m["severity"] == s), None
                )
                if r:
                    xs.append(sev_x[s])
                    ys.append(r[key])
            ax.plot(xs, ys, marker="o", label=c)
        ax.set_xticks(list(sev_x.values()), list(sev_x.keys()))
        ax.set_xlabel("severity")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    fig.suptitle("Fault injection: severity vs detection quality and monitor response (EXP-010)")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{prefix}_curves.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_FRAMES)
    ap.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS))
    ap.add_argument("--severities", nargs="*", default=list(SEVERITIES))
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--out-prefix", default="fault_injection", help="results file prefix")
    args = ap.parse_args()

    ids = subset_ids(args.n)
    subset_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    print(f"fault injection: {len(ids)} kitti-test frames (report-only), {len(args.corruptions)}x{len(args.severities)} conditions")

    all_metrics, all_logs = [], []
    m, logs = run_condition(ids, None, None, args.weights)
    all_metrics.append(m)
    all_logs.extend(logs)
    for c in args.corruptions:
        for s in args.severities:
            m, logs = run_condition(ids, c, s, args.weights)
            all_metrics.append(m)
            all_logs.extend(logs)

    RESULTS.mkdir(exist_ok=True)
    prefix = args.out_prefix
    with (RESULTS / f"{prefix}_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
        w.writeheader()
        w.writerows(all_metrics)
    with (RESULTS / f"{prefix}_monitor_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_logs[0]))
        w.writeheader()
        w.writerows(all_logs)

    from src.monitor.runtime import load_frozen_thresholds

    q95, q99, method = load_frozen_thresholds()
    clean = all_metrics[0]
    worst = min(all_metrics[1:], key=lambda r: r["mAP50"], default=None)
    most_flagged = max(all_metrics[1:], key=lambda r: 1 - r["nominal_fraction"], default=None)
    summary = {
        "experiment": "EXP-010",
        "date": date.today().isoformat(),
        "seed": SEED,
        "n_frames": len(ids),
        "subset_source": "configs/splits/test.txt (kitti-test, REPORT-ONLY; thresholds frozen from kitti-val in EXP-008)",
        "subset_sha256": subset_hash,
        "corruptions": args.corruptions,
        "severities": args.severities,
        "backend": clean["backend"],
        "thresholds": {"method": method, "q95": q95, "q99": q99},
        "headline": {
            "clean_mAP50": clean["mAP50"],
            "worst_condition": f"{worst['corruption']}:{worst['severity']}" if worst else "",
            "worst_mAP50": worst["mAP50"] if worst else "",
            "most_flagged_condition": (
                f"{most_flagged['corruption']}:{most_flagged['severity']}" if most_flagged else ""
            ),
            "most_flagged_nonnominal_fraction": (
                round(1 - most_flagged["nominal_fraction"], 4) if most_flagged else ""
            ),
            "all_conditions_within_latency_budget": all(r["within_budget_p95"] for r in all_metrics),
        },
        "command": " ".join(sys.argv),
    }
    (RESULTS / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    if len(all_metrics) > 1:
        plot_curves(all_metrics, prefix)
    print(json.dumps(summary["headline"], indent=2))
    print(f"wrote {RESULTS / (prefix + '_metrics.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
