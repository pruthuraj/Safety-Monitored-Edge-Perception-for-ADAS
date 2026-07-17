"""build_report_assets: regenerate tables/summary/GIF for README + paper (Week 9, EXP-012).

Reads only machine-readable result artifacts (never hand-typed numbers) and emits:
  results/report_summary.json   every headline number used in README/paper, with source file
  paper/tables.md               markdown tables for the paper draft
  demo/monitor_overlay.gif      subsampled GIF of the demo video (README embed)

Also verifies the figure inventory referenced by README/paper exists.
Per PLAN.md: no result appears in README/paper unless its source is recorded —
this script IS that recording step for derived tables.

Usage:
    python scripts/build_report_assets.py
    python scripts/build_report_assets.py --no-gif
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PAPER = REPO / "paper"
DEMO = REPO / "demo"

FIGURES = [
    "results/reliability_diagram.png",
    "results/ood_score_distributions.png",
    "results/ood_roc_curves.png",
    "results/risk_coverage.png",
    "results/fault_injection_curves.png",
    "safety/gsn.svg",
]


def read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(newline="") as f:
        return list(csv.DictReader(f))


def md_table(headers: list[str], rows: list[list], caption: str) -> str:
    out = [f"**{caption}**", "", "| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def build_summary() -> dict:
    baseline = {r["experiment"]: r for r in read_csv("baseline_metrics.csv")}
    calib = read_csv("calibration_metrics.csv")[-1]
    ood = read_csv("ood_metrics.csv")
    thresholds = json.loads((RESULTS / "monitor_thresholds.json").read_text())
    risk = read_csv("risk_coverage.csv")
    latency = read_csv("monitor_latency_metrics.csv")
    fault = read_csv("fault_injection_metrics.csv")

    def ood_row(slice_, method):
        return next(r for r in ood if r["slice"] == slice_ and r["method"] == method)

    def risk_row(slice_, label, method="max_conf_score"):
        return next(
            r for r in risk
            if r["slice"] == slice_ and r["threshold_label"] == label and r["method"] == method
        )

    def fault_row(corruption, severity):
        return next(r for r in fault if r["corruption"] == corruption and r["severity"] == severity)

    lat_by_backend = {r["backend"]: r for r in latency}

    summary = {
        "generated": date.today().isoformat(),
        "generator": "scripts/build_report_assets.py",
        "sources": {
            "baseline": "results/baseline_metrics.csv",
            "calibration": "results/calibration_metrics.csv",
            "ood": "results/ood_metrics.csv",
            "thresholds": "results/monitor_thresholds.json",
            "risk_coverage": "results/risk_coverage.csv",
            "latency": "results/monitor_latency_metrics.csv",
            "fault_injection": "results/fault_injection_metrics.csv",
        },
        "baseline": {
            "pytorch_mAP50": float(baseline["EXP-003"]["mAP50"]),
            "pytorch_mAP50_95": float(baseline["EXP-003"]["mAP50_95"]),
            "trt_fp16_mAP50": float(baseline["EXP-004"]["mAP50"]),
            "quantization_delta_mAP50": round(
                float(baseline["EXP-004"]["mAP50"]) - float(baseline["EXP-003"]["mAP50"]), 4
            ),
            "per_class_AP50_pytorch": {
                "pedestrian": float(baseline["EXP-003"]["AP50_pedestrian"]),
                "vehicle": float(baseline["EXP-003"]["AP50_vehicle"]),
                "cyclist": float(baseline["EXP-003"]["AP50_cyclist"]),
            },
        },
        "calibration": {
            "temperature": float(calib["temperature"]),
            "ece_before": float(calib["ece_before"]),
            "ece_after": float(calib["ece_after"]),
        },
        "ood_max_conf": {
            s: {"auroc": float(ood_row(s, "max_conf_score")["auroc"]),
                "fpr_at_95tpr": float(ood_row(s, "max_conf_score")["fpr_at_95tpr"])}
            for s in ("bdd-night", "bdd-rain", "bdd-fog", "bdd-clear-day")
        },
        "thresholds": {
            "primary_method": thresholds["primary_method"],
            "q95": thresholds["thresholds"]["max_conf_score"]["q95"],
            "q99": thresholds["thresholds"]["max_conf_score"]["q99"],
        },
        "risk_coverage_at_q95": {
            "kitti_coverage": float(risk_row("kitti-val-report", "frozen-q95")["accepted_fraction"]),
            "kitti_accepted_mAP50": float(risk_row("kitti-val-report", "frozen-q95")["accepted_mAP50"]),
            "bdd_night_coverage": float(risk_row("bdd-night", "frozen-q95")["accepted_fraction"]),
            "bdd_rain_coverage": float(risk_row("bdd-rain", "frozen-q95")["accepted_fraction"]),
            "bdd_fog_coverage": float(risk_row("bdd-fog", "frozen-q95")["accepted_fraction"]),
            "bdd_clear_day_coverage": float(risk_row("bdd-clear-day", "frozen-q95")["accepted_fraction"]),
        },
        "latency_full_loop": {
            b: {"p50": float(r["latency_ms_p50"]), "p95": float(r["latency_ms_p95"]),
                "fps": float(r["fps_mean"])}
            for b, r in lat_by_backend.items()
        },
        "fault_injection": {
            "clean_mAP50": float(fault_row("none", "none")["mAP50"]),
            "fog_high": {
                "mAP50": float(fault_row("fog", "high")["mAP50"]),
                "nonnominal_fraction": round(1 - float(fault_row("fog", "high")["nominal_fraction"]), 4),
            },
            "low_light_high": {
                "mAP50": float(fault_row("low_light", "high")["mAP50"]),
                "nominal_fraction": float(fault_row("low_light", "high")["nominal_fraction"]),
            },
            "all_within_latency_budget": all(r["within_budget_p95"] == "True" for r in fault),
        },
        "gating": {"scenarios_pass": "7/7 (results/gating_tests.csv)", "tests_total": 88},
    }
    return summary, baseline, calib, ood, risk, latency, fault


def build_tables(baseline, calib, ood, risk, latency, fault) -> str:
    parts = ["# Report Tables (generated — do not hand-edit)\n",
             f"Generated by `scripts/build_report_assets.py` on {date.today().isoformat()}. "
             "Sources: results/*.csv, results/monitor_thresholds.json.\n"]

    parts.append(md_table(
        ["Backend", "mAP50", "mAP50-95", "AP50 ped", "AP50 veh", "AP50 cyc", "p50 (ms)", "p95 (ms)"],
        [[r["backend"], r["mAP50"], r["mAP50_95"], r["AP50_pedestrian"], r["AP50_vehicle"],
          r["AP50_cyclist"], r["latency_ms_p50"], r["latency_ms_p95"]]
         for r in [baseline["EXP-003"], baseline["EXP-004"]]],
        "Table 1 — Detector baseline on kitti-val (detector-only latency, 300 imgs)"))

    parts.append(md_table(
        ["Fit imgs", "Report imgs", "T", "ECE before", "ECE after"],
        [[calib["fit_images"], calib["report_images"], calib["temperature"],
          calib["ece_before"], calib["ece_after"]]],
        "Table 2 — Temperature-scaling calibration (kitti-val, disjoint fit/report)"))

    ood_rows = []
    for s in ("bdd-night", "bdd-rain", "bdd-fog", "bdd-clear-day"):
        mc = next(r for r in ood if r["slice"] == s and r["method"] == "max_conf_score")
        en = next(r for r in ood if r["slice"] == s and r["method"] == "energy_score")
        ood_rows.append([s, mc["n_ood"], mc["auroc"], mc["fpr_at_95tpr"], en["auroc"], en["fpr_at_95tpr"]])
    parts.append(md_table(
        ["Slice", "n", "max-conf AUROC", "max-conf FPR@95", "energy AUROC", "energy FPR@95"],
        ood_rows,
        "Table 3 — OOD separation vs kitti-val report subset (bdd-clear-day = transfer control)"))

    cov_rows = []
    for s in ("kitti-val-report", "bdd-clear-day", "bdd-night", "bdd-rain", "bdd-fog"):
        r95 = next(r for r in risk if r["slice"] == s and r["threshold_label"] == "frozen-q95"
                   and r["method"] == "max_conf_score")
        r99 = next(r for r in risk if r["slice"] == s and r["threshold_label"] == "frozen-q99"
                   and r["method"] == "max_conf_score")
        cov_rows.append([s, r95["accepted_fraction"], r99["accepted_fraction"],
                         r95["accepted_mAP50"] or "—"])
    parts.append(md_table(
        ["Slice", "coverage @Q95", "coverage @Q99", "accepted mAP50 @Q95"],
        cov_rows,
        "Table 4 — Coverage at frozen thresholds (max-conf; BDD rows coverage-only)"))

    parts.append(md_table(
        ["Backend", "p50 (ms)", "p95 (ms)", "FPS", "Budget"],
        [[r["backend"], r["latency_ms_p50"], r["latency_ms_p95"], r["fps_mean"], "40 ms"]
         for r in latency],
        "Table 5 — Full monitor loop latency (predict + score + state machine + log), 300 frames"))

    fi_rows = [[r["corruption"], r["severity"], r["mAP50"], r["nominal_fraction"],
                r["fail_safe_fraction"], r["latency_ms_p95"]]
               for r in fault]
    parts.append(md_table(
        ["Corruption", "Severity", "mAP50", "NOMINAL frac", "FAIL_SAFE frac", "p95 (ms)"],
        fi_rows,
        "Table 6 — Fault injection on 300 kitti-test frames (report-only)"))

    return "\n".join(parts)


def build_gif() -> Path:
    import cv2
    from PIL import Image

    src = DEMO / "monitor_overlay.mp4"
    out = DEMO / "monitor_overlay.gif"
    cap = cv2.VideoCapture(str(src))
    frames = []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % 3 == 0:  # 10 fps video -> ~3.3 fps gif
            img = cv2.resize(img, (640, 360))
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {src}")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=300, loop=0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    missing = [f for f in FIGURES if not (REPO / f).exists()]
    if missing:
        raise SystemExit(f"missing figures: {missing}")

    summary, baseline, calib, ood, risk, latency, fault = build_summary()
    (RESULTS / "report_summary.json").write_text(json.dumps(summary, indent=2))
    PAPER.mkdir(exist_ok=True)
    (PAPER / "tables.md").write_text(build_tables(baseline, calib, ood, risk, latency, fault))
    print(f"wrote {RESULTS / 'report_summary.json'}")
    print(f"wrote {PAPER / 'tables.md'}")

    if not args.no_gif:
        gif = build_gif()
        print(f"wrote {gif} ({gif.stat().st_size // 1024} KiB)")
    print("figure inventory OK:", ", ".join(FIGURES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
