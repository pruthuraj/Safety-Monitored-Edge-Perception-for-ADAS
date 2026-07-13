# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Safety-Monitored Edge Perception for ADAS: YOLOv8n object detection on KITTI (pedestrian/vehicle/cyclist), TensorRT FP16/INT8 export, calibrated confidence + OOD monitoring, fail-safe runtime gating, and safety artifacts (STPA/HARA, GSN, SOTIF, traceability matrix). The point is connecting edge ML performance to safety-engineering evidence — documenting when the perception stack should NOT be trusted, not just when it works.

`PLAN.md` is the authoritative 10-week execution plan. Read it before making scope or design decisions.

## Commands

Standard entry points (defined in PLAN.md; implemented as scripts as the project progresses):

- `train_baseline` — train/evaluate YOLOv8n on KITTI (seed 42, deterministic split)
- `export_trt` — export FP16/INT8 TensorRT engines
- `evaluate_monitor` — calibration, OOD metrics, risk-coverage
- `run_demo` — annotated video with monitor-state overlay
- `build_report_assets` — regenerate plots/tables for README and paper

Final metrics must run from scripts, never notebooks. Notebooks are for exploration only.

## Architecture

Module boundaries (keep them clean):

- **dataset loading** — KITTI (ID) and BDD100K slices (OOD: night/rain/fog + clear-day transfer control)
- **model inference** — YOLOv8n, PyTorch and TensorRT backends
- **monitor scoring** — temperature scaling calibration; energy-style OOD score (committed method), max-confidence baseline
- **state machine** — `NOMINAL` / `DEGRADED` / `FAIL_SAFE_REQUEST`; triggers on sustained threshold breaches, not single-frame spikes. Thresholds: `DEGRADED = Q95`, `FAIL_SAFE_REQUEST = Q99`, set from validation data only
- **metrics** — mAP, ECE, AUROC, FPR@95, risk-coverage, latency p50/p95 (40 ms budget)
- **visualization** — overlays, reliability diagrams, plots

Configs live separate from code.

Key artifacts land in fixed paths: `results/*.csv` (machine-readable metrics), `safety/` (requirements.csv, traceability_matrix.csv, stpa_report, sotif_argument.md, gsn.svg), `paper/`, `demo/`. See PLAN.md "Interfaces And Artifacts" for the full list.

## Non-negotiable rules

- **Traceability**: every monitor feature maps to an `SR-xx` safety requirement; every `SR-xx` maps to implementation → test → metric → result → evidence file (`safety/traceability_matrix.csv`).
- **Experiment hygiene**: thresholds tuned on validation only, never on final test slices. Keep ID / shifted / corrupted / demo evaluations separated.
- **No unsourced claims**: no result appears in README/paper/CV/demo without a recorded source command/config/result file. Update docs during the work (`docs/experiment_log.md`), not after.
- **Safety language**: never claim certified/ISO-compliant/proven safe. Use "aligned with", "argued using", "monitors for", "requests fail-safe", "within the defined ODD".
- **Scope cuts** (in order, if needed): Mahalanobis → CODA → temporal checks → YOLOv8s. Never cut: STPA, GSN, traceability, latency measurement, calibration, one working OOD/gating method.
- **Reproducibility**: fixed seed 42; record dataset versions, splits, preprocessing, weights, TensorRT settings, calibration data, hardware (default: local RTX 3050 Ti).
