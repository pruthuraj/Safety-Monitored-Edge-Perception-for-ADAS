# v1.0 — Safety-Monitored Edge Perception for ADAS

First complete release. A camera-only pedestrian/vehicle/cyclist detector for an AEB-style ADAS function, supervised by a runtime monitor that requests degraded/fail-safe handling when perception evidence degrades — packaged with the full safety-engineering evidence chain from hazard analysis to verified requirements. Built over a 10-week plan (EXP-001..013); all metrics script-generated, seed 42, RTX 3050 Ti Laptop 4 GB.

> Not certified, not ISO-compliant, not proven safe. An evidence-backed safety argument bounded to a defined KITTI-like ODD, with residual risks stated openly.

## Why I built this

Two earlier threads collided into this project. I first stumbled on YOLO while doing an IoT seminar research project — that's where real-time object detection on constrained edge hardware stopped being an abstraction and became something I wanted to actually run. Separately, working on an ECG digital-twin project is where I started taking *safety* seriously as an engineering discipline: a model that's usually right isn't enough when being wrong has consequences; you have to reason about when the model should not be trusted, and make that failure observable.

This project is the deliberate intersection of those two. Take an edge-ML detector (the YOLO thread) and wrap it in the safety mindset (the digital-twin thread): don't just report accuracy, build the runtime evidence and the argument for *when the perception stack should not be trusted*. The low-light blind spot below is exactly that instinct paying off — the interesting result isn't that the monitor works, it's that the campaign found and documented where it doesn't.

## Headline results

| | |
|---|---|
| Detector (KITTI val) | mAP50 **0.859** PyTorch / **0.856** TensorRT FP16 (Δ −0.0024) |
| Full-loop latency | p95 **17.2 ms** TRT / 17.5 ms PyTorch vs 40 ms budget |
| Calibration | ECE **0.081 → 0.039** (temperature scaling, T=0.600) |
| OOD (night) | AUROC **0.982**; 6% of night frames accepted at Q95 |
| Fault response | severe fog: **99%** of frames flagged as mAP50 collapses to 0.071 |
| Documented residual | severe low-light: **−19% mAP50 at 97.7% NOMINAL** (monitor blind spot) |
| Verification | SR-01..06 all verified; **88** automated tests |

## What's in the box

- **Perception + monitor pipeline** — YOLOv8n (KITTI) + TensorRT FP16; temperature-scaled confidence; frame-level max-confidence OOD score; validation-frozen Q95/Q99 thresholds; `NOMINAL / DEGRADED / FAIL_SAFE_REQUEST` sustained-breach state machine; per-frame evidence logging.
- **Safety artifacts** — HARA-lite + STPA, six measurable requirements with end-to-end traceability, GSN safety case (`safety/gsn.svg`), SOTIF argument, ISO/PAS 8800 alignment table, 24-row evidence index, SR-by-SR verification report.
- **Evidence** — machine-readable `results/*.csv` + `report_summary.json`; every README/paper/CV number traces to a committed result file.
- **Portfolio** — IEEE-style paper (`paper/main.pdf` + markdown source), ~72 s demo (`demo/final_demo.mp4`), overlay GIF, reproduction guide, CV material.

## The point of the project

The deliverable is the *evidence chain* — hazard → requirement → metric → result — and the honesty of documenting the monitor's blind spot. A fault-injection campaign shows the monitor firing exactly where detection collapses (night, fog, blur, noise), then quantifies where it stays silent: gradual low-light recall erosion. That failure is carried as a known-unsafe SOTIF residual, not hidden.

## Known limitations / residual risks

- **Low-light blind spot** — frame-level max-confidence cannot see recall erosion; mitigations (detection-count plausibility, temporal consistency, feature-space Mahalanobis) are stated future work, not implemented.
- **No closed-loop validation** — planner/actuator are simulated boundaries; fail-safe is a *request* (assumption A-03, untested).
- **Synthetic corruptions** are plausibility models, not weather physics; BDD fog slice n=13.
- **Single-domain thresholds** — frozen from KITTI val; do not transfer across camera domains (clear-day control AUROC 0.76).
- INT8 deferred (FP16 met budget); HARA ratings illustrative; single hardware target.

## Reproduce

See [`docs/final_reproduction.md`](docs/final_reproduction.md) for environment, data expectations, the exact command chain (EXP-002..013), and expected outputs. Datasets and weights are gitignored; retrain/re-export to regenerate.

## Provenance note

The official BDD100K mirror (`dl.cv.ethz.ch`) was offline during this work; val images and frame attributes were sourced from Hugging Face mirrors and converted, with 10000/10000 image↔label match verified. Full provenance in [`docs/dataset_splits.md`](docs/dataset_splits.md).
