# CV & Portfolio Material

All figures below trace to `results/report_summary.json` and `docs/experiment_log.md`. Use only these values — do not round differently or add unverified claims.

## CV bullet (canonical)

> Built a safety-monitored ADAS perception prototype using YOLOv8n on KITTI with TensorRT FP16 deployment, calibrated OOD monitoring, and STPA/HARA-derived traceability; achieved 0.856 mAP50 at 17.2 ms p95 full-loop latency, flagged 99% of severe synthetic-fog frames, and documented a low-light monitor blind spot as a SOTIF residual risk.

## Portfolio short description

> A camera-only pedestrian/vehicle/cyclist detector for an AEB-style ADAS function, built to answer a safety-engineering question rather than an accuracy one: *when should this perception stack not be trusted?* YOLOv8n (KITTI, 0.856 mAP50 in TensorRT FP16) is supervised by a runtime monitor — temperature-scaled confidence plus a frame-level OOD score with validation-frozen Q95/Q99 thresholds — that drives a NOMINAL/DEGRADED/FAIL_SAFE_REQUEST state machine at 17.2 ms p95, inside a 40 ms budget. The monitor separates night from in-distribution imagery at 0.982 AUROC and flags 99% of frames under severe synthetic fog; a fault-injection campaign also *quantifies where it fails* — gradual low light removes 19% of mAP50 while the monitor stays 97.7% NOMINAL, documented as a SOTIF residual. The work is packaged as a bounded safety case: STPA/HARA hazard analysis, six verified safety requirements with end-to-end traceability, a GSN argument, a SOTIF classification, and an ISO/PAS 8800 alignment table — all claims scoped to a defined ODD, none claiming certification.

## Talking points (interview)

- **The thesis, not the accuracy:** the deliverable is the *evidence chain* from hazard to metric, and the honesty of documenting the monitor's blind spot — most demos hide theirs.
- **Evidence hygiene:** thresholds set on validation quantiles only; test/shift slices report-only; seeds + split hashes committed; 88 automated tests; no metric hand-typed (all from result CSVs via `build_report_assets`).
- **The low-light finding:** shows understanding that a runtime monitor is itself a component with failure modes — frame-level max-confidence is blind to recall erosion, which is a real SOTIF unknown-unsafe → known-unsafe transition.
- **Scope discipline:** Mahalanobis / temporal checks / INT8 were pre-registered cuts, not omissions; the safety artifacts were protected first.

## Facts sheet (for filling forms — verified values)

| Field | Value | Source |
|---|---|---|
| Model / dataset | YOLOv8n / KITTI (3 classes) | EXP-003 |
| mAP50 (PyTorch / TRT FP16) | 0.8588 / 0.8564 | `results/baseline_metrics.csv` |
| Full-loop latency p95 (TRT) | 17.2 ms (40 ms budget) | `results/monitor_latency_metrics.csv` |
| Calibration ECE | 0.081 → 0.039 | `results/calibration_metrics.csv` |
| OOD night AUROC | 0.982 | `results/ood_metrics.csv` |
| Fog:high frames flagged | 99% (mAP50 → 0.071) | `results/fault_injection_metrics.csv` |
| Low-light residual | −19% mAP50, 97.7% NOMINAL | `results/fault_injection_summary.json` |
| Tests | 88 passing | `tests/` |
| Safety requirements | SR-01..06, all verified | `safety/traceability_matrix.csv` |

## Do-not-say list

- Not "certified", "ISO-compliant", "production-ready", or "proven safe".
- Not "real-time on Jetson" (measured on RTX 3050 Ti laptop only).
- Not "detects all OOD" (low-light blind spot is explicit).
