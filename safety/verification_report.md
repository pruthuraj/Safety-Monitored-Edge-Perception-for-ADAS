# Verification Report — SR-01..SR-06

Status: Week 7 deliverable (EXP-010). Companion: `requirements.csv` (requirement wording and acceptance criteria), `traceability_matrix.csv` (machine-readable status), `hara_lite.md` / `stpa_report.md` (derivation).

> **Scope statement.** Verification here means: each safety requirement's acceptance criteria were checked by scripted, reproducible evidence on the defined datasets within the defined ODD. It does **not** mean certified, ISO-compliant, or proven safe. No closed-loop vehicle testing was performed; the planner/actuator remain simulated boundaries (assumption A-03).

## Evidence base

| Experiment | Content | Key artifacts |
|---|---|---|
| EXP-003/004 | Detector baseline + TensorRT FP16 | `results/baseline_metrics.csv` |
| EXP-006 | Temperature calibration | `results/calibration_metrics.csv`, `results/calibration_params.json`, `results/reliability_diagram.png` |
| EXP-007 | OOD scoring vs BDD100K slices | `results/ood_metrics.csv`, `results/monitor_scores_*.csv`, `results/ood_*.png` |
| EXP-008 | Frozen thresholds + risk-coverage | `results/monitor_thresholds.json`, `results/risk_coverage.csv/.png` |
| EXP-009 | Runtime gating, logging, latency, demo | `results/gating_tests.csv`, `results/runtime_monitor_log*.csv`, `results/monitor_log_check.csv`, `results/monitor_latency_metrics.csv`, `demo/monitor_overlay.mp4` |
| EXP-010 | Fault injection (5 corruptions × 3 severities, 300 kitti-test frames, report-only) | `results/fault_injection_metrics.csv`, `results/fault_injection_monitor_log.csv`, `results/fault_injection_summary.json`, `results/fault_injection_curves.png` |

All experiments: seed 42, RTX 3050 Ti Laptop 4 GB, commands recorded in `docs/experiment_log.md`.

## SR-by-SR verification

### SR-01 — Calibrated detection confidence: **verified**
- **Acceptance:** ECE before/after temperature scaling reported on kitti-val; calibrated ECE improves.
- **Evidence:** EXP-006 — T = 0.6003 fitted on calibration-fit subset (561 imgs); **ECE 0.0812 → 0.0390** on disjoint calibration-report subset (5724 detections). Reliability diagram confirms post-scaling alignment.
- **Tests:** `tests/test_monitor.py` — ECE known-value cases, temperature monotonicity/identity, matching TP/FP/duplicate/no-pred cases.
- **Limitation:** single scalar T, IoU-0.5 correctness definition. The runtime monitor gates on raw max-confidence; temperature scaling is a strictly monotonic transform, so gating decisions are unchanged whether applied or not — calibration evidence stands for confidence *reporting*, not as a separate gating mechanism.

### SR-02 — Runtime OOD detection: **verified**
- **Acceptance:** AUROC and FPR@95 per OOD slice; energy-style score vs max-confidence baseline compared.
- **Evidence:** EXP-007 — max-conf AUROC/FPR@95: night **0.982/0.112**, rain 0.876/0.537, fog 0.926/0.419 (n=13), clear-day transfer control 0.758/0.624; energy score compared on identical slices (not materially better). EXP-008 froze primary = `max_conf_score` with justification. EXP-010 adds corruption response: fog:high flags 99% of frames non-NOMINAL as mAP collapses 0.847 → 0.071.
- **Tests:** `tests/test_monitor.py`, `tests/test_thresholds_risk.py`.
- **Limitation:** fog slice n=13 (unstable); frame-level post-NMS proxy, not feature-space OOD; **known blind spot: low_light** (see EXP-010 findings below).

### SR-03 — Degraded mode entry: **verified**
- **Acceptance:** DEGRADED entered only on sustained Q95 breach; single-frame spikes never transition.
- **Evidence:** EXP-009 — 7/7 deterministic scenario replays (`gating_tests.csv`), incl. single-spike rejection and streak-interruption reset; 300-frame ID run: zero false transitions.
- **Tests:** `tests/test_state_machine.py` (15 tests).

### SR-04 — Fail-safe request: **verified**
- **Acceptance:** FAIL_SAFE_REQUEST on sustained Q99 breach or prolonged degraded condition; thresholds from validation only.
- **Evidence:** EXP-009 — both trigger paths replayed (2×Q99; 10-frame degraded persistence); thresholds frozen from kitti-val only (`monitor_thresholds.json`, EXP-008). BDD-night integration test trips the monitor. EXP-010: fail-safe fraction rises with severity for fog/blur/noise (0.60–0.99 at medium-high).
- **Tests:** `tests/test_state_machine.py`.

### SR-05 — Monitor logging: **verified**
- **Acceptance:** every processed frame logs id, timestamp, backend, confidence summary, OOD score, thresholds, state, transition reason, latency.
- **Evidence:** EXP-009 — 300-frame log with all 13 fields; **15/15 completeness checks** (`monitor_log_check.csv`); one row per frame verified. EXP-010 log covers 4800 corrupted/clean frames with corruption/severity columns.
- **Tests:** log schema + row-count tests in `tests/test_state_machine.py`, `tests/test_fault_injection.py`.

### SR-06 — Latency budget: **verified**
- **Acceptance:** perception + monitor p95 < 40 ms on RTX 3050 Ti, ≥300 val images per backend.
- **Evidence:** EXP-009 full monitor loop (predict + score + state machine + log row): **TensorRT FP16 p95 17.20 ms (67.9 FPS)**, **PyTorch p95 17.50 ms (63.1 FPS)**. EXP-010: all 16 corruption conditions within budget (max p95 31.2 ms, in-memory ndarray path).
- **Tests:** latency summary schema test; budget comparison in every fault-injection row.

## EXP-010 headline findings

1. **Monitor tracks severe degradation.** For fog, motion blur, and gaussian noise, non-NOMINAL fraction rises with severity in step with mAP collapse (fog:high — mAP50 0.071, 99% flagged; blur:high — 0.251, 78% FSR; noise:high — 0.365, 77% FSR). The fail-safe mechanism fires exactly where detection quality is destroyed.
2. **Known blind spot: gradual low-signal degradation.** `low_light:high` drops mAP50 0.847 → 0.689 (−19%) while the monitor stays 97.7% NOMINAL; `dead_pixels:medium` similarly (0.791, 98% NOMINAL). Max-confidence stays high on confidently-detected near objects while recall erodes silently. This is a documented **SOTIF-style unknown-unsafe residual**: frame-level max-confidence cannot see recall loss. Candidate mitigations (out of MVP scope): detection-count plausibility, temporal consistency checks, feature-space OOD (Mahalanobis).
3. **Clean-condition behavior sane.** Clean kitti-test subset: mAP50 0.847, 98.3% NOMINAL (5 frames briefly DEGRADED — consistent with Q95 design leaving ~5% of ID frames above threshold, sustained-breach logic containing them).

## Global limitations

- Synthetic corruptions are plausibility models, not physics-validated weather; fog especially is uniform haze.
- KITTI test used report-only; no threshold or model selection touched it (frozen in EXP-008).
- No closed-loop vehicle test; fail-safe is a *request*, honored by assumption A-03.
- Thresholds are KITTI-val-derived; generalization beyond tested slices/corruptions is not claimed.
- mAP figures here use the project's standalone AP implementation (`src/monitor/detection_metrics.py`), not the ultralytics validator — comparable within this report, not across tools.
