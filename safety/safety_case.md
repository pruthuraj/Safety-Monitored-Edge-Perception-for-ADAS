# Safety Case — Safety-Monitored Edge Perception for ADAS

Status: Week 8 deliverable (EXP-011). Diagram: `gsn.mmd` / `gsn.svg`. Companions: `sotif_argument.md`, `iso_pas_8800_mapping.md`, `evidence_index.csv`, `verification_report.md`.

> **Claim discipline.** This safety case argues *alignment* with safety-engineering practice on scripted, reproducible evidence. It does **not** claim certification, ISO compliance, proof of safety, or vehicle-level safety. All claims are bounded by the ODD, the assumptions, and the residual risks stated below.

## 1. Top Claim (G1)

> Within the defined KITTI-like ODD and stated assumptions, the perception monitor provides evidence-backed detection of degraded perception conditions and requests degraded/fail-safe handling within the 40 ms budget.

Context bounding the claim:

- **C1 — ODD:** daylight, clear/lightly overcast, KITTI-like road scenes, three classes (`docs/project_spec.md` §2).
- **C2 — Item boundary and assumptions:** camera → detector → monitor → simulated planner boundary; A-01..A-09 (`hara_lite.md` §1, §5), most critically A-03 (planner honors monitor states — untested by design).
- **C3 — Threshold provenance:** Q95/Q99 frozen from kitti-val only (`results/monitor_thresholds.json`, EXP-008); kitti-test and BDD never used for tuning.

## 2. Argument Structure (S1)

The claim decomposes into five sub-goals (see `gsn.svg`):

| Goal | Argument | Solution (evidence) |
|---|---|---|
| G2 — Hazards drive requirements | HARA-lite (S3/E4/C3 → ASIL D worked example) + STPA (UCAs, causal scenarios) derive SR-01..SR-06; every requirement traces to H/UCA/CS/LS references | Sn1: `hara_lite.md`, `stpa_report.md`, `requirements.csv` |
| G3 — Baseline quality measured | YOLOv8n on deterministic KITTI split: val mAP50 0.8588; TRT FP16 delta −0.0024 | Sn2: `results/baseline_metrics.csv` (EXP-003/004) |
| G4 — Monitor detects degradation | Calibration (ECE 0.081→0.039); OOD separation (night AUROC 0.982); frozen-threshold gating (7/7 scenarios, zero false transitions on ID); fault-injection response (fog:high 99% flagged) | Sn3/Sn4/Sn5/Sn7: EXP-006..010 artifacts |
| G5 — Latency within budget | Full monitor loop p95 17.20 ms TRT / 17.50 ms PyTorch vs 40 ms; all 16 fault conditions within budget | Sn6: `results/monitor_latency_metrics.csv`, `results/fault_injection_metrics.csv` |
| G6 — Verified and traceable | SR-by-SR verification narrative; machine-readable traceability, all SR `verified`; 88 automated tests | Sn8: `verification_report.md`, `traceability_matrix.csv` |

Full claim→evidence linkage: `evidence_index.csv`.

## 3. Residual Risks (R1)

Explicitly carried, not argued away:

1. **Low-light blind spot (primary residual).** `low_light:high` removes 19% of mAP50 while the monitor stays 97.7% NOMINAL (EXP-010). Frame-level max-confidence cannot see gradual recall erosion. Treated as SOTIF unknown-unsafe residual (`sotif_argument.md` §4); candidate mitigations (detection-count plausibility, temporal consistency, feature-space OOD) are future work, not claimed.
2. **No closed-loop validation.** Fail-safe is a *request*; planner/actuator behavior is assumption A-03. No vehicle, no hardware-in-the-loop.
3. **Synthetic triggering conditions.** Corruptions are plausibility models (fog = uniform haze), not physics-validated; BDD fog slice is n=13.
4. **Single-domain thresholds.** Q95/Q99 derive from KITTI val; the clear-day transfer control (AUROC ~0.76) shows even in-ODD domain shift is detectable, so thresholds do not transfer across camera setups without re-derivation.
5. **Educational HARA ratings.** S/E/C values are illustrative, not calibrated against field data.

## 4. Assumptions Register

A-01..A-06 (`docs/project_spec.md` §3), A-07..A-09 (`hara_lite.md` §5). Violation of any assumption invalidates the top claim; A-03 is the single largest exposure (CS-09 in `stpa_report.md`).

## 5. Evidence Quality

- All metrics script-generated (never hand-typed), commands + environment logged per experiment in `docs/experiment_log.md` (EXP-001..EXP-010).
- Fixed seed 42; splits and slice lists committed with SHA-256 manifests.
- Test/tuning separation enforced: thresholds from kitti-val only; kitti-test report-only (first touched in Week 7, after freeze).
- 88 automated tests passing at package time.
- Dataset provenance recorded incl. the BDD100K mirror substitution (`docs/dataset_splits.md`).

## 6. Verdict

The argument supports the bounded top claim G1 with the residual risks of §3 explicitly open. The monitor is demonstrably effective against abrupt, severe degradation (night, fog, blur, noise) and demonstrably *incomplete* against gradual low-signal degradation — and the safety case's core contribution is documenting both sides with equal rigor.
