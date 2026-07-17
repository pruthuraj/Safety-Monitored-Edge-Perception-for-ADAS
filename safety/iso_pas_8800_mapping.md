# ISO/PAS 8800 Alignment Mapping

Status: Week 8 deliverable (EXP-011). Reference: [ISO/PAS 8800:2024 — Road vehicles, Safety and artificial intelligence](https://www.iso.org/standard/83303.html). GSN notation per [SCSC GSN standardization](https://scsc.uk/gsn); SOTIF framing per [ISO 21448:2022](https://www.iso.org/standard/77490.html).

> **Alignment, not compliance.** This table maps project activities to AI-safety lifecycle *themes* of ISO/PAS 8800. It asserts that each theme was consciously addressed at educational-project depth — not that any clause is satisfied, audited, or certified. The standard text was not licensed for this project; theme names are paraphrased public-domain concepts.

| # | AI-safety lifecycle theme | Project activity | Artifacts | Gaps at project scope |
|---|---|---|---|---|
| 1 | Definition of the AI function and its operating context (ODD) | Item definition, ODD table, class scope, out-of-scope list | `docs/project_spec.md` §1–2 | ODD is dataset-proxied (KITTI-like), not vehicle-programme derived |
| 2 | AI-specific hazard and risk analysis | HARA-lite worked derivation + STPA with AI-specific causal scenarios (OOD input, miscalibration, monitor FN/FP, threshold misuse) | `safety/hara_lite.md`, `safety/stpa_report.md` | Single worked ASIL derivation; ratings illustrative |
| 3 | Safety requirements on the AI system | SR-01..SR-06 with measurable acceptance criteria, derived from SG-01 through UCAs | `safety/requirements.csv` | Requirement set is monitor-centric; no requirements on training data quality beyond validation checks |
| 4 | Data requirements and data quality | Dataset validator (fail-loud unmapped classes), deterministic committed splits with hashes, provenance records incl. mirror substitution | `src/dataset/validate_kitti.py`, `configs/splits/`, `docs/dataset_splits.md` | No bias/coverage analysis of training data; single dataset family in-ODD |
| 5 | AI model performance evidence | Baseline + quantized evaluation on held-out split; per-class AP reported; quantization delta measured | `results/baseline_metrics.csv` (EXP-003/004) | No formal performance-requirement decomposition from vehicle level |
| 6 | Uncertainty representation and calibration | Temperature scaling with before/after ECE on disjoint subset | `results/calibration_metrics.csv` (EXP-006) | Scalar T; no per-class or spatial calibration |
| 7 | Runtime monitoring and safe-state management | Frame-level OOD score, frozen quantile thresholds, 3-state gating with sustained-breach logic, fail-safe request | `src/monitor/*`, `results/monitor_thresholds.json`, `results/gating_tests.csv` (EXP-007..009) | Single monitor signal (max-conf); known low-light blind spot |
| 8 | Verification and validation of the AI system | Unit+integration tests (88), scenario replay, fault injection across 15 corruption conditions, SR-by-SR verification report | `tests/`, `results/fault_injection_*`, `safety/verification_report.md` (EXP-010) | Synthetic corruptions; no field validation, no closed loop |
| 9 | Traceability through the lifecycle | Every SR: derivation → implementation → test → metric → result → evidence file, machine-readable | `safety/traceability_matrix.csv`, `safety/evidence_index.csv` | — |
| 10 | Safety case / safety argumentation | GSN one-pager + written case with bounded top claim and explicit residual-risk nodes | `safety/safety_case.md`, `safety/gsn.svg` | Educational depth; no independent assessment |
| 11 | Operation-phase considerations (monitoring in the field, updates) | Per-frame machine-readable logging designed for post-hoc reconstruction | `results/runtime_monitor_log.csv` schema (SR-05) | No fleet feedback loop, no update/retraining process |

## Reading the table

Themes 1–3 land in Weeks 1–3 (definition + analysis), 4–6 in Weeks 2–4 (data + model + calibration), 7 in Weeks 5–6 (runtime), 8–10 in Weeks 7–8 (V&V + case), 11 partially (logging only). The "Gaps" column is deliberate: an alignment table that lists no gaps would itself violate this project's no-unsourced-claims rule.
