# Safety-Monitored Edge Perception for ADAS: 10-Week Execution Plan

## Summary
Build a portfolio-grade ADAS perception project around a credible MVP: YOLOv8n object detection on KITTI, TensorRT FP16/INT8 export, calibrated confidence/OOD monitoring, fail-safe gating, latency evidence, and safety artifacts.

Primary goal: show that you can connect edge ML performance to safety engineering evidence. The project must document not only what works, but also when the perception stack should not be trusted.

## Weekly Execution
- **Week 1: Project Definition**
  - Define item: camera-only pedestrian/vehicle/cyclist perception for AEB-style ADAS.
  - Define narrow ODD: daylight, forward-facing camera, KITTI-style road scenes, three target classes.
  - Create repo structure, experiment log template, dataset split notes, and traceability matrix skeleton.
  - Deliverable: project spec, ODD definition, assumptions, hazards, initial `SR-xx` placeholders.

- **Week 2: Baseline Detector**
  - Use YOLOv8n as committed model; YOLOv8s is stretch.
  - Train/evaluate on KITTI with deterministic split and seed `42`.
  - Export TensorRT FP16 and INT8; measure mAP, latency p50/p95, FPS, and quantization delta.
  - Deliverable: baseline metrics table and reproducible training/export commands.

- **Week 3: Safety Requirements**
  - Produce HARA-lite for “undetected pedestrian/vehicle/cyclist in ego path.”
  - Build STPA control structure: camera -> perception -> planner/AEB -> brake actuator -> vehicle/driver/road.
  - Derive monitor-specific safety requirements: detection confidence, OOD detection, degraded mode, fail-safe request, logging, and latency.
  - Deliverable: STPA/HARA report and traceability matrix with requirement-to-test placeholders.

- **Week 4: Calibration + OOD MVP**
  - Implement temperature scaling and report ECE before/after calibration.
  - Build OOD slices: KITTI validation as ID; BDD100K subsets for night, rain, fog, and clear-day transfer control.
  - Implement max-confidence baseline and energy-style score.
  - Deliverable: reliability diagrams, AUROC, FPR@95, and score distribution plots.

- **Week 5: Thresholds + Risk-Coverage**
  - Select MVP monitor method by accuracy/latency; default to energy-style score.
  - Set thresholds from validation data only: `DEGRADED = Q95`, `FAIL_SAFE_REQUEST = Q99`.
  - Produce risk-coverage curves showing accepted-frame performance as thresholds vary.
  - Deliverable: threshold table, method comparison, risk-coverage plots.

- **Week 6: Runtime Gating Demo**
  - Implement state machine: `NOMINAL`, `DEGRADED`, `FAIL_SAFE_REQUEST`.
  - Trigger degraded/fail-safe states on sustained threshold breaches, not single-frame spikes.
  - Add overlay with detections, monitor score, state, threshold, and latency.
  - Deliverable: demo video and latency table against the 40 ms budget.

- **Week 7: Fault Injection + Verification**
  - Use synthetic corruptions: fog, blur, noise, low light, exposure drift, dead pixels.
  - Verify every `SR-xx` with named tests and recorded results.
  - Fill traceability matrix: requirement -> implementation -> test -> metric -> result -> evidence file.
  - Deliverable: verification report and completed traceability matrix.

- **Week 8: Safety Case**
  - Build one-page GSN with top goal: “perception is acceptably safe within defined ODD.”
  - Write SOTIF argument around triggering conditions, ODD limits, unknown unsafe scenarios, and monitor response.
  - Add ISO/PAS 8800:2024 alignment table, using it as an AI-safety mapping that extends ISO 26262/ISO 21448 thinking. Reference: [ISO/PAS 8800:2024](https://www.iso.org/obp/ui/en/).
  - Deliverable: safety case document.

- **Week 9: Paper + README**
  - Write 6-8 page IEEE-style paper.
  - README must show architecture, demo GIF, GSN figure, key metrics, reproduction commands, and limitations.
  - Deliverable: paper draft and portfolio-grade README.

- **Week 10: Packaging**
  - Record 60-90 second demo video.
  - Freeze final commands, environment notes, and measured results.
  - Update CV bullet using real values only.
  - Deliverable: shipped repo, demo video, paper PDF, final CV bullet.

## Project Rules
- **Documentation is mandatory**
  - Every experiment, result, threshold, safety claim, dataset split, limitation, and design decision must be documented.
  - No result can appear in the README, paper, CV, or demo unless its source command/config/result file is recorded.

- **Docs-as-you-go**
  - Update docs during the week, not at the end.
  - Each weekly deliverable must include: what changed, why it changed, command used, result, limitation, and next step.

- **Traceability first**
  - Maintain `SR-xx` IDs from Week 3 onward.
  - Every monitor feature must map to at least one safety requirement.
  - Every safety requirement must map to implementation, test, metric, and evidence.

- **Reproducibility**
  - Use fixed seeds where possible.
  - Record dataset versions, splits, preprocessing, model weights, TensorRT settings, calibration data, and hardware.
  - Store all metrics as machine-readable CSV/JSON plus human-readable plots.

- **Experiment hygiene**
  - Do not tune thresholds on final test slices.
  - Use validation data for thresholds and final held-out slices for reporting.
  - Keep ID, shifted, corrupted, and demo-video evaluations clearly separated.

- **Safety claim discipline**
  - Do not claim the system is certified, ISO-compliant, or proven safe.
  - Use precise language: “aligned with,” “argued using,” “monitors for,” “requests fail-safe,” and “within the defined ODD.”
  - Every safety claim must cite evidence or be listed as a limitation.

- **Scope control**
  - Preserve safety artifacts and the MVP monitor before adding extra methods.
  - Cut in this order if needed: Mahalanobis, CODA, temporal checks, YOLOv8s.
  - Never cut STPA, GSN, traceability, latency measurement, calibration, or one working OOD/gating method.

- **Code quality**
  - Keep configs separate from code.
  - Use clear module boundaries: dataset loading, model inference, monitor scoring, state machine, metrics, visualization.
  - Avoid notebook-only work for final results; notebooks are acceptable for exploration, but final metrics must run from scripts.

## Interfaces And Artifacts
- Standard commands:
  - `train_baseline`: train/evaluate YOLOv8n on KITTI.
  - `export_trt`: export FP16/INT8 TensorRT engines.
  - `evaluate_monitor`: run calibration, OOD metrics, and risk-coverage.
  - `run_demo`: generate annotated driving video with monitor-state overlay.
  - `build_report_assets`: regenerate plots/tables for README and paper.

- Required artifacts:
  - `results/baseline_metrics.csv`
  - `results/ood_metrics.csv`
  - `results/latency_metrics.csv`
  - `results/risk_coverage.csv`
  - `safety/requirements.csv`
  - `safety/traceability_matrix.csv`
  - `safety/stpa_report.pdf`
  - `safety/sotif_argument.md`
  - `safety/iso_pas_8800_mapping.md`
  - `safety/gsn.svg`
  - `paper/main.pdf`
  - `demo/monitor_overlay.mp4`

## Test Plan
- Calibration: ECE before/after temperature scaling and reliability diagrams.
- OOD: AUROC, FPR@95, score histograms, and per-slice results for KITTI/BDD100K/corruptions.
- Gating: deterministic tests for nominal, degraded, fail-safe, recovery, and noisy threshold crossings.
- Latency: p50/p95 for PyTorch, TensorRT FP16, TensorRT INT8, and TensorRT INT8 plus monitor.
- Traceability: every `SR-xx` has a linked test, result, and evidence artifact.
- Documentation audit: before final packaging, verify that every README/paper/CV claim maps to a result file.

## Assumptions And Defaults
- Time budget is 10-12 hours/week.
- Hardware default is local RTX 3050 Ti; Jetson is optional.
- YOLOv8n is the committed model; YOLOv8s is stretch.
- Energy-style OOD score is the committed monitor method unless validation shows max-confidence is better.
- Mahalanobis, CODA, and temporal plausibility checks are stretch work.
- Final project should prioritize a defensible safety-monitor story over maximizing detector accuracy.
