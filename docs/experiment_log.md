# Experiment Log

Append-only. One entry per experiment/change. No result may appear in README/paper/CV/demo without an entry here pointing at its command + evidence file.

## Entry template

```markdown
### EXP-NNN — <short title>
- **Date:** YYYY-MM-DD
- **Week:** N
- **What changed:**
- **Why:**
- **Command(s):**
  ```
  <exact command(s), config file, seed>
  ```
- **Environment:** <hardware, driver/CUDA/TensorRT versions if relevant>
- **Result:** <numbers + path to evidence file, e.g. results/baseline_metrics.csv>
- **Limitation:**
- **Next step:**
```

---

### EXP-001 — Week 1 scaffold
- **Date:** 2026-07-13
- **Week:** 1
- **What changed:** Created repo structure, project spec (item + ODD + assumptions + hazards), SR-xx placeholders, traceability skeleton, dataset split notes.
- **Why:** Week 1 deliverable per PLAN.md.
- **Command(s):** n/a (documentation only)
- **Environment:** n/a
- **Result:** `docs/project_spec.md`, `safety/requirements.csv`, `safety/traceability_matrix.csv`, `docs/dataset_splits.md`
- **Limitation:** SR-xx wording is placeholder until Week 3 STPA/HARA. No data downloaded yet.
- **Next step:** Week 2 — KITTI acquisition, deterministic split, YOLOv8n baseline training.

### EXP-002 — KITTI acquisition, validation, deterministic splits
- **Date:** 2026-07-13
- **Week:** 2
- **What changed:** Downloaded KITTI labels (+images in progress) from official AWS mirror; added dataset validator, split generator, KITTI→YOLO converter (3-class mapping), and pipeline tests (9 passing).
- **Why:** Evidence-first baseline requires validated data and committed, deterministic splits before any training.
- **Command(s):**
  ```
  python -m src.dataset.validate_kitti --root data/raw/kitti --labels-only --out results/kitti_validation.json
  python -m src.dataset.make_splits --root data/raw/kitti --seed 42
  python -m pytest tests/ -q
  ```
- **Environment:** Python 3.11.3, torch 2.7.1+cu118 (CUDA available), ultralytics 8.4.93, RTX 3050 Ti Laptop 4 GB, driver 592.27, Windows 11.
- **Result:** 7481/7481 labels valid, zero unmapped types (`results/kitti_validation.json`). Splits 5237/1122/1122 committed under `configs/splits/` with SHA-256 in `manifest.json`. 9/9 tests pass.
- **Limitation:** Image-dependent checks (image/label ID match, YOLO conversion) pending 12 GB image download completion.
- **Next step:** Extract images → full validation → conversion → smoke train (EXP-003).

### EXP-003 — YOLOv8n KITTI baseline (full 50-epoch train)
- **Date:** 2026-07-14
- **Week:** 2
- **What changed:** Trained YOLOv8n from `yolov8n.pt` on the committed KITTI train split (5237 images), evaluated on the val split (1122 images), measured per-image PyTorch latency. Smoke run (EXP-003-smoke, 2026-07-13) validated the pipeline first; NumPy 2 ABI break fixed beforehand (see requirements.txt note).
- **Why:** Week 2 deliverable — evidence-backed detection baseline before any monitor/TensorRT work.
- **Command(s):**
  ```
  python scripts/train_baseline.py --epochs 50
  # imgsz=640, seed=42, deterministic=True, AutoBatch; splits per configs/splits/manifest.json
  ```
- **Environment:** Python 3.11.3, torch 2.7.1+cu118, ultralytics 8.4.93, RTX 3050 Ti Laptop 4 GB, driver 592.27, Windows 11.
- **Result:** Val split: precision 0.8605, recall 0.7915, **mAP50 0.8588**, mAP50-95 0.5561. Per-class AP50: pedestrian 0.7492, vehicle 0.9476, cyclist 0.8796. Latency (PyTorch, 300 val images): **p50 17.91 ms, p95 20.63 ms**, 54.9 FPS — within the 40 ms budget (SR-06, partial evidence). Evidence: `results/baseline_metrics.csv` (EXP-003 row); weights `runs/detect/baseline/weights/best.pt` (untracked).
- **Limitation:** Latency is PyTorch backend without monitor overhead — SR-06 must be re-verified with TensorRT engine + monitor in the loop. Pedestrian AP50 lowest of three classes (small objects). Test split untouched.
- **Next step:** Stretch: ONNX export → TensorRT FP16 (EXP-004). Week 3: STPA/HARA, finalize SR-xx.

### EXP-004 — ONNX export + TensorRT FP16 engine (Week 2 stretch)
- **Date:** 2026-07-16
- **Week:** 2
- **What changed:** Exported EXP-003 `best.pt` to ONNX (opset 17/19, onnxslim), ran PyTorch-vs-ONNX parity check (20 val images, ORT CPU), built TensorRT FP16 engine (ModelOpt AutoCast, 225/231 nodes fp16), evaluated engine on full val split + latency.
- **Why:** Week 2 stretch per plan policy: ONNX first → FP16 only if tooling clean. Tooling was clean.
- **Command(s):**
  ```
  python scripts/export_trt.py --trt
  python scripts/train_baseline.py --eval-only --weights runs/detect/baseline/weights/best.engine --experiment EXP-004
  ```
- **Environment:** tensorrt 11.1.0.106 (pip), onnx 1.21.0, onnxruntime 1.27.0 (CPU, parity only), torch 2.7.1+cu118, RTX 3050 Ti Laptop 4 GB, driver 592.27. See requirements-export.txt for the torch-clobber warning hit during install.
- **Result:** TRT FP16 val: **mAP50 0.8564** (PyTorch: 0.8588, Δ −0.0024), mAP50-95 0.5584, AP50 ped/veh/cyc 0.7453/0.9473/0.8766. Latency (300 val images, end-to-end predict): **p50 16.90 ms, p95 18.01 ms**, 58.8 FPS; pure engine inference 2.6 ms/img (ultralytics val speed). Engine 8.5 MB vs 12.3 MB ONNX / 5.9 MB pt. Evidence: `results/baseline_metrics.csv` (EXP-004 row), `results/export_summary.json`. Parity PT-vs-ONNX: mean abs conf diff 0.020; det-count mismatches 10/20 attributed to preprocessing difference (PyTorch rect letterbox vs ONNX fixed 640×640), not export corruption.
- **Limitation:** End-to-end latency dominated by Python pre/postprocess, so FP16 gain looks small (p50 17.9→16.9 ms) despite 2.6 ms pure inference; a C++/optimized pipeline would show the real speedup. `pip install tensorrt` replaced pinned cu118 torch with CPU build (restored; documented in requirements-export.txt). INT8 not attempted — needs calibration set from train split (Week 5 per PLAN.md). Monitor overhead still absent from SR-06 evidence.
- **Next step:** Week 3 — STPA/HARA, finalize SR-xx wording. INT8 + calibration in TensorRT week.

### EXP-005 — Week 3 safety analysis and requirements finalization
- **Date:** 2026-07-16
- **Week:** 3
- **What changed:** Wrote `safety/hara_lite.md` (item boundary, H-01..H-06 hazard table, worked S3/E4/C3 → ASIL D derivation for undetected pedestrian, safety goal SG-01, assumptions A-07..A-09) and `safety/stpa_report.md` (losses LS-01..LS-04, control structure, UCA-DET-01..04 and UCA-MON-01..05, causal scenarios CS-01..CS-09, SR derivation table); added `safety/control_structure.mmd`. Finalized `safety/requirements.csv` — SR-01..SR-06 placeholder→active with measurable acceptance criteria and concrete `derived_from` (SG/H/UCA/CS/LS refs). Updated `safety/traceability_matrix.csv` with expected evidence file paths; SR-06 remains `partial` on EXP-003/EXP-004 latency evidence.
- **Why:** Week 3 deliverable per PLAN.md — hazards, unsafe control actions, and finalized safety requirements must exist before Week 4 calibration/OOD implementation, so every monitor feature lands with a requirement to trace to.
- **Command(s):** n/a (safety analysis / documentation only; no training or evaluation runs)
- **Environment:** n/a
- **Result:** `safety/hara_lite.md`, `safety/stpa_report.md`, `safety/control_structure.mmd`, `safety/requirements.csv` (6/6 active), `safety/traceability_matrix.csv` (SR-06 partial, rest open with planned evidence paths). All H-01..H-06 map to ≥1 requirement or explicit boundary assumption; CS-09 (planner ignores states) carried as assumption A-03, not a requirement.
- **Limitation:** HARA-lite ratings are educational/illustrative, not calibrated against field data; only H-01 worked in full. No new runtime evidence — SR-01..SR-05 verification lands Weeks 4-6.
- **Next step:** Week 4 — temperature-scaling calibration (SR-01) and energy-score OOD monitoring with BDD100K slices (SR-02).

### EXP-006 — Confidence calibration via temperature scaling (SR-01)
- **Date:** 2026-07-17
- **Week:** 4
- **What changed:** Added `src/monitor/calibration.py` (ECE, 15-bin reliability, greedy one-to-one IoU>=0.50 same-class GT matching, scalar temperature fit via golden-section on binary NLL) and `scripts/evaluate_monitor.py --calibrate`. kitti-val split 50/50 (seed 42) into calibration-fit (561 imgs) / calibration-report (561 imgs); detections collected at conf>=0.05, imgsz 640. 26 new unit tests in `tests/test_monitor.py`.
- **Why:** SR-01 — detection confidences must be calibrated so monitor thresholds correspond to actual reliability (mitigates CS-02/UCA-DET-03/H-04).
- **Command(s):**
  ```
  python scripts/evaluate_monitor.py --calibrate
  # seed 42, conf_min 0.05, iou_thr 0.50, n_bins 15, weights runs/detect/baseline/weights/best.pt
  ```
- **Environment:** Python 3.11.3, torch 2.7.1+cu118, ultralytics 8.4.93, RTX 3050 Ti Laptop 4 GB, Windows 11.
- **Result:** Fitted **T = 0.6003** (model underconfident — accuracy above diagonal pre-scaling; T<1 sharpens). **ECE 0.0812 → 0.0390** on calibration-report subset (5724 detections; fit on 5423). SR-01 acceptance met (calibrated ECE improves). Evidence: `results/calibration_metrics.csv`, `results/calibration_params.json`, `results/reliability_diagram.png`.
- **Limitation:** Single scalar T over all classes/confidence ranges; per-class calibration not attempted. Correctness defined at IoU 0.50 same-class only. Calibration valid for kitti-val distribution; behavior under shift measured in EXP-007.
- **Next step:** EXP-007 — BDD100K slices + OOD scoring (blocked on manual BDD100K download).

### EXP-007 — BDD100K OOD slices + monitor scoring (SR-02)
- **Date:** 2026-07-17
- **Week:** 4
- **What changed:** Added `src/monitor/scoring.py` (frame-level max-confidence score `1-max(conf)`, post-NMS energy-style score = -logsumexp over top-10 detection-confidence logits — documented proxy, not raw-logit energy or Mahalanobis; rank-based AUROC; FPR@95) and `src/dataset/bdd100k_slices.py` (deterministic slice builder, seed 42, manifest with hashes). Acquired BDD100K 100k val (10,000 images): official mirror `dl.cv.ethz.ch` is offline (NXDOMAIN), so images came from HF `hirundo-io/bdd100k-validation-only` (single zip, official filenames/layout) and frame attributes from HF `dgural/bdd100k` `samples.json` (FiftyOne export of official val labels), converted via new `src/dataset/bdd_fiftyone_convert.py`. Label/image match verified 10000/10000, zero missing/extra.
- **Why:** SR-02 — runtime OOD score flagging out-of-ODD inputs (mitigates CS-01/CS-04/UCA-MON-01/H-04/H-05).
- **Command(s):**
  ```
  python -m src.dataset.bdd_fiftyone_convert --samples <hf-cache>/samples.json --out data/raw/bdd100k/labels/bdd100k_labels_images_val.json
  python -m src.dataset.bdd100k_slices --root data/raw/bdd100k --seed 42
  python scripts/evaluate_monitor.py --ood
  ```
- **Environment:** as EXP-006.
- **Result:** Slices (sampled/available): clear-day 500/1764, night 500/3929, rain 500/738, **fog 13/13**. ID = kitti-val calibration-report subset (561 imgs). AUROC / FPR@95:

  | Slice | max-conf | energy |
  |---|---|---|
  | bdd-night | **0.982 / 0.112** | **0.985 / 0.123** |
  | bdd-rain | 0.876 / 0.537 | 0.864 / 0.606 |
  | bdd-fog (n=13) | 0.926 / 0.419 | 0.890 / 0.592 |
  | bdd-clear-day (transfer control) | 0.758 / 0.624 | 0.715 / 0.727 |

  Score distributions show large no-detection mass on night/rain (score spike at max) — detector fails silently on OOD, which is precisely the H-04 mechanism the monitor flags. SR-02 acceptance met (AUROC+FPR@95 per slice, both methods compared). Evidence: `results/ood_metrics.csv`, `results/monitor_scores_kitti_val.csv`, `results/monitor_scores_bdd100k.csv`, `results/ood_score_distributions.png`, `results/ood_roc_curves.png`.
- **Limitation:** Fog slice n=13 (all available in official val) — metrics unstable, report-only. Clear-day transfer control shows AUROC ~0.72-0.76: KITTI→BDD camera/scene shift is detectable even in-ODD, so thresholds must come from KITTI val quantiles, not BDD separation. Energy score no better than max-conf baseline at this granularity (post-NMS proxy). Images/labels from HF mirrors, not the offline official server — provenance recorded in `docs/dataset_splits.md`.
- **Next step:** Week 5-6 — Q95/Q99 thresholds from kitti-val score quantiles, state machine + gating (SR-03/SR-04), logging (SR-05), monitor latency overhead for SR-06.

### EXP-008 — Monitor thresholds and risk-coverage
- **Date:** 2026-07-17
- **Week:** 5
- **What changed:** Added `src/monitor/thresholds.py` (Q95/Q99 quantile thresholds, coverage), `src/monitor/detection_metrics.py` (standalone VOC-style AP50/AP50-95, micro precision/recall over frame subsets), and `scripts/evaluate_monitor.py --thresholds` / `--risk-coverage`. 10 new tests (`tests/test_thresholds_risk.py`). Weekly plan notes added to `.gitignore`.
- **Why:** Freeze validation-only gating thresholds and quantify risk removed by rejecting suspect frames — the evidence Week 6 SR-03/SR-04 gating builds on.
- **Command(s):**
  ```
  python scripts/evaluate_monitor.py --thresholds
  python scripts/evaluate_monitor.py --risk-coverage
  ```
- **Environment:** as EXP-006.
- **Result:** **Primary monitor score: `max_conf_score`** — energy proxy mean OOD AUROC 0.913 vs 0.928 (gain −0.015, below +0.01 materiality margin; simpler score wins per plan policy). Frozen thresholds (kitti-val report subset, 561 frames, seed 42): max-conf **Q95 = 0.1862, Q99 = 0.3728**; energy Q95 = −1.594, Q99 = −1.149 (`results/monitor_thresholds.json`). Risk-coverage (kitti-val): Q95 keeps 94.8% coverage at mAP50 0.854 (vs 0.850 full), recall ~0.90 flat. BDD coverage at frozen max-conf Q95: **night 6.0%** accepted, rain 36.0%, fog 23.1%, clear-day control 68.0% — monitor rejects the bulk of out-of-ODD frames while keeping ~95% of ID. Evidence: `results/risk_coverage.csv`, `results/risk_coverage.png`.
- **Limitation:** Accepted-frame mAP50 does not rise much at lower coverage — frame-level max-conf ranks frames by easiest detection, not scene difficulty; the risk-removal value is OOD rejection, not ID mAP gain (stated honestly). BDD rows are coverage-only (attribute labels, no detection GT). Thresholds frozen against PyTorch-backend scores; re-check under TensorRT backend when gating is integrated.
- **Next step:** Week 6 — state machine (`NOMINAL`/`DEGRADED`/`FAIL_SAFE_REQUEST`) on sustained Q95/Q99 breaches (SR-03/SR-04), per-frame logging (SR-05), monitor latency overhead (SR-06).

### EXP-009 — Runtime gating, logging, latency, and demo
- **Date:** 2026-07-17
- **Week:** 6
- **What changed:** Added `src/monitor/state_machine.py` (3-state gating: 3×Q95→DEGRADED, 2×Q99→FAIL_SAFE_REQUEST, 10 degraded frames→escalate, 5 clean→recover one level; strict `>` breach), `src/monitor/runtime.py` (RuntimeMonitor: predict + max_conf_score + frozen thresholds from EXP-008 + state machine + SR-05 log row; TRT engine default, PyTorch fallback), `scripts/run_demo.py` (gating scenario replay → gating_tests.csv, 300-frame latency+log run, annotated demo video). 15 new tests (`tests/test_state_machine.py`) incl. GPU integration (BDD-night must trip monitor). Fixed relative-weights path bug in RuntimeMonitor (resolve()).
- **Why:** SR-03/SR-04 (sustained-breach gating, mitigates UCA-MON-01/02/04), SR-05 (evidence logging, CS-06/CS-07), SR-06 (monitor overhead in latency budget, CS-08).
- **Command(s):**
  ```
  python scripts/run_demo.py
  python scripts/run_demo.py --latency-only --weights runs/detect/baseline/weights/best.pt
  python -m pytest tests/ -q
  ```
- **Environment:** as EXP-006; TensorRT 11.1.0.106 engine from EXP-004.
- **Result:** Gating scenario replay **7/7 pass** (`results/gating_tests.csv`). 300-frame KITTI-val run: all NOMINAL, zero false transitions; log completeness **15/15 checks pass** (`results/runtime_monitor_log.csv`, `results/monitor_log_check.csv`). Latency incl. monitor+logging (`results/monitor_latency_metrics.csv`): **TRT FP16 p50 14.61 / p95 17.20 ms (67.9 FPS)**; **PyTorch p50 15.68 / p95 17.50 ms (63.1 FPS)** — both under the 40 ms budget; monitor overhead vs EXP-003/004 detector-only figures is negligible (within run-to-run variance). Demo `demo/monitor_overlay.mp4` (133 frames: KITTI→night→rain→fog): NOMINAL on KITTI, FAIL_SAFE_REQUEST on night (score 1.0, zero detections) — stills verified. Full suite **60 passed**.
- **Limitation:** Demo is illustrative only, not a metric source. Gating policy constants (3/2/10/5) are engineering choices, not tuned on data; sensitivity unexplored. KITTI-val run produced no DEGRADED episodes (expected on ID data) — negative evidence only; OOD transition behavior evidenced via BDD-night integration test and demo.
- **Next step:** Week 7 — synthetic corruptions of kitti-test slices, fault injection vs monitor response; Week 8 — safety case (GSN/SOTIF) consuming this evidence.

### EXP-010 — Fault injection and SR verification
- **Date:** 2026-07-17
- **Week:** 7
- **What changed:** Added `src/dataset/corruptions.py` (fog, motion_blur, gaussian_noise, low_light, dead_pixels × low/medium/high; deterministic from frame-id+corruption+severity; OpenCV/NumPy only) and `scripts/run_fault_injection.py` (300-frame seeded kitti-test subset — report-only, thresholds frozen — run clean + 15 corruption conditions through the runtime monitor; in-memory corruption, nothing written to git). RuntimeMonitor accepts ndarray frames. 28 new tests incl. smoke integration. Wrote `safety/verification_report.md` (SR-by-SR narrative).
- **Why:** SOTIF-style robustness evidence: does the monitor actually flag conditions that destroy detection quality (H-04/H-05), and where does it stay silent?
- **Command(s):**
  ```
  python scripts/run_fault_injection.py
  python -m pytest tests/ -q
  ```
- **Environment:** as EXP-006; TensorRT FP16 backend.
- **Result:** Clean subset mAP50 0.8471, 98.3% NOMINAL. Monitor tracks severe degradation: **fog:high mAP50 0.071 → 99% flagged non-NOMINAL**; blur:high 0.251 → 78% FSR; noise:high 0.365 → 77% FSR. **Blind spot found and documented: low_light:high mAP50 0.689 (−19%) with monitor 97.7% NOMINAL**; dead_pixels:medium similar — frame-level max-confidence misses silent recall erosion. All 16 conditions within 40 ms p95 budget (max 31.2 ms). Evidence: `results/fault_injection_metrics.csv`, `results/fault_injection_monitor_log.csv` (4800 rows), `results/fault_injection_summary.json`, `results/fault_injection_curves.png`. SR-01..SR-06 all **verified** per `safety/verification_report.md` with explicit limitations.
- **Limitation:** Corruptions are plausibility models, not physics-validated (fog = uniform haze). Low-light blind spot is a documented SOTIF unknown-unsafe residual; candidate mitigations (detection-count plausibility, temporal checks, Mahalanobis) out of MVP scope. Standalone AP implementation not comparable to ultralytics numbers across tools.
- **Next step:** Week 8 — GSN safety case + SOTIF argument consuming EXP-006..010 evidence; ISO/PAS 8800 alignment table.

### EXP-011 — Safety case, SOTIF argument, ISO/PAS 8800 mapping
- **Date:** 2026-07-17
- **Week:** 8
- **What changed:** Wrote `safety/safety_case.md` (bounded top claim G1, five sub-goals, residual-risk register), `safety/gsn.mmd` + rendered `safety/gsn.svg` (GSN one-pager with context/strategy/solution/residual nodes), `safety/sotif_argument.md` (known/unknown-safe/unsafe quadrants; two triggering-condition campaigns; low-light residual moved unknown→known-unsafe), `safety/iso_pas_8800_mapping.md` (11 lifecycle themes, alignment-not-compliance, explicit gaps column), `safety/evidence_index.csv` (24 claim→evidence→experiment→SR rows).
- **Why:** Week 8 deliverable — package Weeks 1-7 evidence into a coherent, honestly-bounded safety argument.
- **Command(s):** n/a (documentation; validation script checked evidence paths, SR coverage, forbidden-claim absence, low-light presence; Mermaid rendered via preview tool)
- **Environment:** n/a
- **Result:** All 24 evidence paths resolve; SR-01..06 each link to evidence; no unqualified "certified/ISO-compliant/proven safe" claims; low-light residual present in safety case, SOTIF argument, and mapping gaps; 88 tests still pass.
- **Limitation:** Educational-depth safety case; no independent assessment; ISO/PAS 8800 mapped by public-domain theme names, standard text not licensed.
- **Next step:** Week 9-10 — paper/report assets (`build_report_assets`), README final tables, CV/portfolio material.

### EXP-012 — Report assets, paper draft, portfolio README
- **Date:** 2026-07-17
- **Week:** 9
- **What changed:** Added `scripts/build_report_assets.py` (reads results CSVs/JSON only → `results/report_summary.json` with per-number source files, `paper/tables.md` with Tables 1-6, `demo/monitor_overlay.gif` subsampled from the demo mp4; verifies figure inventory). Wrote `paper/main.md` — IEEE-style draft (abstract, intro, related work, system, monitor design, six-part evaluation, safety case/SOTIF, limitations, conclusion, 12 refs). Rewrote `README.md` portfolio-grade: architecture Mermaid, demo GIF, GSN figure, key-metrics tables, full reproduction command chain, limitations.
- **Why:** Week 9 deliverable per PLAN.md; also implements the `build_report_assets` standard entry point.
- **Command(s):**
  ```
  python scripts/build_report_assets.py
  python -m pytest tests/ -q
  ```
- **Environment:** n/a (documentation + asset generation; no training/inference)
- **Result:** 14-point claims audit: every headline number in README/paper matches `report_summary.json` (which records its source CSV); no unqualified certified/ISO-compliant/proven-safe phrasing; all referenced files exist; 88 tests pass.
- **Limitation:** Paper is a markdown draft — IEEE two-column PDF (`paper/main.pdf`) is Week 10 packaging. GIF is 4.9 MiB (README load weight). Claims audit is regex-based spot-check of headline numbers, not exhaustive.
- **Next step:** Week 10 — 60-90 s demo cut, paper PDF, final environment freeze, CV bullet from real values.
