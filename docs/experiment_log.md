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
