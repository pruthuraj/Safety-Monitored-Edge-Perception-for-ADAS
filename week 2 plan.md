# Week 2 Plan: Evidence-First Baseline Detector

## Summary
Week 2 should prioritize a reproducible KITTI baseline over forcing TensorRT before the environment is ready. The machine has an RTX 3050 Ti and CUDA-enabled PyTorch, but `ultralytics`, `onnx`, `onnxruntime`, and `tensorrt` are not installed yet. TensorRT FP16/INT8 remains a Week 2 stretch target after the PyTorch baseline is complete.

Success means: KITTI is acquired/validated, deterministic splits are committed, YOLOv8n trains/evaluates reproducibly, baseline metrics are saved, and every result is documented in the experiment log.

## Key Changes
- Clean repo hygiene first:
  - Keep `.claude-flow/` and `.claude/` ignored.
  - Remove `CLAUDE.md` from `.gitignore`; it is an intentional tracked project file.
  - Do not commit datasets, model weights, TensorRT engines, or generated media.

- Add environment and dependency setup:
  - Create a Python environment spec for Week 2 with `torch`, `ultralytics`, `opencv-python`, `numpy`, `pandas`, `pyyaml`, `scikit-learn`, `matplotlib`, and `tqdm`.
  - Add optional export dependencies separately: `onnx`, `onnxruntime-gpu`, and TensorRT only if install/export succeeds.
  - Record exact Python, PyTorch, CUDA, GPU, driver, and package versions in the experiment log.

- Add KITTI dataset pipeline:
  - Expect raw data under `data/raw/kitti/`, ignored by git.
  - Add a validator that checks expected KITTI image/label folders, class names, counts, and missing labels.
  - Generate deterministic split files under `configs/splits/` using seed `42`.
  - Use 70/15/15 split of the 7,481 labeled KITTI images unless dataset validation shows missing/corrupt samples.

- Add YOLO conversion and training pipeline:
  - Convert KITTI labels to YOLO format for three classes: `pedestrian`, `vehicle`, `cyclist`.
  - Generate `configs/kitti_yolov8.yaml`.
  - Train YOLOv8n only; YOLOv8s remains out of scope for Week 2.
  - Use `imgsz=640`, fixed seed `42`, and a GPU-safe batch size discovered by a short dry run.
  - Save trained weights under ignored model-output paths, not as tracked files.

- Add baseline metrics and reporting:
  - Write `results/baseline_metrics.csv` with PyTorch validation metrics: mAP50, mAP50-95, precision, recall, per-class AP, FPS, latency p50/p95, model path, split hash, command, seed, and hardware.
  - Update `docs/dataset_splits.md` with exact split counts, hashes, KITTI version/download date, preprocessing, excluded frames, and INT8 calibration-set rule.
  - Add `EXP-002` for dataset/split creation and `EXP-003` for baseline training/evaluation.

- TensorRT export policy:
  - After PyTorch baseline passes, attempt ONNX export first.
  - Attempt TensorRT FP16 only if TensorRT tooling is installed cleanly.
  - Attempt INT8 only if FP16 works and a calibration subset from `kitti-train` is defined.
  - If TensorRT fails, document it as a setup blocker instead of spending the week debugging export. Ultralytics supports ONNX/TensorRT export paths, but local TensorRT is currently absent: [Ultralytics export docs](https://docs.ultralytics.com/modes/export/) and [TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/).

## Test Plan
- Dataset validation:
  - Fails if expected KITTI image/label folders are missing.
  - Fails if labels contain unmapped classes without explicit exclusion.
  - Fails if generated train/val/test split overlap exists.

- Label conversion:
  - Verify YOLO labels are normalized to `[0,1]`.
  - Verify excluded KITTI classes are not silently mapped.
  - Verify at least one sample per target class exists in train/val/test where possible.

- Training/evaluation:
  - Run a small smoke train first on a tiny subset to validate the pipeline.
  - Run full YOLOv8n training only after smoke test passes.
  - Confirm `results/baseline_metrics.csv` is produced from script output, not manually typed.

- Export:
  - ONNX export must load successfully if attempted.
  - TensorRT FP16/INT8 metrics are optional for Week 2 and must be clearly marked as unavailable if tooling fails.

- Documentation audit:
  - Every metric in README or docs must point to an experiment-log entry and result file.
  - `safety/traceability_matrix.csv` should keep SR-06 open but reference the baseline latency evidence once produced.

## Assumptions
- Week 2 target is **Evidence-first**, not strict TensorRT.
- KITTI raw data is allowed to live locally under `data/raw/kitti/` and remain untracked.
- No BDD100K work is required this week.
- No monitor, calibration, OOD scoring, or gating implementation is required this week.
- TensorRT is stretch because it is not installed in the current Python environment.
