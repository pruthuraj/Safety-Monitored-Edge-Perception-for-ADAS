# Final Reproduction Guide

Frozen at Week 10 (EXP-013). Reproduces every result in the README, paper, and safety case from raw data. All metrics come from scripts (never notebooks); seed 42 throughout.

## 1. Environment

| | |
|---|---|
| OS | Windows 11 (26100) |
| Python | 3.11.3 |
| GPU | NVIDIA RTX 3050 Ti Laptop (4 GB), driver 592.27, CUDA 11.8 |
| Torch | 2.7.1+cu118, torchvision 0.22.1+cu118 (cu118 index — see note) |
| Key libs | ultralytics ≥8.3, numpy 2.4.4, opencv-python 4.13.0.92, matplotlib 3.11.0, scikit-learn 1.8.0 |
| TensorRT | 11.1.0.106 (pip), onnx 1.21.0 (see `requirements-export.txt`) |
| Packaging | reportlab ≥4.2,<5 (paper PDF only) |

```bash
pip install -r requirements.txt          # core
pip install -r requirements-export.txt   # TensorRT export (read its torch-clobber warning first)
```

**Dependency warnings (real, documented):**
- `pip install torch ...` must use the cu118 index or pip swaps in CPU wheels: `pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118`.
- Installing `tensorrt` via pip replaced pinned cu118 torch with a CPU build during EXP-004; restore torch afterward (noted in `requirements-export.txt`).
- numpy 2.x ABI: pandas/matplotlib were upgraded (2026-07-13) because their NumPy-1.x builds broke under numpy 2.4.4.

## 2. Data expectations (not committed — datasets are gitignored)

| Dataset | Local path | Source |
|---|---|---|
| KITTI object detection | `data/raw/kitti/{training,testing}` | official AWS mirror (`s3.eu-central-1.amazonaws.com/avg-kitti/`) |
| BDD100K 100k val (10k imgs) | `data/raw/bdd100k/images/100k/val/` | HF `hirundo-io/bdd100k-validation-only` (official mirror offline) |
| BDD100K attributes | `data/raw/bdd100k/labels/bdd100k_labels_images_val.json` | HF `dgural/bdd100k` `samples.json` → `src/dataset/bdd_fiftyone_convert.py` |

Full provenance incl. mirror substitution: `docs/dataset_splits.md`.

## 3. Pipeline (in order)

```bash
# --- data prep ---
python -m src.dataset.validate_kitti --root data/raw/kitti --out results/kitti_validation.json
python -m src.dataset.make_splits --root data/raw/kitti --seed 42      # splits committed; do NOT regenerate silently
python -m src.dataset.kitti_to_yolo                                    # -> data/processed/kitti_yolo

# --- baseline + export (EXP-003 / EXP-004) ---
python scripts/train_baseline.py --epochs 50
python scripts/export_trt.py --trt
python scripts/train_baseline.py --eval-only --weights runs/detect/baseline/weights/best.engine --experiment EXP-004

# --- monitor evidence (EXP-006 / 007 / 008) ---
python -m src.dataset.bdd_fiftyone_convert --samples <hf_cache>/samples.json --out data/raw/bdd100k/labels/bdd100k_labels_images_val.json
python scripts/evaluate_monitor.py            # calibrate, bdd-slices, ood, thresholds, risk-coverage

# --- runtime + verification (EXP-009 / 010) ---
python scripts/run_demo.py                    # gating tests, latency+log, overlay video
python scripts/run_demo.py --latency-only --weights runs/detect/baseline/weights/best.pt
python scripts/run_fault_injection.py

# --- packaging (EXP-012 / 013) ---
python scripts/build_report_assets.py         # report_summary.json, tables.md, GIF
python scripts/build_paper_pdf.py             # paper/main.pdf
python scripts/build_final_demo.py            # demo/final_demo.mp4
python scripts/audit_final_package.py         # final consistency gate
python -m pytest tests/ -q                    # 88 tests
```

## 4. Expected outputs

| Stage | Evidence file(s) | Headline |
|---|---|---|
| Baseline | `results/baseline_metrics.csv` | val mAP50 0.8588; TRT FP16 0.8564 |
| Calibration | `results/calibration_metrics.csv` | ECE 0.0812 → 0.0390, T=0.600 |
| OOD | `results/ood_metrics.csv` | night AUROC 0.982 |
| Thresholds | `results/monitor_thresholds.json` | Q95 0.1862 / Q99 0.3728 (max-conf) |
| Risk-coverage | `results/risk_coverage.csv` | 94.8% ID coverage / 6% night at Q95 |
| Gating + latency | `results/gating_tests.csv`, `results/monitor_latency_metrics.csv` | 7/7 scenarios; p95 17.2 ms TRT |
| Fault injection | `results/fault_injection_metrics.csv` | fog:high 99% flagged; low-light blind spot |
| Verification | `safety/verification_report.md` | SR-01..06 verified |
| Packaging | `paper/main.pdf`, `demo/final_demo.mp4` | 4-page PDF; ~72 s demo |

## 5. Notes

- Thresholds are frozen from kitti-val only (EXP-008); kitti-test is report-only and first evaluated in Week 7.
- GPU nondeterminism: mAP/latency may vary in the last decimal across runs and driver versions; headline figures are stable to the reported precision.
- Weights (`*.pt`, `*.engine`) and datasets are gitignored; retrain/re-export to regenerate.
