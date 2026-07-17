# Safety-Monitored Edge Perception for ADAS

Camera-only pedestrian/vehicle/cyclist detection (YOLOv8n on KITTI) deployed via TensorRT FP16/INT8, supervised by a calibrated-confidence + OOD runtime monitor that gates output through `NOMINAL → DEGRADED → FAIL_SAFE_REQUEST` states — with STPA/HARA, GSN safety case, and full requirement-to-evidence traceability.

> Status: Week 4 of 10 — monitor evidence MVP: temperature calibration ECE 0.081→0.039 (EXP-006), OOD scoring vs BDD100K slices, night AUROC 0.98 (EXP-007). See [PLAN.md](PLAN.md) for the execution plan and [docs/project_spec.md](docs/project_spec.md) for item definition, ODD, assumptions, and hazards.

## Baseline results (EXP-003)

YOLOv8n, KITTI val split (1122 images), imgsz 640, seed 42, 50 epochs. Source: `results/baseline_metrics.csv`, `docs/experiment_log.md` (EXP-003), command `python scripts/train_baseline.py --epochs 50`.

| Metric | PyTorch (EXP-003) | TensorRT FP16 (EXP-004) |
|---|---|---|
| mAP50 / mAP50-95 | 0.859 / 0.556 | 0.856 / 0.558 |
| AP50 pedestrian / vehicle / cyclist | 0.749 / 0.948 / 0.880 | 0.745 / 0.947 / 0.877 |
| Latency p50 / p95 (RTX 3050 Ti) | 17.9 ms / 20.6 ms | 16.9 ms / 18.0 ms |

Latency is end-to-end single-image predict without the runtime monitor; the 40 ms budget (SR-06) will be re-verified with the monitor in the loop. TensorRT FP16 export: `python scripts/export_trt.py --trt` (`results/export_summary.json`).

## Repository layout

```
configs/    experiment configs, dataset split file lists (code-free)
docs/       project spec, experiment log, dataset split notes
src/        dataset | inference | monitor | gating | metrics | viz modules
scripts/    entry points: train_baseline, export_trt, evaluate_monitor, run_demo, build_report_assets
safety/     requirements.csv, traceability_matrix.csv, STPA/HARA, SOTIF, GSN
results/    machine-readable metrics (CSV/JSON) + plots
paper/      IEEE-style paper
demo/       monitor-overlay demo video
```

## Safety claim discipline

This project does **not** claim certification or ISO compliance. All claims are scoped to the defined ODD, phrased as "aligned with" / "argued using" / "monitors for", and backed by evidence files or listed as limitations.
