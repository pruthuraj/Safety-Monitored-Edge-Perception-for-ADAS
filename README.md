# Safety-Monitored Edge Perception for ADAS

Camera-only pedestrian/vehicle/cyclist detection (YOLOv8n on KITTI) deployed via TensorRT FP16/INT8, supervised by a calibrated-confidence + OOD runtime monitor that gates output through `NOMINAL → DEGRADED → FAIL_SAFE_REQUEST` states — with STPA/HARA, GSN safety case, and full requirement-to-evidence traceability.

> Status: Week 2 of 10 — baseline detector trained. See [PLAN.md](PLAN.md) for the execution plan and [docs/project_spec.md](docs/project_spec.md) for item definition, ODD, assumptions, and hazards.

## Baseline results (EXP-003)

YOLOv8n, KITTI val split (1122 images), imgsz 640, seed 42, 50 epochs. Source: `results/baseline_metrics.csv`, `docs/experiment_log.md` (EXP-003), command `python scripts/train_baseline.py --epochs 50`.

| Metric | Value |
|---|---|
| mAP50 / mAP50-95 | 0.859 / 0.556 |
| AP50 pedestrian / vehicle / cyclist | 0.749 / 0.948 / 0.880 |
| Latency p50 / p95 (PyTorch, RTX 3050 Ti) | 17.9 ms / 20.6 ms |

Latency is PyTorch-backend inference without the runtime monitor; the 40 ms budget (SR-06) will be re-verified once TensorRT export and monitor are in the loop.

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
