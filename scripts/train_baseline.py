"""train_baseline: train/evaluate YOLOv8n on KITTI (Week 2, EXP-003).

Modes:
  --smoke   tiny run (fraction of train, 1 epoch) to validate the pipeline
  (default) full training: imgsz=640, seed=42, AutoBatch (GPU-safe batch)

After training, evaluates on the val split and writes
results/baseline_metrics.csv (script-generated, never hand-typed), including
per-image latency p50/p95 measured over the val split.

Usage:
    python scripts/train_baseline.py --smoke
    python scripts/train_baseline.py --epochs 50
    python scripts/train_baseline.py --eval-only --weights runs/detect/baseline/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA_YAML = REPO / "configs" / "kitti_yolov8.yaml"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "manifest.json"
RESULTS_CSV = REPO / "results" / "baseline_metrics.csv"
SEED = 42


def measure_latency(model, image_dir: Path, imgsz: int, n: int = 300) -> dict:
    """Per-image end-to-end predict latency (ms) over up to n val images."""
    images = sorted(image_dir.glob("*.png"))[:n]
    # warmup
    for p in images[:10]:
        model.predict(p, imgsz=imgsz, verbose=False)
    times = []
    for p in images:
        t0 = time.perf_counter()
        model.predict(p, imgsz=imgsz, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "latency_ms_p50": round(float(np.percentile(arr, 50)), 2),
        "latency_ms_p95": round(float(np.percentile(arr, 95)), 2),
        "fps_mean": round(1000.0 / float(arr.mean()), 1),
        "latency_n_images": len(images),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline-validation run")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--weights", default=None, help="weights for --eval-only (.pt, .onnx, .engine)")
    ap.add_argument("--experiment", default=None, help="experiment id override for the CSV row")
    args = ap.parse_args()

    from ultralytics import YOLO

    run_name = "smoke" if args.smoke else "baseline"
    if not args.eval_only:
        model = YOLO(args.model)
        train_kwargs = dict(
            data=str(DATA_YAML),
            imgsz=args.imgsz,
            seed=SEED,
            deterministic=True,
            batch=-1,  # AutoBatch: discovers GPU-safe batch size
            project=str(REPO / "runs" / "detect"),
            name=run_name,
            exist_ok=True,
        )
        if args.smoke:
            train_kwargs.update(epochs=1, fraction=0.02, batch=8)
        else:
            train_kwargs.update(epochs=args.epochs)
        model.train(**train_kwargs)
        weights = REPO / "runs" / "detect" / run_name / "weights" / "best.pt"
    else:
        weights = Path(args.weights)
        model = YOLO(str(weights))

    # --- evaluate on val split ---
    metrics = model.val(data=str(DATA_YAML), split="val", imgsz=args.imgsz, verbose=False)
    per_class_ap = {
        metrics.names[c]: round(float(m), 4) for c, m in zip(metrics.box.ap_class_index, metrics.box.ap50)
    }

    lat = measure_latency(model, REPO / "data" / "processed" / "kitti_yolo" / "images" / "val", args.imgsz)

    backend = {".engine": "tensorrt_fp16", ".onnx": "onnxruntime"}.get(weights.suffix, "pytorch")
    manifest = json.loads(SPLIT_MANIFEST.read_text())
    row = {
        "date": date.today().isoformat(),
        "experiment": args.experiment or ("EXP-003-smoke" if args.smoke else "EXP-003"),
        "backend": backend,
        "model": str(weights.relative_to(REPO)) if weights.is_absolute() else str(weights),
        "imgsz": args.imgsz,
        "seed": SEED,
        "epochs": "eval-only" if args.eval_only else (1 if args.smoke else args.epochs),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        **{f"AP50_{k}": v for k, v in per_class_ap.items()},
        **lat,
        "split_train_sha256": manifest["sha256"]["train"][:12],
        "split_val_sha256": manifest["sha256"]["val"][:12],
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "torch": torch.__version__,
        "command": " ".join(sys.argv),
    }

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    exists = RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)

    print(json.dumps(row, indent=2))
    print(f"\nwrote {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
