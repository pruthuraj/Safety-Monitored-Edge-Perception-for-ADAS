"""export_trt: export baseline YOLOv8n to ONNX (and optionally TensorRT).

Week 2 stretch policy (week 2 plan.md): ONNX first; TensorRT FP16 only if
tooling is clean; INT8 only after FP16 works. On failure, record the blocker
in results/export_summary.json and stop — do not debug indefinitely.

Parity check: PyTorch vs ONNX predictions on N val images (detection count
and confidence deltas). Latency is NOT measured here for ONNX-CPU — latency
claims come only from PyTorch (train_baseline.py) or TensorRT backends.

Usage:
    python scripts/export_trt.py                 # ONNX export + parity check
    python scripts/export_trt.py --trt           # + attempt TensorRT FP16 engine
    python scripts/export_trt.py --weights runs/detect/baseline/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VAL_IMAGES = REPO / "data" / "processed" / "kitti_yolo" / "images" / "val"
SUMMARY = REPO / "results" / "export_summary.json"
IMGSZ = 640
PARITY_N = 20
CONF = 0.25


def parity_check(pt_model, onnx_model, images) -> dict:
    """Compare detections between backends on the same images."""
    count_diffs, conf_diffs = [], []
    for p in images:
        r_pt = pt_model.predict(p, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        r_ox = onnx_model.predict(p, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        n_pt, n_ox = len(r_pt.boxes), len(r_ox.boxes)
        count_diffs.append(abs(n_pt - n_ox))
        n = min(n_pt, n_ox)
        if n:
            c_pt = sorted(r_pt.boxes.conf.tolist(), reverse=True)[:n]
            c_ox = sorted(r_ox.boxes.conf.tolist(), reverse=True)[:n]
            conf_diffs.extend(abs(a - b) for a, b in zip(c_pt, c_ox))
    return {
        "images": len(images),
        "det_count_mismatches": sum(1 for d in count_diffs if d),
        "max_det_count_diff": max(count_diffs),
        "mean_abs_conf_diff": round(sum(conf_diffs) / len(conf_diffs), 6) if conf_diffs else None,
        "max_abs_conf_diff": round(max(conf_diffs), 6) if conf_diffs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(REPO / "runs" / "detect" / "baseline" / "weights" / "best.pt"))
    ap.add_argument("--trt", action="store_true", help="attempt TensorRT FP16 engine after ONNX")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    summary = {"date": date.today().isoformat(), "experiment": "EXP-004",
               "weights": str(weights), "imgsz": IMGSZ, "command": " ".join(sys.argv)}

    # --- ONNX export ---
    model = YOLO(str(weights))
    onnx_path = Path(model.export(format="onnx", imgsz=IMGSZ, dynamic=False))
    summary["onnx"] = {"path": str(onnx_path), "size_mb": round(onnx_path.stat().st_size / 1e6, 2)}

    # --- parity: PyTorch vs ONNX (onnxruntime CPU) ---
    images = sorted(VAL_IMAGES.glob("*.png"))[:PARITY_N]
    summary["parity_pt_vs_onnx"] = parity_check(model, YOLO(str(onnx_path)), images)

    # --- optional TensorRT FP16 ---
    if args.trt:
        try:
            engine_path = Path(YOLO(str(weights)).export(format="engine", imgsz=IMGSZ, half=True))
            summary["tensorrt_fp16"] = {"path": str(engine_path),
                                        "size_mb": round(engine_path.stat().st_size / 1e6, 2)}
        except Exception as e:  # blocker policy: record, don't debug here
            summary["tensorrt_fp16"] = {"blocker": f"{type(e).__name__}: {e}"}

    SUMMARY.parent.mkdir(exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
