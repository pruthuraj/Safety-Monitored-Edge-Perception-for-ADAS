"""Runtime monitor wrapper (SR-03..SR-06, Week 6, EXP-009).

Wraps YOLO inference with the frozen monitor: per frame it computes
`max_conf_score`, applies the frozen Q95/Q99 thresholds (loaded from
results/monitor_thresholds.json — never recomputed here), advances the
gating state machine, and emits a machine-readable log row (SR-05).
Latency covers predict + score + state update + log-row construction.

Default backend: TensorRT FP16 engine if present, else PyTorch best.pt.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.monitor.scoring import max_conf_score
from src.monitor.state_machine import MonitorStateMachine, Transition

REPO = Path(__file__).resolve().parents[2]
THRESHOLDS_JSON = REPO / "results" / "monitor_thresholds.json"
ENGINE = REPO / "runs" / "detect" / "baseline" / "weights" / "best.engine"
PT_WEIGHTS = REPO / "runs" / "detect" / "baseline" / "weights" / "best.pt"

LOG_COLUMNS = [
    "frame_id",
    "image_path",
    "timestamp",
    "backend",
    "n_detections",
    "max_confidence",
    "monitor_score",
    "q95_threshold",
    "q99_threshold",
    "state_before",
    "state_after",
    "transition_reason",
    "latency_ms",
]


def default_weights() -> Path:
    return ENGINE if ENGINE.exists() else PT_WEIGHTS


def load_frozen_thresholds(method: str | None = None) -> tuple[float, float, str]:
    """(q95, q99, method) from the frozen EXP-008 artifact."""
    frozen = json.loads(THRESHOLDS_JSON.read_text())
    method = method or frozen["primary_method"]
    t = frozen["thresholds"][method]
    return float(t["q95"]), float(t["q99"]), method


@dataclass
class FrameResult:
    row: dict
    transition: Transition
    boxes: object
    classes: object
    confs: object


class RuntimeMonitor:
    """Frozen-threshold gating monitor around a YOLO model."""

    def __init__(self, weights: Path | None = None, imgsz: int = 640, conf_min: float = 0.05):
        from ultralytics import YOLO

        self.weights = (Path(weights) if weights else default_weights()).resolve()
        self.backend = {".engine": "tensorrt_fp16", ".onnx": "onnxruntime"}.get(
            self.weights.suffix, "pytorch"
        )
        self.model = YOLO(str(self.weights))
        self.imgsz = imgsz
        self.conf_min = conf_min
        self.q95, self.q99, self.method = load_frozen_thresholds()
        self.machine = MonitorStateMachine(self.q95, self.q99)
        self._frame_no = 0

    def process(self, image_path: Path) -> FrameResult:
        """One frame through predict -> score -> state machine -> log row."""
        t0 = time.perf_counter()
        r = self.model.predict(image_path, imgsz=self.imgsz, conf=self.conf_min, verbose=False)[0]
        confs = r.boxes.conf.cpu().numpy()
        score = max_conf_score(confs)
        transition = self.machine.step(score)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self._frame_no += 1
        row = {
            "frame_id": self._frame_no,
            "image_path": str(image_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": self.backend,
            "n_detections": int(len(confs)),
            "max_confidence": round(float(confs.max()), 6) if len(confs) else "",
            "monitor_score": round(float(score), 6),
            "q95_threshold": self.q95,
            "q99_threshold": self.q99,
            "state_before": transition.state_before,
            "state_after": transition.state_after,
            "transition_reason": transition.reason,
            "latency_ms": round(latency_ms, 2),
        }
        return FrameResult(
            row=row,
            transition=transition,
            boxes=r.boxes.xyxy.cpu().numpy(),
            classes=r.boxes.cls.cpu().numpy().astype(int),
            confs=confs,
        )
