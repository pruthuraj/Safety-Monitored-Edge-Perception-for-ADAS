# Demo Narration Script — `final_demo.mp4`

~72 s, silent by default. Narration below is optional; timings match the card/clip layout in `scripts/build_final_demo.py`. All figures trace to `results/report_summary.json`.

## 0:00–0:05 — Title card
> Safety-monitored edge perception for ADAS: a camera-only detector that also knows when it should not be trusted.

## 0:05–0:49 — Annotated overlay clip (KITTI → BDD night / rain / fog)
> YOLOv8n runs on an RTX 3050 Ti through TensorRT. A runtime monitor scores every frame and drives a three-state machine. On clean KITTI daylight it stays NOMINAL, in green. As the scene shifts to night, rain, and fog — outside the operational design domain — detection confidence collapses and the monitor escalates to DEGRADED and then FAIL_SAFE_REQUEST, in red. Transitions require sustained breaches, so single-frame spikes do not trip it.

## 0:49–0:57 — Key results card
> On the held-out KITTI validation set the detector reaches 0.856 mAP50 in FP16, with the full perception-plus-monitor loop at 17.2 milliseconds p95 — well inside the 40 millisecond budget. The monitor separates night from in-distribution data at 0.982 AUROC, and under severe synthetic fog it flags 99% of frames as detection quality collapses to 0.071 mAP50. Every number traces to a committed result file.

## 0:57–1:07 — Residual-risk card
> The project's most important result is a negative one. Severe low light removes 19% of mAP50 while the monitor stays 97.7% NOMINAL — a frame-level confidence score cannot see silent recall erosion. This is documented as a known-unsafe SOTIF residual rather than hidden, with feature-space and temporal mitigations named as future work.

## 1:07–1:12 — Closing card
> No claim of certification or proven safety — an evidence-backed safety argument, bounded to its ODD.
