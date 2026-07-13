# Project Specification — Safety-Monitored Edge Perception for ADAS

Status: Week 1 deliverable. Living document; refined as weeks progress.
Authoritative plan: `../PLAN.md`.

## 1. Item Definition

**Item:** Camera-only forward perception function providing pedestrian, vehicle, and cyclist detections to an AEB-style ADAS planner.

- **Function:** Detect and localize (2D bounding boxes) pedestrians, vehicles, and cyclists in the forward camera view, at real-time rates, on edge hardware.
- **Consumers:** Downstream planner / AEB logic (simulated boundary in this project — no real actuator).
- **Out of scope:** Tracking, fusion (radar/lidar), trajectory prediction, actuation, driver monitoring.
- **Safety concept:** The detector is *supervised by a runtime monitor* (calibrated confidence + OOD score) that gates its output. When evidence quality degrades, the system transitions `NOMINAL → DEGRADED → FAIL_SAFE_REQUEST` rather than emitting untrustworthy detections.

## 2. Operational Design Domain (ODD)

The perception function is claimed valid **only** within:

| Dimension | In ODD | Out of ODD |
|---|---|---|
| Illumination | Daylight | Night, dawn/dusk, tunnel transitions |
| Weather | Clear / lightly overcast | Rain, fog, snow, heavy glare |
| Camera | Single forward-facing, KITTI-like mounting and FOV | Side/rear cameras, fisheye, unusual mounting |
| Scene | KITTI-style road scenes (urban, suburban, highway) | Off-road, parking structures, unpaved |
| Classes | Pedestrian, vehicle (car/van/truck), cyclist | All other classes (animals, debris, trains…) |
| Image quality | Nominal exposure, clean lens | Severe blur, dead pixels, exposure drift, occluded lens |

Out-of-ODD conditions are exactly what the OOD monitor targets (Week 4: BDD100K night/rain/fog slices; Week 7: synthetic corruptions).

## 3. Assumptions

- **A-01** KITTI train/val distribution is representative of the in-ODD operating conditions.
- **A-02** Camera intrinsics/mounting match KITTI setup; no online recalibration needed.
- **A-03** Downstream planner honors `DEGRADED` and `FAIL_SAFE_REQUEST` states (boundary assumption — planner is out of scope).
- **A-04** Frame-level latency budget for perception + monitor is 40 ms (25 FPS floor).
- **A-05** Edge target is RTX 3050 Ti (laptop); Jetson figures, if reported, are separate and optional.
- **A-06** Single-frame perception: no temporal smoothing of detections in MVP (temporal plausibility is stretch).

## 4. Hazards (initial, refined in Week 3 HARA-lite)

| ID | Hazard | Consequence sketch |
|---|---|---|
| H-01 | Undetected pedestrian/vehicle/cyclist in ego path | AEB does not trigger → collision |
| H-02 | Late detection (latency > budget) | AEB triggers too late → reduced mitigation |
| H-03 | False detection (phantom object) | Spurious AEB braking → rear-end collision risk |
| H-04 | Confidently wrong output outside ODD (night/fog/corruption) | Planner trusts invalid perception → H-01/H-03 with no warning |
| H-05 | Monitor failure: missed degradation | System stays `NOMINAL` when it should not → silent H-04 |
| H-06 | Monitor failure: excessive false alarms | Availability loss; driver/planner ignores warnings (cry-wolf) |

H-01 within ODD and H-04 at ODD boundary are the primary hazards this project's monitor addresses.

## 5. Safety Requirement Placeholders

Initial `SR-xx` IDs. Formal derivation, wording, and acceptance criteria come from Week 3 STPA/HARA. Tracked in `../safety/requirements.csv`.

| ID | Area (placeholder) |
|---|---|
| SR-01 | Detection confidence shall be calibrated (post-hoc temperature scaling; ECE reported) |
| SR-02 | Runtime OOD monitoring shall flag inputs outside the ODD |
| SR-03 | System shall enter `DEGRADED` on sustained monitor-score threshold breach (Q95) |
| SR-04 | System shall issue `FAIL_SAFE_REQUEST` on sustained severe breach (Q99) |
| SR-05 | Monitor scores, states, and transitions shall be logged per frame |
| SR-06 | End-to-end perception + monitor latency shall meet the 40 ms budget (p95) |

## 6. Success Criteria (MVP)

- YOLOv8n on KITTI with reproducible split (seed 42), TensorRT FP16 + INT8 engines, quantization delta reported.
- ECE before/after temperature scaling; energy-score OOD with AUROC / FPR@95 on BDD100K shift slices.
- Working 3-state gating demo with overlay; latency table vs 40 ms budget.
- Complete traceability: every SR-xx → implementation → test → metric → result → evidence file.
- Safety case (GSN + SOTIF argument + ISO/PAS 8800 alignment table) with claims limited to the defined ODD.

## 7. Limitations (known at Week 1)

- Single dataset family for ID (KITTI); ODD claims limited accordingly.
- No real vehicle, actuator, or closed-loop evaluation — fail-safe is a *request*, honored by assumption A-03.
- 2D detection only; no distance/TTC estimation, so "in ego path" is approximated in HARA scenarios.
- Monitor thresholds derived from validation data; generalization beyond tested slices is not claimed.
