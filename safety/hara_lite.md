# HARA-Lite — Safety-Monitored Edge Perception for ADAS

Status: Week 3 deliverable (EXP-005). Companion documents: `stpa_report.md`, `control_structure.mmd`, `requirements.csv`, `traceability_matrix.csv`.

> **Disclaimer.** This is an educational HARA-lite exercise, not a certification artifact. It is not a full ISO 26262 HARA, not a complete vehicle-level safety analysis, and no claim of certified or proven safety is made. Methods are *aligned with* ISO 26262 Part 3 concepts; ratings are illustrative and limited to the defined ODD. The item is a perception function evaluated offline and in a simulated runtime — there is no real vehicle, actuator, or closed-loop validation.

## 1. Item Boundary

**Item:** Camera-only forward perception function providing pedestrian, vehicle, and cyclist 2D detections to an AEB-style planner, supervised by a runtime monitor (calibrated confidence + OOD score) that gates its output via `NOMINAL` / `DEGRADED` / `FAIL_SAFE_REQUEST` states.

Inside the item boundary:

- Forward camera input (KITTI-like mounting, single frame).
- YOLOv8n detector (PyTorch and TensorRT backends).
- Runtime monitor: temperature-scaled confidence, energy-style OOD score, threshold logic.
- State machine emitting `NOMINAL` / `DEGRADED` / `FAIL_SAFE_REQUEST`.
- Per-frame machine-readable logging.

Outside the item boundary (interfaces only; behavior assumed per `docs/project_spec.md` A-01..A-06):

- Planner / AEB decision logic (assumed to honor monitor states — A-03).
- Brake actuator and vehicle dynamics.
- Driver and road environment.
- Camera hardware faults (dealt with only as image-quality degradation visible to the OOD monitor).

## 2. Operating Scenario for the Worked Example

Urban daytime driving within the ODD (`docs/project_spec.md` §2): clear/lightly overcast daylight, KITTI-style urban road, ego speed typical of urban traffic (~30–50 km/h), pedestrian entering or present in the ego path, AEB function relying on the perception item as its only obstacle sensor (camera-only configuration).

## 3. Hazard Table

Hazards carried over from Week 1 (`docs/project_spec.md` §4) and refined here. "Malfunctioning behavior" is stated at the perception-output level; the vehicle-level hazardous event arises through the planner/AEB boundary.

| ID | Malfunctioning behavior (item level) | Vehicle-level hazardous event | Worst-case consequence | Primary driver of risk |
|---|---|---|---|---|
| H-01 | Pedestrian/vehicle/cyclist in ego path not detected | AEB does not trigger | Collision with vulnerable road user | Missed detection within ODD |
| H-02 | Detection provided too late (latency > 40 ms budget, A-04) | AEB triggers late; braking distance shortfall | Reduced-mitigation collision | Latency budget violation |
| H-03 | Phantom detection (false positive in ego path) | Spurious hard braking | Rear-end collision by following traffic | False positives at high confidence |
| H-04 | Confidently wrong output outside ODD (night/fog/corruption) with no warning | Planner trusts invalid perception | H-01 or H-03 consequence, with no forewarning | Miscalibration + missing OOD flag |
| H-05 | Monitor misses degradation (stays `NOMINAL` when evidence is untrustworthy) | Silent continuation of H-04 exposure | Same as H-04 | Monitor false negative |
| H-06 | Monitor raises excessive false alarms | Availability loss; planner/driver desensitized (cry-wolf) | Ignored warning during a real H-04 event | Monitor false positive rate |

## 4. Worked ASIL-Style Derivation (H-01 instance)

**Hazardous event:** undetected pedestrian in ego path during urban daytime AEB scenario (within ODD).

| Parameter | Rating | Rationale |
|---|---|---|
| Severity | **S3** | Impact with a pedestrian at urban speed is life-threatening; unmitigated AEB failure removes the intended protection. |
| Exposure | **E4** | Urban daytime driving with pedestrians present is a high-probability operating situation — routinely encountered on nearly every urban drive. |
| Controllability | **C3** | The driver is not warned (the failure is silent within a nominally trusted function); the pedestrian may be unable to evade. Fewer than 90% of drivers can be expected to avert harm. |

**Resulting classification: ASIL D** (S3 + E4 + C3), as an educational HARA-lite classification of the hazardous event, not a certified rating of this project.

**Safety goal SG-01:** the perception function shall either provide trustworthy pedestrian/vehicle/cyclist detections within the defined ODD, or request degraded/fail-safe handling when confidence or domain evidence is insufficient.

SG-01 is the parent of safety requirements `SR-01..SR-06` (`requirements.csv`); the decomposition path from SG-01 through the STPA unsafe control actions is documented in `stpa_report.md` §6.

### Ratings for remaining hazards (summary, not individually worked)

These are coarse, illustrative ratings to prioritize monitor work; only H-01 is worked in full above.

| Hazard | S | E | C | Indicative class | Note |
|---|---|---|---|---|---|
| H-02 | S3 | E4 | C3 | high (ASIL D-like) | Same event chain as H-01 with partial mitigation; treated jointly with H-01 via SR-06. |
| H-03 | S2 | E4 | C2 | mid (ASIL B-like) | Rear-end at urban speed; following driver has some controllability. |
| H-04 | S3 | E3 | C3 | high (ASIL C/D-like) | Out-of-ODD exposure (night/rain/fog) lower than daytime-urban but still regular. |
| H-05 | S3 | E3 | C3 | high | Conditional on H-04; monitor false negative removes the designed mitigation. |
| H-06 | S2 | E4 | C2 | mid | Harm is indirect (cry-wolf → ignored real alarms); bounded via FPR@95 targets in SR-02. |

## 5. Assumptions

Inherited from `docs/project_spec.md` §3 (A-01..A-06) and extended:

- **A-07** The worked scenario assumes the pedestrian is within detector range and nominally visible (not fully occluded); full occlusion is outside what a camera-only item can mitigate and is an accepted limitation.
- **A-08** HARA-lite exposure/controllability judgments are made qualitatively from the scenario description, without field data.
- **A-09** The 40 ms latency budget (A-04) is taken as the boundary between H-02 occurring and not occurring; no finer-grained timing analysis is performed.

## 6. Limitations

- Ratings are illustrative; no S/E/C tables were calibrated against accident statistics or expert panels.
- Single worked derivation (H-01); remaining hazards receive summary ratings only.
- No vehicle-level verification: the planner, actuator, and driver reactions are assumptions (A-03), not analyzed elements.
- 2D detection without distance/TTC means "in ego path" is approximated geometrically in evaluation scenarios.
- HARA-lite covers the KITTI-like ODD only; claims do not extend to unlisted conditions.
