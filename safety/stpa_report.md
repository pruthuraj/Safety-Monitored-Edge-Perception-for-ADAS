# STPA Report — Safety-Monitored Edge Perception for ADAS

Status: Week 3 deliverable (EXP-005). Companion documents: `hara_lite.md`, `control_structure.mmd`, `requirements.csv`, `traceability_matrix.csv`.

> **Disclaimer.** This is an educational STPA-style analysis aligned with the STPA Handbook method, scoped to the perception item and its simulated planner/actuator boundary. It is not a complete vehicle-level STPA and makes no certification claim. Elements outside the item boundary (planner internals, actuator, driver) are analyzed only as interface assumptions.

## 1. Losses

| ID | Loss |
|---|---|
| LS-01 | Loss of life or injury to road users (collision involving pedestrian, cyclist, or vehicle occupants). |
| LS-02 | Collision damage or injury caused by inappropriate braking (spurious AEB → rear-end impact). |
| LS-03 | Loss of the designed mitigation: system fails to warn or request fallback when perception evidence is untrustworthy. |
| LS-04 | Loss of valid safety evidence: logs/metrics misrepresent runtime behavior, making the safety argument misleading. |

## 2. System-Level Hazards

Hazards H-01..H-06 are defined in `hara_lite.md` §3 (originating in `docs/project_spec.md` §4). Mapping to losses:

| Hazard | Leads to loss |
|---|---|
| H-01 undetected object in ego path | LS-01 |
| H-02 detection too late (latency > budget) | LS-01 |
| H-03 phantom detection | LS-02 |
| H-04 confidently wrong output outside ODD | LS-01, LS-02, LS-03 |
| H-05 monitor misses degradation | LS-03 (thereby LS-01/LS-02) |
| H-06 monitor false alarms (cry-wolf) | LS-02 (spurious fallback), LS-03 (desensitization) |

## 3. Control Structure

Rendered diagram source: `control_structure.mmd`. Chain: road/objects → camera → YOLOv8n detector → runtime monitor → planner/AEB boundary → brake actuator boundary → road; driver supervises the planner and receives fallback notifications.

Control actions analyzed (perception item outputs only — CA-3/CA-4 belong to the planner and are covered by assumption A-03):

| CA | Controller → controlled process | Action |
|---|---|---|
| CA-1 | Monitor → planner | Gated detections (bounding boxes + calibrated confidences) |
| CA-2 | Monitor → planner | Monitor state (`NOMINAL` / `DEGRADED` / `FAIL_SAFE_REQUEST`) |

Feedback paths: detector raw confidences → monitor; monitor evidence records → per-frame log.

## 4. Unsafe Control Actions (UCAs)

Guide words: **not provided**, **provided incorrectly**, **provided too late**, **stopped too soon**, **continued too long**.

### CA-1: Gated detections

| ID | Guide word | Unsafe control action | Hazard |
|---|---|---|---|
| UCA-DET-01 | Not provided | Detections not provided while an object is in the ego path within the ODD. | H-01 |
| UCA-DET-02 | Provided incorrectly | High-confidence phantom detection provided in the ego path. | H-03 |
| UCA-DET-03 | Provided incorrectly | Detections provided with miscalibrated (overconfident) scores while input is outside the ODD, without any accompanying degradation signal. | H-04 |
| UCA-DET-04 | Provided too late | Detections provided after the 40 ms p95 latency budget in a closing-distance scenario. | H-02 |

### CA-2: Monitor state

| ID | Guide word | Unsafe control action | Hazard |
|---|---|---|---|
| UCA-MON-01 | Not provided | `DEGRADED`/`FAIL_SAFE_REQUEST` not issued while perception evidence is untrustworthy (OOD input or confidence collapse). | H-05 → H-04 |
| UCA-MON-02 | Provided incorrectly | `DEGRADED`/`FAIL_SAFE_REQUEST` issued while perception is operating nominally within the ODD. | H-06 |
| UCA-MON-03 | Provided too late | State transition issued only after prolonged exposure because sustained-breach logic (or latency overhead) delays it beyond the hazard-exposure window. | H-05, H-02 |
| UCA-MON-04 | Stopped too soon | Monitor returns to `NOMINAL` while the degrading condition persists (single good frame resets state). | H-05 |
| UCA-MON-05 | Continued too long | `FAIL_SAFE_REQUEST` maintained long after conditions recover, causing unnecessary unavailability. | H-06 |

## 5. Causal Scenarios

| ID | Scenario | Related UCAs |
|---|---|---|
| CS-01 | OOD input (night/rain/fog/corruption) produces silently degraded detections; energy/confidence signals fail to separate ID from OOD. | UCA-DET-03, UCA-MON-01 |
| CS-02 | Poor calibration: raw confidences overconfident, so thresholds computed on them do not correspond to actual reliability. | UCA-DET-03, UCA-MON-01/02 |
| CS-03 | Threshold misuse: Q95/Q99 thresholds tuned on test slices (leakage) or stale after retraining, shifting false-negative/false-positive balance unpredictably. | UCA-MON-01/02 |
| CS-04 | Monitor false negative: OOD score below threshold for a genuinely degraded input (score distribution overlap). | UCA-MON-01 |
| CS-05 | Monitor false positive: benign in-ODD input (e.g., unusual but valid scene) exceeds threshold; cry-wolf effect accumulates. | UCA-MON-02, UCA-MON-05 |
| CS-06 | Stale state: state machine fails to update per frame (skipped frames, exception in monitor path), planner acts on an outdated state. | UCA-MON-03/04 |
| CS-07 | Logging failure: per-frame records dropped or incomplete, so post-hoc analysis cannot reconstruct monitor behavior; safety evidence misleading. | (LS-04 direct) |
| CS-08 | Latency budget violation: monitor overhead pushes perception+monitor p95 beyond 40 ms; detections and states arrive too late. | UCA-DET-04, UCA-MON-03 |
| CS-09 | Planner ignores `DEGRADED`/`FAIL_SAFE_REQUEST` (violated assumption A-03); mitigation exists but has no effect. | boundary assumption, out of item scope |

CS-09 is retained as an explicit interface risk: it cannot be mitigated inside the item and is carried as assumption A-03 in `docs/project_spec.md` and `hara_lite.md` §1.

## 6. Derived Monitor Requirements

Safety goal SG-01 (`hara_lite.md` §4) decomposes through the UCAs/scenarios above into `SR-01..SR-06` (authoritative wording and acceptance criteria in `requirements.csv`):

| SR | Mitigates | Rationale |
|---|---|---|
| SR-01 Calibrated confidence | CS-02, UCA-DET-03, H-04, LS-04 | Temperature scaling makes confidence values meaningful so thresholds correspond to actual reliability. |
| SR-02 Runtime OOD scoring | CS-01, CS-04, UCA-MON-01, H-04/H-05 | Energy-style score (with max-confidence baseline) detects out-of-ODD inputs; AUROC/FPR@95 bound CS-04/CS-05 rates. |
| SR-03 Degraded-mode entry (sustained Q95) | UCA-MON-01/02/04, CS-05, H-05/H-06 | Sustained-breach criterion filters single-frame spikes (limits UCA-MON-02) while guaranteeing entry under persistent breach (limits UCA-MON-01/04). |
| SR-04 Fail-safe request (sustained Q99) | UCA-MON-01, H-01/H-04, LS-01/LS-03 | Severe sustained breach escalates to `FAIL_SAFE_REQUEST` per SG-01's fallback branch. |
| SR-05 Per-frame logging | CS-06, CS-07, LS-04 | Complete per-frame records make stale state detectable and keep safety evidence reconstructible. |
| SR-06 Latency budget | CS-08, UCA-DET-04, UCA-MON-03, H-02 | Perception + monitor p95 < 40 ms keeps detections and state transitions timely. |

Residual risks not covered by any SR: CS-09 (assumption A-03) and full occlusion (assumption A-07, `hara_lite.md` §5) — both documented as boundary assumptions rather than requirements.
