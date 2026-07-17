# SOTIF Argument — Safety of the Intended Functionality

Status: Week 8 deliverable (EXP-011). Framing aligned with [ISO 21448:2022](https://www.iso.org/standard/77490.html) concepts (alignment, not compliance). Companions: `safety_case.md`, `verification_report.md`.

SOTIF asks: is the *intended* function — camera-only detection gated by a runtime monitor — safe in the presence of performance limitations and triggering conditions, absent component faults? For this project the answer is argued per operating region.

## 1. Intended Functionality

Detect pedestrians/vehicles/cyclists in-ODD and, when perception evidence degrades, transition `NOMINAL → DEGRADED → FAIL_SAFE_REQUEST` rather than emit untrustworthy detections (SG-01, `hara_lite.md` §4).

## 2. Triggering-Condition Exploration

Two exploration campaigns stand in for the SOTIF "identify and evaluate triggering conditions" activity:

| Campaign | Conditions | Nature |
|---|---|---|
| BDD100K slices (EXP-007/008) | night, rain, fog + clear-day transfer control | Real imagery, real ODD exits |
| Synthetic fault injection (EXP-010) | fog, motion blur, gaussian noise, low light, dead pixels × 3 severities | Controlled, deterministic, in-memory |

## 3. Region Classification

Using the SOTIF known/unknown × safe/unsafe quadrants:

### Known-safe (Area 1)
- In-ODD KITTI operation: val mAP50 0.8588; clean kitti-test subset 98.3% NOMINAL with zero false fail-safe episodes (EXP-009/010).
- Latency: full monitor loop p95 ≤ 17.5 ms vs 40 ms budget on all measured conditions.

### Known-unsafe, detected → made safe by monitor (Area 2 → mitigated)
- **Night:** AUROC 0.982; only 6% of night frames accepted at Q95 (EXP-007/008).
- **Severe fog (synthetic):** mAP50 collapses 0.847 → 0.071; monitor flags 99% non-NOMINAL (EXP-010).
- **Severe blur/noise:** ~77–78% FAIL_SAFE_REQUEST as mAP halves or worse (EXP-010).
- Mechanism: detection-confidence collapse is exactly what `max_conf_score` measures; abrupt degradation triggers sustained breaches within 2–3 frames.

### Known-unsafe, UNDETECTED — the documented residual (Area 2, unmitigated)
- **Gradual low-light:** mAP50 0.847 → 0.689 (−19%) at `low_light:high`, monitor 97.7% NOMINAL (EXP-010).
- **Moderate dead pixels:** mAP50 0.791, 98.3% NOMINAL.
- Mechanism of failure: the detector keeps *some* high-confidence detections (near, high-contrast objects) while recall erodes on marginal ones; a frame-level max over confidences is blind to what is *missing*. This was unknown-unsafe until EXP-010 moved it into known-unsafe — the campaign's main SOTIF yield.

### Unknown-unsafe (Area 3, residual by definition)
- Untested real-world conditions: real fog/rain physics, lens contamination, exposure oscillation, adversarial scenes, non-KITTI camera geometries.
- Reduced by, not eliminated by, the two campaigns. Explicitly carried in `safety_case.md` §3.

## 4. Mitigations — Stated as Future Work, Not Claimed

| Residual | Candidate mitigation | Why plausible |
|---|---|---|
| Low-light recall erosion | Detection-count plausibility (expected-object-rate prior) | Missing detections are visible in counts even when max-conf is high |
| Same | Temporal consistency checks (track drop-rate) | Recall erosion manifests as track churn across frames |
| Same + feature drift | Feature-space OOD (Mahalanobis on backbone features) | Sees representation shift before output confidence moves |
| All | Broader real-world validation (real night/rain fleets, HIL) | Synthetic + BDD slices are proxies |

None of these are implemented; none are claimed. Per PLAN.md scope policy they were the first scope cuts.

## 5. Verification & Validation Evidence

- Monitor V&V: `verification_report.md` (SR-01..06 verified, 88 tests).
- Triggering-condition response: `results/fault_injection_metrics.csv`, `results/ood_metrics.csv`.
- Acceptance separation: thresholds frozen before kitti-test was ever evaluated.

## 6. SOTIF Statement

Within the explored triggering-condition space, the residual risk from *abrupt, severe* perception degradation is argued acceptably mitigated by the fail-safe request mechanism; the residual risk from *gradual, low-signal* degradation is identified, quantified (−19% mAP50 undetected), and explicitly **not** mitigated in this MVP. No claim is made about unexplored conditions.
