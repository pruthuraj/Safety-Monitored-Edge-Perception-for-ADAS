"""Runtime gating state machine (SR-03/SR-04, Week 6, EXP-009).

States: NOMINAL -> DEGRADED -> FAIL_SAFE_REQUEST. Transitions trigger on
SUSTAINED threshold breaches, never single-frame spikes (SR-03), using the
balanced policy fixed in the Week 6 plan:

- NOMINAL -> DEGRADED:            score > Q95 for 3 consecutive frames
- NOMINAL/DEGRADED -> FAIL_SAFE:  score > Q99 for 2 consecutive frames
- DEGRADED -> FAIL_SAFE:          10 frames in DEGRADED without recovery
- FAIL_SAFE -> DEGRADED:          5 consecutive clean frames (score <= Q95)
- DEGRADED -> NOMINAL:            5 consecutive clean frames (score <= Q95)

Breach is strict: score > threshold (equality is clean). Thresholds come
frozen from results/monitor_thresholds.json (kitti-val only, EXP-008) and
are never recomputed here.
"""

from __future__ import annotations

from dataclasses import dataclass

NOMINAL = "NOMINAL"
DEGRADED = "DEGRADED"
FAIL_SAFE_REQUEST = "FAIL_SAFE_REQUEST"

Q95_CONSECUTIVE = 3
Q99_CONSECUTIVE = 2
DEGRADED_PERSIST_LIMIT = 10
CLEAN_RECOVERY = 5


@dataclass
class Transition:
    state_before: str
    state_after: str
    reason: str


class MonitorStateMachine:
    """Deterministic three-state gating machine; one `step` per frame."""

    def __init__(self, q95: float, q99: float):
        if not q95 <= q99:
            raise ValueError(f"require q95 <= q99, got {q95} > {q99}")
        self.q95 = float(q95)
        self.q99 = float(q99)
        self.state = NOMINAL
        self._q95_streak = 0
        self._q99_streak = 0
        self._clean_streak = 0
        self._degraded_frames = 0

    def step(self, score: float) -> Transition:
        """Advance one frame with the monitor score; returns the transition."""
        before = self.state
        score = float(score)

        # streak bookkeeping (strict >: equality counts as clean/no-breach)
        self._q95_streak = self._q95_streak + 1 if score > self.q95 else 0
        self._q99_streak = self._q99_streak + 1 if score > self.q99 else 0
        self._clean_streak = self._clean_streak + 1 if score <= self.q95 else 0
        if self.state == DEGRADED:
            self._degraded_frames += 1

        reason = "no_change"
        if self.state in (NOMINAL, DEGRADED) and self._q99_streak >= Q99_CONSECUTIVE:
            self.state = FAIL_SAFE_REQUEST
            reason = f"q99_breach_x{Q99_CONSECUTIVE}"
        elif self.state == DEGRADED and self._degraded_frames >= DEGRADED_PERSIST_LIMIT:
            self.state = FAIL_SAFE_REQUEST
            reason = f"degraded_persist_{DEGRADED_PERSIST_LIMIT}"
        elif self.state == NOMINAL and self._q95_streak >= Q95_CONSECUTIVE:
            self.state = DEGRADED
            reason = f"q95_breach_x{Q95_CONSECUTIVE}"
        elif self.state == FAIL_SAFE_REQUEST and self._clean_streak >= CLEAN_RECOVERY:
            self.state = DEGRADED
            reason = f"recovered_clean_x{CLEAN_RECOVERY}"
        elif self.state == DEGRADED and self._clean_streak >= CLEAN_RECOVERY:
            self.state = NOMINAL
            reason = f"recovered_clean_x{CLEAN_RECOVERY}"

        if self.state != before:
            # entering a state resets its bookkeeping window
            self._degraded_frames = 0
            self._clean_streak = 0
            self._q95_streak = 0
            self._q99_streak = 0

        return Transition(before, self.state, reason)
