"""Deterministic decision engine.

This module is NOT allowed to call any AI model, run any CV, or touch a
camera frame — it only consumes numbers already computed upstream
(perception/depth/localization confidence, local risk, corridor
reachability, a raw clearance reading) and applies configurable thresholds.
That's what "deterministic" means here: same inputs, same output, every
time, with no model inference in the loop.

Two independent hazard-detection paths feed this engine:
  1. The "smart" path — perception/depth/ground-deviation confidence and
     the risk field, already fused upstream.
  2. A "dumb", always-on raw-distance backstop — see check_emergency().
     Phase 1g/1j's ground-deviation detector has a disclosed blind spot at
     very close range (its own interior blends into its blurred
     expectation once an object is close/large enough — see
     depth/ground_deviation.py). The backstop exists specifically so that
     blind spot cannot silently become a collision: it reads raw distance
     directly (in this simulation, Environment.nearest_obstacle_distance —
     standing in for a real UGV's ultrasonic/IR/bump sensor), never touches
     a camera frame, a perception result, or a risk grid, and overrides
     everything else the instant something is too close. A smarter
     ground-deviation detector would still share failure modes with the
     rest of the vision stack (bad light, motion blur, lens dirt); a cheap
     dumb proximity sensor doesn't.

Recovery is intentionally gradual: state can worsen by any number of
severity levels in one cycle (safety-critical faults must be immediate),
but can only IMPROVE by one severity level per cycle. This is what
guarantees DEGRADED never silently resolves straight back to SAFE — it
must pass through WARNING first, on a later, still-good cycle.
"""

from __future__ import annotations

from app.modules.decision.config import DecisionThresholds
from app.modules.decision.schemas import STATE_TO_ACTION, ActionCommand, DecisionResult, SystemState

_SEVERITY = {
    SystemState.SAFE: 0,
    SystemState.WARNING: 1,
    SystemState.DEGRADED: 2,
    SystemState.CRITICAL: 3,
}
_STATE_BY_SEVERITY = {v: k for k, v in _SEVERITY.items()}


class DecisionEngine:
    def __init__(self, thresholds: DecisionThresholds | None = None) -> None:
        self.thresholds = thresholds or DecisionThresholds()
        self.current_state = SystemState.SAFE

    def reset(self) -> None:
        self.current_state = SystemState.SAFE

    def check_emergency(self, raw_min_clearance_m: float | None) -> bool:
        """The dumb, always-on backstop. Deliberately takes nothing but a
        raw distance — no PerceptionResult, no DepthResult, no risk grid —
        so it cannot share a failure mode with the vision pipeline it backs up."""
        if raw_min_clearance_m is None:
            return False
        return raw_min_clearance_m <= self.thresholds.emergency_min_clearance_m

    def evaluate(
        self,
        perception_confidence: float,
        depth_confidence: float,
        localization_confidence: float,
        local_risk: float,
        goal_reachable: bool,
        raw_min_clearance_m: float | None,
    ) -> DecisionResult:
        t = self.thresholds
        combined_confidence = min(perception_confidence, depth_confidence, localization_confidence)

        emergency = self.check_emergency(raw_min_clearance_m)
        no_safe_path = not goal_reachable

        if emergency:
            target_state = SystemState.CRITICAL
            message = f"EMERGENCY STOP: obstacle within {raw_min_clearance_m:.2f}m (raw proximity backstop)"
        elif no_safe_path:
            target_state = SystemState.CRITICAL
            message = "NO SAFE PATH"
        elif combined_confidence < t.degraded_confidence_min or local_risk > t.degraded_risk_max:
            target_state = SystemState.CRITICAL
            message = f"Confidence/risk critical (confidence={combined_confidence:.2f}, risk={local_risk:.2f})"
        elif combined_confidence < t.warning_confidence_min or local_risk > t.warning_risk_max:
            target_state = SystemState.DEGRADED
            message = f"Confidence/risk degraded (confidence={combined_confidence:.2f}, risk={local_risk:.2f})"
        elif combined_confidence < t.safe_confidence_min or local_risk > t.safe_risk_max:
            target_state = SystemState.WARNING
            message = f"Confidence/risk marginal (confidence={combined_confidence:.2f}, risk={local_risk:.2f})"
        else:
            target_state = SystemState.SAFE
            message = "Nominal"

        new_state = self._apply_hysteresis(self.current_state, target_state)
        self.current_state = new_state

        return DecisionResult(
            state=new_state,
            action=STATE_TO_ACTION[new_state],
            message=message,
            combined_confidence=combined_confidence,
            emergency_triggered=emergency,
            no_safe_path=no_safe_path,
        )

    @staticmethod
    def _apply_hysteresis(current: SystemState, target: SystemState) -> SystemState:
        current_severity = _SEVERITY[current]
        target_severity = _SEVERITY[target]
        if target_severity >= current_severity:
            return target  # worsening (or unchanged): immediate, no delay
        return _STATE_BY_SEVERITY[current_severity - 1]  # improving: at most one level per cycle

    def apply_action(self, action: ActionCommand, linear: float, angular: float) -> tuple[float, float]:
        """Gate a planner-proposed cmd_vel through the current action."""
        if action == ActionCommand.RUN:
            return linear, angular
        if action == ActionCommand.SLOW:
            return linear * self.thresholds.slow_speed_factor, angular
        if action == ActionCommand.REPLAN:
            return linear * self.thresholds.slow_speed_factor, angular
        return 0.0, 0.0  # PAUSE — the vehicle does not move
