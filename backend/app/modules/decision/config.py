from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionThresholds:
    # combined_confidence = min(perception, depth, localization) — the
    # weakest sensor gates the system, not an average that a lone bad
    # reading could get diluted out of.
    safe_confidence_min: float = 0.6
    warning_confidence_min: float = 0.4
    degraded_confidence_min: float = 0.2  # below this -> CRITICAL

    safe_risk_max: float = 0.35
    warning_risk_max: float = 0.6
    degraded_risk_max: float = 0.8  # above this -> CRITICAL

    # Raw-distance emergency backstop (see engine.py docstring) — an
    # idealized proximity-sensor reading, independent of the vision stack.
    emergency_min_clearance_m: float = 0.4

    # Local speed multiplier applied under the SLOW/REPLAN actions.
    slow_speed_factor: float = 0.35
