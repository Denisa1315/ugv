from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SystemState(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class ActionCommand(str, Enum):
    RUN = "RUN"
    SLOW = "SLOW"
    REPLAN = "REPLAN"
    PAUSE = "PAUSE"


# Deterministic 1:1 mapping, per the architecture spec.
STATE_TO_ACTION: dict[SystemState, ActionCommand] = {
    SystemState.SAFE: ActionCommand.RUN,
    SystemState.WARNING: ActionCommand.SLOW,
    SystemState.DEGRADED: ActionCommand.REPLAN,
    SystemState.CRITICAL: ActionCommand.PAUSE,
}


@dataclass
class DecisionResult:
    state: SystemState
    action: ActionCommand
    message: str
    combined_confidence: float
    emergency_triggered: bool
    no_safe_path: bool
