from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CorridorConfig:
    # Scoring weights for picking a local target waypoint within the
    # reachable region — distance-to-goal, risk, clearance, and heading
    # consistency all matter, not just "nearest free cell".
    distance_weight: float = 1.0
    risk_weight: float = 3.0
    clearance_weight: float = 0.4
    heading_weight: float = 0.5
    lookahead_radius_m: float = 4.0


@dataclass
class SafeCorridorResult:
    reachable_mask: np.ndarray  # bool grid: safe cells connected to the vehicle's current cell
    clearance_m: np.ndarray  # float grid: distance (meters) to the nearest unsafe cell
    goal_reachable: bool  # is the goal cell in the same connected component as the vehicle?
    target_waypoint: tuple[float, float] | None  # best local waypoint within the corridor, or None
