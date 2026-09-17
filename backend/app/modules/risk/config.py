from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskWeights:
    """risk = w1*terrain_cost + w2*obstacle_cost + w3*depth_cost + w4*localization_uncertainty_cost"""

    # w1_terrain is deliberately dominant: a single confident ground-plane-
    # deviation detection must clear the safe/unsafe threshold with enough
    # margin to produce a real keep-out radius, not just a 1-cell pinprick
    # — detection position has real error, and the obstacle itself has
    # physical extent, both need covering.
    w1_terrain: float = 1.0
    w2_obstacle: float = 0.25
    w3_depth: float = 0.15
    w4_localization: float = 0.20


@dataclass
class RiskFieldConfig:
    weights: RiskWeights = field(default_factory=RiskWeights)

    resolution_cells_per_meter: float = 2.0

    # terrain_cost/obstacle_cost persist between cycles (memory of past
    # observations) and decay back toward this neutral baseline if a cell
    # isn't re-observed — this fraction of the way back per second.
    default_cost: float = 0.15
    decay_per_second: float = 0.35

    # How far (in meters) a single detected hazard's risk contribution
    # spreads into neighboring cells. Generous relative to a typical small
    # hazard's own size: the ground-plane depth heuristic's position
    # estimate carries real error (a few tenths of a meter), so the splat
    # needs margin beyond the hazard's true footprint to leave the vehicle
    # actual clearance rather than grazing it.
    splat_radius_m: float = 3.0

    # Per the phase 1g "IMPORTANT ADDITION": YOLO/MOG2 detections
    # (perception.obstacles) are a SUPPLEMENTARY signal layered on top of
    # the depth-based ground-plane-deviation PRIMARY signal, not the other
    # way around — scaled down relative to raw detection confidence.
    obstacle_supplementary_weight: float = 0.6

    # depth_cost: a monocular relative-depth estimate gets less reliable
    # with range (perspective foreshortening) — modeled as cost growing
    # with distance from the vehicle, independent of any specific detection.
    depth_cost_growth_per_meter: float = 0.03

    # localization_uncertainty_cost: dead-reckoning-style growing
    # uncertainty cone away from the vehicle, steeper when the localizer's
    # own confidence is low, plus a flat floor scaled by (1 - confidence)
    # so poor localization raises risk even right next to the vehicle.
    localization_cost_growth_per_meter: float = 0.05
    localization_uncertainty_floor_weight: float = 0.2

    safe_risk_threshold: float = 0.35  # risk <= this => traversable
