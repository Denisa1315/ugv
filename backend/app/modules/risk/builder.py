"""Populates a RiskGrid from one pipeline cycle's perception/depth/
localization outputs and the vehicle's current (ground-truth) pose.

Per the phase 1g "IMPORTANT ADDITION": terrain_cost is fed PRIMARILY by
depth's class-agnostic ground_deviation_regions (catches un-classified,
static hazards regardless of motion or COCO class); perception.obstacles
(YOLO/MOG2) are layered on top of obstacle_cost as a SUPPLEMENTARY signal.
depth_cost and localization_uncertainty_cost are range-based confidence
layers, not hazard-presence layers — see risk/config.py docstrings.
"""

from __future__ import annotations

import numpy as np

from app.modules.depth.schemas import DepthResult
from app.modules.geometry import CameraGeometryConfig, image_point_to_world
from app.modules.localization.schemas import LocalizationResult
from app.modules.perception.schemas import ObstacleDetection, PerceptionResult
from app.modules.risk.grid import RiskGrid
from app.modules.simulator.vehicle import VehicleState


def _contact_point(box: ObstacleDetection) -> tuple[int, int]:
    return (box.x1 + box.x2) // 2, box.y2


class RiskFieldBuilder:
    def __init__(self, camera_geometry: CameraGeometryConfig | None = None) -> None:
        self.camera_geometry = camera_geometry or CameraGeometryConfig()

    def update(
        self,
        grid: RiskGrid,
        vehicle: VehicleState,
        frame_shape: tuple[int, int],
        perception: PerceptionResult,
        depth: DepthResult,
        localization: LocalizationResult,
        dt: float,
    ) -> None:
        cfg = grid.config

        decay = float(np.clip(1.0 - cfg.decay_per_second * dt, 0.0, 1.0)) if dt > 0 else 1.0
        grid.terrain_cost = grid.terrain_cost * decay + cfg.default_cost * (1 - decay)
        grid.obstacle_cost = grid.obstacle_cost * decay + cfg.default_cost * (1 - decay)

        # PRIMARY signal: class-agnostic ground-plane deviation -> terrain_cost.
        for region, relative_depth in zip(depth.ground_deviation_regions, depth.ground_deviation_relative_depth):
            px, py = _contact_point(region)
            wx, wy = image_point_to_world(px, py, frame_shape, vehicle, self.camera_geometry, depth.relative_depth_map)
            grid.splat(grid.terrain_cost, wx, wy, value=region.confidence, radius_m=cfg.splat_radius_m)

        # SUPPLEMENTARY signal: YOLO/MOG2 named-or-moving-object detections -> obstacle_cost.
        for obstacle, relative_depth in zip(perception.obstacles, depth.obstacle_relative_depth):
            px, py = _contact_point(obstacle)
            wx, wy = image_point_to_world(px, py, frame_shape, vehicle, self.camera_geometry, depth.relative_depth_map)
            grid.splat(
                grid.obstacle_cost,
                wx,
                wy,
                value=obstacle.confidence * cfg.obstacle_supplementary_weight,
                radius_m=cfg.splat_radius_m,
            )

        grid.depth_cost = grid.range_cost_from(vehicle.x, vehicle.y, cfg.depth_cost_growth_per_meter)

        loc_growth = cfg.localization_cost_growth_per_meter * (1.0 - localization.confidence)
        loc_range_cost = grid.range_cost_from(vehicle.x, vehicle.y, loc_growth)
        loc_floor = (1.0 - localization.confidence) * cfg.localization_uncertainty_floor_weight
        grid.localization_uncertainty_cost = np.clip(loc_range_cost + loc_floor, 0.0, 1.0).astype(np.float32)

        grid.combine()
