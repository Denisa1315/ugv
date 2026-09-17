"""Shared camera <-> world projection math.

Everything downstream of perception/depth operates in pixel space; the risk
field, corridors, and planners operate in world meters (matching the
Simulator's Environment). This module is the one place that bridges the
two, using a simple pinhole-camera + flat-ground-plane model consistent
with the depth module's row-to-distance heuristic:

  - Depth's relative_depth_map[row] represents forward (optical-axis, "Z")
    distance for ground-plane points at that image row, independent of
    column — true for a level camera looking at a flat ground plane.
  - CameraGeometryConfig.near_m/far_m is an explicit, documented ASSUMPTION
    mapping that relative [0, 1] depth to a metric range, since this MVP
    has no real depth sensor. A real depth camera / stereo rig would
    replace relative_depth_to_meters()/meters_to_relative_depth() without
    changing any planner code (see README "Future Industrial Version").

world_point_to_image() and image_point_to_world() are exact algebraic
inverses of each other (verified by round-trip tests) given the same
CameraGeometryConfig, vehicle pose, and image shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.modules.simulator.vehicle import VehicleState


@dataclass
class CameraGeometryConfig:
    horizontal_fov_deg: float = 80.0
    near_m: float = 0.5  # assumed metric distance at relative_depth = 0
    far_m: float = 15.0  # assumed metric distance at relative_depth = 1
    horizon_fraction: float = 0.45  # must match perception/depth's horizon assumption

    def focal_px(self, image_width: int) -> float:
        return (image_width / 2) / math.tan(math.radians(self.horizontal_fov_deg) / 2)

    def relative_depth_to_meters(self, relative_depth: float) -> float:
        return self.near_m + float(np.clip(relative_depth, 0.0, 1.0)) * (self.far_m - self.near_m)

    def meters_to_relative_depth(self, meters: float) -> float:
        span = self.far_m - self.near_m
        if span <= 0:
            return 0.0
        return float(np.clip((meters - self.near_m) / span, 0.0, 1.0))


def image_point_to_world(
    px: int,
    py: int,
    image_shape: tuple[int, int],
    vehicle: VehicleState,
    geometry: CameraGeometryConfig,
    relative_depth_map: np.ndarray,
) -> tuple[float, float]:
    """Project a pixel (assumed to lie on the ground plane) into world (x, y)."""
    h, w = image_shape
    py_clamped = int(np.clip(py, 0, h - 1))
    px_clamped = int(np.clip(px, 0, w - 1))
    relative_depth = float(relative_depth_map[py_clamped, px_clamped])

    forward = geometry.relative_depth_to_meters(relative_depth)  # Z
    focal = geometry.focal_px(w)
    lateral = forward * (px_clamped - w / 2) / focal  # X

    theta = vehicle.heading
    world_x = vehicle.x + forward * math.cos(theta) + lateral * math.sin(theta)
    world_y = vehicle.y + forward * math.sin(theta) - lateral * math.cos(theta)
    return world_x, world_y


def world_point_to_image(
    wx: float,
    wy: float,
    image_shape: tuple[int, int],
    vehicle: VehicleState,
    geometry: CameraGeometryConfig,
) -> tuple[int, int, float] | None:
    """Project a world point into image pixels, or None if behind/outside FOV."""
    h, w = image_shape
    theta = vehicle.heading
    dx = wx - vehicle.x
    dy = wy - vehicle.y

    forward = dx * math.cos(theta) + dy * math.sin(theta)  # Z
    lateral = dx * math.sin(theta) - dy * math.cos(theta)  # X

    if forward <= geometry.near_m * 0.5:
        return None  # behind the camera or too close to project meaningfully

    bearing = math.atan2(lateral, forward)
    if abs(bearing) > math.radians(geometry.horizontal_fov_deg) / 2:
        return None  # outside the horizontal field of view

    focal = geometry.focal_px(w)
    px = w / 2 + focal * (lateral / forward)
    if not (0 <= px < w):
        return None

    relative_depth = geometry.meters_to_relative_depth(forward)
    horizon_row = int(h * geometry.horizon_fraction)
    span = max(1, h - 1 - horizon_row)
    py = horizon_row + span * (1.0 - relative_depth)

    return int(px), int(np.clip(py, horizon_row, h - 1)), forward
