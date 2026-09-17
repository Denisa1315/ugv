"""Safe corridor generator: finds the connected low-risk region reachable
from the vehicle's current cell, and (within it) scores candidate local
waypoints by distance-to-goal, risk, clearance, and heading consistency —
not just "nearest free pixel".

This connected-component reachability check is also what lets the global
planner (1i) correctly report "no safe path" (goal not in the same
component as the vehicle) instead of silently failing, per the safety rule:
if no safe path exists, the vehicle must not move.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.modules.corridor.types import CorridorConfig, SafeCorridorResult
from app.modules.risk.grid import RiskGrid
from app.modules.simulator.vehicle import VehicleState


def _wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2 * np.pi) - np.pi


class SafeCorridorGenerator:
    def __init__(self, config: CorridorConfig | None = None) -> None:
        self.config = config or CorridorConfig()

    def generate(self, grid: RiskGrid, vehicle: VehicleState, goal: tuple[float, float]) -> SafeCorridorResult:
        traversable = grid.safe_mask.astype(np.uint8)
        vehicle_row, vehicle_col = grid.world_to_cell(vehicle.x, vehicle.y)

        num_labels, labels = cv2.connectedComponents(traversable, connectivity=8)
        vehicle_label = labels[vehicle_row, vehicle_col]

        if vehicle_label == 0:
            # The vehicle's own cell reads unsafe (e.g. a momentary risk
            # spike right under it) — treat it as a trivial reachable seed
            # rather than declaring the whole map unreachable.
            reachable_mask = np.zeros_like(traversable, dtype=bool)
            reachable_mask[vehicle_row, vehicle_col] = True
        else:
            reachable_mask = labels == vehicle_label

        clearance_px = cv2.distanceTransform((traversable * 255).astype(np.uint8), cv2.DIST_L2, 5)
        clearance_m = (clearance_px / grid.resolution).astype(np.float32)

        goal_row, goal_col = grid.world_to_cell(*goal)
        goal_reachable = bool(reachable_mask[goal_row, goal_col])

        target_waypoint = self._select_target_waypoint(grid, vehicle, goal, reachable_mask, clearance_m)

        return SafeCorridorResult(
            reachable_mask=reachable_mask,
            clearance_m=clearance_m,
            goal_reachable=goal_reachable,
            target_waypoint=target_waypoint,
        )

    def _select_target_waypoint(
        self,
        grid: RiskGrid,
        vehicle: VehicleState,
        goal: tuple[float, float],
        reachable_mask: np.ndarray,
        clearance_m: np.ndarray,
    ) -> tuple[float, float] | None:
        cfg = self.config
        rows_idx, cols_idx = np.where(reachable_mask)
        if len(rows_idx) == 0:
            return None

        world_xs = (cols_idx + 0.5) / grid.resolution
        world_ys = (rows_idx + 0.5) / grid.resolution
        dist_from_vehicle = np.hypot(world_xs - vehicle.x, world_ys - vehicle.y)

        within_lookahead = dist_from_vehicle <= cfg.lookahead_radius_m
        if not np.any(within_lookahead):
            within_lookahead = np.ones_like(dist_from_vehicle, dtype=bool)
        idxs = np.where(within_lookahead)[0]

        gx, gy = goal
        dist_to_goal = np.hypot(gx - world_xs[idxs], gy - world_ys[idxs])
        risk_vals = grid.risk[rows_idx[idxs], cols_idx[idxs]]
        clearance_vals = clearance_m[rows_idx[idxs], cols_idx[idxs]]
        desired_headings = np.arctan2(world_ys[idxs] - vehicle.y, world_xs[idxs] - vehicle.x)
        heading_dev = np.abs(_wrap_angle_array(desired_headings - vehicle.heading))

        # Normalize every term to a comparable ~[0, 1-2] range before
        # weighting — raw clearance/distance are in meters (can be several
        # times larger than heading_dev's [0, pi] or risk's [0, 1] range),
        # which would otherwise silently swamp the other terms regardless
        # of the configured weights.
        lookahead = max(cfg.lookahead_radius_m, 1e-6)
        norm_dist_to_goal = dist_to_goal / lookahead
        norm_clearance = np.minimum(clearance_vals, lookahead) / lookahead
        norm_heading_dev = heading_dev / np.pi

        scores = (
            -cfg.distance_weight * norm_dist_to_goal
            - cfg.risk_weight * risk_vals
            + cfg.clearance_weight * norm_clearance
            - cfg.heading_weight * norm_heading_dev
        )
        best = idxs[int(np.argmax(scores))]
        return grid.cell_to_world(int(rows_idx[best]), int(cols_idx[best]))
