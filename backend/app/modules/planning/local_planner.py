"""Local planner: re-evaluated every cycle against the current risk grid.

Holds the active global path and steers toward it (pure-pursuit-style
lookahead). Before steering each cycle, it checks the upcoming segment of
that path against the LATEST risk grid — if a newly-detected hazard has
made it unsafe, the segment is invalidated and a fresh global plan is
requested immediately (same cycle), demonstrating real-time replanning
rather than committing to a stale path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.modules.planning.global_planner import AStarPlanner
from app.modules.risk.grid import RiskGrid
from app.modules.simulator.vehicle import CmdVel, VehicleState, wrap_angle


@dataclass
class LocalPlannerConfig:
    lookahead_distance_m: float = 2.0
    goal_tolerance_m: float = 0.4
    base_linear_speed: float = 1.0
    max_angular_speed: float = 1.2
    heading_kp: float = 2.0
    # Local speed modulation is a best-effort default only — phase 1k's
    # Decision Engine is the authoritative RUN/SLOW/REPLAN/PAUSE gate and
    # is not wired in yet (not part of this phase set).
    risk_slowdown_start: float = 0.2
    risk_slowdown_full: float = 0.6
    path_invalidate_risk_threshold: float = 0.6
    replan_check_lookahead_m: float = 3.0
    # A plan may legitimately only reach partway to a distant goal (the
    # corridor generator's frontier waypoint, bounded by its lookahead
    # radius) — once the vehicle arrives at the end of the current path,
    # request a fresh plan toward the next frontier rather than idling.
    path_end_reached_tolerance_m: float = 0.5


@dataclass
class LocalPlanResult:
    cmd_vel: CmdVel
    replanned: bool
    blocked: bool
    goal_reached: bool
    reason: str
    active_path: list[tuple[float, float]] = field(default_factory=list)


class LocalPlanner:
    def __init__(
        self,
        global_planner: AStarPlanner,
        config: LocalPlannerConfig | None = None,
    ) -> None:
        self.global_planner = global_planner
        self.config = config or LocalPlannerConfig()
        self._current_path: list[tuple[float, float]] = []
        self.replan_count = 0

    def reset(self) -> None:
        self._current_path = []
        self.replan_count = 0

    def force_replan(self) -> None:
        """Invalidate the active path so the next step() call plans fresh.

        Called by the decision engine's REPLAN action (phase 1k) — the
        engine decides *that* a replan is warranted from confidence/risk
        it already has; it never touches planning internals directly.
        """
        self._current_path = []

    def step(
        self,
        grid: RiskGrid,
        reachable_mask: np.ndarray,
        vehicle: VehicleState,
        goal: tuple[float, float],
        planning_target: tuple[float, float] | None = None,
    ) -> LocalPlanResult:
        """`goal` is always the true mission goal (used for the goal-reached
        check). `planning_target` optionally overrides what A* actually
        plans toward this cycle — the orchestrator passes the corridor's
        frontier waypoint here when the true goal isn't within the
        currently-known-safe region yet, so the vehicle makes incremental
        progress and re-evaluates as it goes, instead of refusing to move
        until it can already see all the way to a possibly-distant goal."""
        cfg = self.config

        if math.hypot(goal[0] - vehicle.x, goal[1] - vehicle.y) <= cfg.goal_tolerance_m:
            self._current_path = []
            return LocalPlanResult(
                cmd_vel=CmdVel(0.0, 0.0), replanned=False, blocked=False, goal_reached=True, reason="GOAL_REACHED"
            )

        target = planning_target if planning_target is not None else goal

        replanned = False
        if not self._current_path or self._is_path_blocked(grid) or self._reached_path_end(vehicle):
            plan_result = self.global_planner.plan(grid, reachable_mask, (vehicle.x, vehicle.y), target)
            replanned = True
            self.replan_count += 1
            if not plan_result.success:
                self._current_path = []
                return LocalPlanResult(
                    cmd_vel=CmdVel(0.0, 0.0),
                    replanned=True,
                    blocked=True,
                    goal_reached=False,
                    reason=plan_result.reason,
                )
            self._current_path = plan_result.path_world

        target = self._lookahead_point(vehicle)
        desired_heading = math.atan2(target[1] - vehicle.y, target[0] - vehicle.x)
        heading_error = wrap_angle(desired_heading - vehicle.heading)
        angular = float(np.clip(cfg.heading_kp * heading_error, -cfg.max_angular_speed, cfg.max_angular_speed))

        local_risk = float(grid.risk[grid.world_to_cell(vehicle.x, vehicle.y)])
        slowdown_span = max(1e-6, cfg.risk_slowdown_full - cfg.risk_slowdown_start)
        speed_factor = 1.0 - float(np.clip((local_risk - cfg.risk_slowdown_start) / slowdown_span, 0.0, 1.0))
        linear = cfg.base_linear_speed * speed_factor * max(0.0, math.cos(heading_error))

        return LocalPlanResult(
            cmd_vel=CmdVel(linear=linear, angular=angular),
            replanned=replanned,
            blocked=False,
            goal_reached=False,
            reason="TRACKING",
            active_path=list(self._current_path),
        )

    def _reached_path_end(self, vehicle: VehicleState) -> bool:
        if not self._current_path:
            return False
        end = self._current_path[-1]
        return math.hypot(end[0] - vehicle.x, end[1] - vehicle.y) <= self.config.path_end_reached_tolerance_m

    def _is_path_blocked(self, grid: RiskGrid) -> bool:
        cfg = self.config
        cumulative = 0.0
        prev = self._current_path[0] if self._current_path else None
        if prev is None:
            return True
        for wp in self._current_path:
            cumulative += math.hypot(wp[0] - prev[0], wp[1] - prev[1])
            if grid.risk[grid.world_to_cell(*wp)] > cfg.path_invalidate_risk_threshold:
                return True
            if cumulative > cfg.replan_check_lookahead_m:
                break
            prev = wp
        return False

    def _lookahead_point(self, vehicle: VehicleState) -> tuple[float, float]:
        """Standard pure-pursuit lookahead: find the path vertex closest to
        the vehicle first, then advance lookahead_distance_m from there.

        Walking cumulative distance starting from the vehicle's raw
        position to path[0] (the naive version) is unstable whenever the
        vehicle isn't already sitting on the path — small position changes
        near the path's start can make the target jump around unpredictably."""
        cfg = self.config
        path = self._current_path
        if not path:
            return (vehicle.x, vehicle.y)

        distances = [math.hypot(p[0] - vehicle.x, p[1] - vehicle.y) for p in path]
        closest_idx = distances.index(min(distances))

        cumulative = 0.0
        prev = path[closest_idx]
        for wp in path[closest_idx + 1 :]:
            seg_len = math.hypot(wp[0] - prev[0], wp[1] - prev[1])
            if cumulative + seg_len >= cfg.lookahead_distance_m:
                remaining = cfg.lookahead_distance_m - cumulative
                t = remaining / seg_len if seg_len > 0 else 0.0
                return (prev[0] + (wp[0] - prev[0]) * t, prev[1] + (wp[1] - prev[1]) * t)
            cumulative += seg_len
            prev = wp
        return path[-1]
