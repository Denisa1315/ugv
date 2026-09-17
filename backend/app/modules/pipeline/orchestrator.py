"""Ties camera -> perception -> depth -> localization -> risk field ->
corridor -> global/local planner into one per-cycle pipeline, and drives
the Simulator's cmd_vel from it every tick via Simulator.pre_tick_hook —
this is what makes phase 1j's replanning demonstrably real-time within the
actual simulation loop, not a one-off script.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from app.modules.camera.base import CameraSource
from app.modules.corridor.generator import SafeCorridorGenerator
from app.modules.corridor.types import CorridorConfig
from app.modules.decision.config import DecisionThresholds
from app.modules.decision.engine import DecisionEngine
from app.modules.decision.schemas import ActionCommand, SystemState
from app.modules.depth.base import DepthEstimator
from app.modules.geometry import CameraGeometryConfig
from app.modules.localization.base import Localizer
from app.modules.localization.schemas import RelativePose
from app.modules.perception.base import PerceptionModule
from app.modules.planning.global_planner import AStarCostConfig, AStarPlanner
from app.modules.planning.local_planner import LocalPlanner, LocalPlannerConfig
from app.modules.risk.builder import RiskFieldBuilder
from app.modules.risk.config import RiskFieldConfig
from app.modules.risk.grid import RiskGrid
from app.modules.simulator.environment import Environment
from app.modules.simulator.simulator import Simulator
from app.modules.simulator.vehicle import CmdVel
from app.modules.telemetry.event_logger import EventLogger


@dataclass
class PipelineCycleResult:
    timestamp: float
    perception_confidence: float
    perception_mode: str
    num_obstacles: int
    num_terrain_deviations: int
    depth_confidence: float
    localization_confidence: float
    localization_status: str
    localization_pose: RelativePose
    goal_reachable: bool
    replanned: bool
    blocked: bool
    goal_reached: bool
    plan_reason: str
    active_path_length: int
    cmd_vel: CmdVel
    replan_count_total: int
    # decision engine (phase 1k) — the authoritative safety gate
    system_state: SystemState
    action: ActionCommand
    decision_message: str
    emergency_triggered: bool
    raw_min_clearance_m: float | None
    local_risk: float


class NavigationPipeline:
    def __init__(
        self,
        environment: Environment,
        camera: CameraSource,
        perception: PerceptionModule,
        depth: DepthEstimator,
        localizer: Localizer,
        camera_geometry: CameraGeometryConfig | None = None,
        risk_config: RiskFieldConfig | None = None,
        corridor_config: CorridorConfig | None = None,
        astar_config: AStarCostConfig | None = None,
        local_config: LocalPlannerConfig | None = None,
        decision_thresholds: DecisionThresholds | None = None,
        event_logger: EventLogger | None = None,
    ) -> None:
        self.camera = camera
        self.perception = perception
        self.depth = depth
        self.localizer = localizer
        self.camera_geometry = camera_geometry or CameraGeometryConfig()

        self.risk_grid = RiskGrid(environment, risk_config or RiskFieldConfig())
        self.risk_builder = RiskFieldBuilder(self.camera_geometry)
        self.corridor_generator = SafeCorridorGenerator(corridor_config)
        self.local_planner = LocalPlanner(AStarPlanner(astar_config), local_config)
        self.decision_engine = DecisionEngine(decision_thresholds)
        self.event_logger = event_logger or EventLogger()

        self.camera.start()

        self.last_cycle: PipelineCycleResult | None = None
        self.last_annotated_frame = None
        self.last_active_path: list[tuple[float, float]] = []
        self._previous_state: SystemState | None = None
        self._previous_emergency = False
        self._previous_goal_reached = False
        self._fault_injection_until = 0.0
        self._fault_injection_value = 1.0
        self._localization_confidence_ema: float | None = None

    def simulate_low_confidence(self, duration_s: float = 5.0, forced_value: float = 0.1) -> None:
        """DEMO/TEST fault injection ("Simulate Low Confidence" dashboard
        control): forces perception confidence down for a window, so the
        decision engine's DEGRADED/CRITICAL response and its gradual,
        staged recovery (phase 1k) are directly observable. This overrides
        the value actually used for both reporting and decision-making —
        it is a deliberate, disclosed fault injection, not a fabricated
        "real" sensor reading."""
        self._fault_injection_until = time.time() + duration_s
        self._fault_injection_value = forced_value

    def reset(self) -> None:
        self.localizer.reset()
        self.local_planner.reset()
        self.decision_engine.reset()
        self.risk_grid.terrain_cost[:] = self.risk_grid.config.default_cost
        self.risk_grid.obstacle_cost[:] = self.risk_grid.config.default_cost
        self.risk_grid.combine()
        self.last_cycle = None
        self.last_annotated_frame = None
        self.last_active_path = []
        self._previous_state = None
        self._previous_emergency = False
        self._previous_goal_reached = False
        self._localization_confidence_ema = None

    def cycle(self, simulator: Simulator, dt: float) -> PipelineCycleResult | None:
        self.camera.update_world_context(simulator.vehicle.state, simulator.environment)
        frame = self.camera.read_frame()
        if frame is None:
            return None

        perception_result = self.perception.process(frame)
        depth_result = self.depth.estimate(frame, perception_result)
        localization_result = self.localizer.track(frame, dt)

        if time.time() < self._fault_injection_until:
            perception_result.confidence = min(perception_result.confidence, self._fault_injection_value)

        self.risk_builder.update(
            self.risk_grid,
            simulator.vehicle.state,
            frame.shape[:2],
            perception_result,
            depth_result,
            localization_result,
            dt,
        )
        true_goal = simulator.environment.goal
        corridor_result = self.corridor_generator.generate(self.risk_grid, simulator.vehicle.state, true_goal)
        # Plan toward the true goal once it's within the known-safe region;
        # until then, aim for the corridor's frontier waypoint so the
        # vehicle makes incremental progress instead of stalling until it
        # can already see all the way to a possibly-distant goal.
        planning_target = true_goal if corridor_result.goal_reachable else corridor_result.target_waypoint
        local_result = self.local_planner.step(
            self.risk_grid,
            corridor_result.reachable_mask,
            simulator.vehicle.state,
            true_goal,
            planning_target=planning_target,
        )

        # Decision engine (phase 1k): the authoritative safety gate. Two
        # independent hazard checks feed it — the fused vision-stack
        # confidence/risk already computed above, and a raw-distance
        # backstop read directly from Environment.nearest_obstacle_distance
        # (standing in for a real UGV's ultrasonic/IR/bump sensor). The
        # backstop never touches the camera frame, perception, or risk
        # grid, so it cannot share the ground-deviation detector's
        # disclosed close-range blind spot.
        local_risk = float(self.risk_grid.risk[self.risk_grid.world_to_cell(simulator.vehicle.state.x, simulator.vehicle.state.y)])
        raw_min_clearance_m = simulator.environment.nearest_obstacle_distance(simulator.vehicle.state.x, simulator.vehicle.state.y)

        # Monocular VO confidence is single-frame noisy by nature (it
        # depends on how many corners this specific frame happened to
        # match) — reacting to one bad reading instantly is what produced a
        # "climb two cycles, crash one cycle" stall that never netted
        # forward progress. Smoothing (EMA) is standard practice before
        # feeding a noisy sensor into a safety-critical gate: it responds to
        # SUSTAINED degradation, not single-frame flicker.
        alpha = 0.3
        if self._localization_confidence_ema is None:
            self._localization_confidence_ema = localization_result.confidence
        else:
            self._localization_confidence_ema = (
                alpha * localization_result.confidence + (1 - alpha) * self._localization_confidence_ema
            )
        effective_localization_confidence = self._localization_confidence_ema

        # Monocular VO also has a genuine cold-start/self-lock problem: it
        # needs motion to produce any confidence at all, but PAUSE (from low
        # confidence) produces no motion — a real deadlock, normally broken
        # by fusing wheel odometry/IMU (out of scope here). Dead-reckoning
        # uncertainty does not grow while genuinely stationary, so gating on
        # VO confidence while the vehicle isn't moving is simply the wrong
        # model — it's floored at the WARNING boundary whenever (a) no
        # estimate has been obtained yet ("initializing" — not the same
        # claim as "tracked and found unreliable", i.e. "lost"), or (b) the
        # vehicle's last applied velocity was ~0, so a degenerate
        # zero-baseline frame pair can't be mistaken for a tracking failure.
        was_near_stationary = abs(simulator.vehicle.state.velocity) < 0.05
        if localization_result.status == "initializing" or was_near_stationary:
            effective_localization_confidence = max(
                effective_localization_confidence, self.decision_engine.thresholds.warning_confidence_min
            )
            self._localization_confidence_ema = effective_localization_confidence

        decision_result = self.decision_engine.evaluate(
            perception_confidence=perception_result.confidence,
            depth_confidence=depth_result.confidence,
            localization_confidence=effective_localization_confidence,
            local_risk=local_risk,
            # NOT corridor_result.goal_reachable: that's merely "is the
            # literal (possibly distant) goal within the confidently-safe
            # region yet" — normally False early in almost any mission, and
            # not itself an emergency (see phase 1j's frontier-waypoint
            # mechanism). local_result.blocked is the genuine "no path
            # forward exists at all, even toward the corridor's frontier"
            # signal this safety rule actually means.
            goal_reachable=not local_result.blocked,
            raw_min_clearance_m=raw_min_clearance_m,
        )

        if decision_result.action == ActionCommand.REPLAN:
            self.local_planner.force_replan()

        self._log_events(simulator, local_result, decision_result, local_risk, effective_localization_confidence)

        if local_result.goal_reached:
            # Arrival is a successful stop, not a safety fault — don't let
            # the gate relabel it.
            final_linear, final_angular = 0.0, 0.0
        else:
            final_linear, final_angular = self.decision_engine.apply_action(
                decision_result.action, local_result.cmd_vel.linear, local_result.cmd_vel.angular
            )
        simulator.set_cmd_vel(final_linear, final_angular)

        cycle_result = PipelineCycleResult(
            timestamp=time.time(),
            perception_confidence=perception_result.confidence,
            perception_mode=perception_result.mode,
            num_obstacles=len(perception_result.obstacles),
            num_terrain_deviations=len(depth_result.ground_deviation_regions),
            depth_confidence=depth_result.confidence,
            localization_confidence=localization_result.confidence,
            localization_status=localization_result.status,
            localization_pose=localization_result.pose,
            goal_reachable=corridor_result.goal_reachable,
            replanned=local_result.replanned,
            blocked=local_result.blocked,
            goal_reached=local_result.goal_reached,
            plan_reason=local_result.reason,
            active_path_length=len(local_result.active_path),
            cmd_vel=CmdVel(linear=final_linear, angular=final_angular),
            replan_count_total=self.local_planner.replan_count,
            system_state=decision_result.state,
            action=decision_result.action,
            decision_message=decision_result.message,
            emergency_triggered=decision_result.emergency_triggered,
            raw_min_clearance_m=raw_min_clearance_m,
            local_risk=local_risk,
        )
        self.last_cycle = cycle_result
        self.last_annotated_frame = perception_result.annotated_frame
        self.last_active_path = local_result.active_path
        return cycle_result

    def _log_events(self, simulator, local_result, decision_result, local_risk, confidence) -> None:
        x, y = simulator.vehicle.state.x, simulator.vehicle.state.y

        if local_result.replanned and not local_result.blocked:
            self.event_logger.log(
                "PATH_REPLANNED", "INFO", x, y, local_risk, confidence,
                f"Local planner replanned ({self.local_planner.replan_count} total)",
            )

        if decision_result.state != self._previous_state:
            severity = "CRITICAL" if decision_result.state == SystemState.CRITICAL else (
                "WARNING" if decision_result.state in (SystemState.WARNING, SystemState.DEGRADED) else "INFO"
            )
            self.event_logger.log(
                f"{decision_result.state.value}_ENTERED", severity, x, y, local_risk, confidence,
                decision_result.message,
            )
            self._previous_state = decision_result.state

        if decision_result.emergency_triggered and not self._previous_emergency:
            self.event_logger.log(
                "EMERGENCY_STOP", "CRITICAL", x, y, local_risk, confidence, decision_result.message
            )
        self._previous_emergency = decision_result.emergency_triggered

        if local_result.goal_reached and not self._previous_goal_reached:
            self.event_logger.log("GOAL_REACHED", "INFO", x, y, local_risk, confidence, "Vehicle reached the goal")
        self._previous_goal_reached = local_result.goal_reached

    def as_pre_tick_hook(self, simulator: Simulator) -> Callable[[float], None]:
        def hook(dt: float) -> None:
            self.cycle(simulator, dt)

        return hook
