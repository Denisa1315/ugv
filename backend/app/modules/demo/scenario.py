"""Phase 1n: the one-button "Start SIH Demo" scripted scenario.

Deterministic and repeatable *by construction*: milestones trigger on the
vehicle's PROGRESS toward the goal (fraction of straight-line distance
covered), never on wall-clock time, so the scenario reaches the same
sequence of events regardless of exact pipeline timing on a given machine.

Scripted moments, in order:
  1. Point A -> navigating toward Point B.
  2. ~25% of the way: an obstacle appears on the route — the vision
     pipeline must detect it (ground-deviation + perception) and replan.
  3. ~45%: a simulated confidence-degradation fault is injected (same
     mechanism as the dashboard's "Simulate Low Confidence" button) —
     shows DEGRADED/CRITICAL entered and then staged, gradual recovery.
  4. ~70%: a hazard is placed within the raw-distance emergency backstop's
     clearance threshold — demonstrates phase 1k's dumb, independent
     proximity check forcing PAUSE — then is cleared a couple seconds
     later so the mission can complete.
  5. Point B reached -> SUCCESS.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

from app.modules.pipeline.factory import create_pipeline
from app.modules.simulator.simulator import Simulator

START = (1.0, 1.0)
GOAL = (18.0, 18.0)
TIMEOUT_S = 90.0
POLL_INTERVAL_S = 0.2


@dataclass
class DemoState:
    status: str = "IDLE"  # IDLE | RUNNING | SUCCESS | FAILED | TIMEOUT
    phase: str = "idle"
    log: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {message}")


class DemoScenario:
    def __init__(self, app_state, db_path: str = "ugv_events.db") -> None:
        self.app_state = app_state
        self.db_path = db_path
        self.state = DemoState()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return  # already running — ignore duplicate start
        self.state = DemoState(status="RUNNING", phase="starting")
        self._task = asyncio.create_task(self._run_safely())

    async def _run_safely(self) -> None:
        try:
            await self._run()
        except Exception as exc:  # pragma: no cover - defensive, surfaced to the dashboard
            self.state.status = "FAILED"
            self.state.note(f"Demo scenario error: {exc}")

    async def _run(self) -> None:
        sim: Simulator = self.app_state.simulator
        sim.pre_tick_hook = None
        sim.environment.set_goal(*GOAL)
        sim.environment.clear_obstacles()
        sim.reset(x=START[0], y=START[1], heading=math.atan2(GOAL[1] - START[1], GOAL[0] - START[0]))

        pipeline = create_pipeline(sim.environment, camera_mode="demo", perception_mode="fallback", db_path=self.db_path)
        self.app_state.pipeline = pipeline
        sim.pre_tick_hook = pipeline.as_pre_tick_hook(sim)

        self.state.note(f"Point A {START} -> Point B {GOAL}: mission started")
        sim.start()
        sim.ensure_running_task()

        total_distance = math.hypot(GOAL[0] - START[0], GOAL[1] - START[1])
        obstacle_done = confidence_done = emergency_done = False
        emergency_obstacle_id: str | None = None
        emergency_triggered_at: float | None = None
        start_time = time.time()

        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            vx, vy = sim.vehicle.state.x, sim.vehicle.state.y
            progress = 1.0 - (sim.environment.distance_to_goal(vx, vy) / total_distance)

            if not obstacle_done and progress >= 0.25:
                heading = sim.vehicle.state.heading
                ox, oy = vx + 3.0 * math.cos(heading), vy + 3.0 * math.sin(heading)
                sim.environment.add_obstacle(ox, oy, radius=0.8)
                self.state.phase = "obstacle_appeared"
                self.state.note(f"Obstacle appeared near ({ox:.1f}, {oy:.1f}) — awaiting detection and replan")
                obstacle_done = True

            if not confidence_done and progress >= 0.45:
                pipeline.simulate_low_confidence(duration_s=4.0, forced_value=0.08)
                self.state.phase = "confidence_degraded"
                self.state.note("Simulated confidence degradation injected — watch it degrade, then recover in stages")
                confidence_done = True

            if not emergency_done and progress >= 0.70:
                heading = sim.vehicle.state.heading
                ex, ey = vx + 0.3 * math.cos(heading), vy + 0.3 * math.sin(heading)
                obstacle = sim.environment.add_obstacle(ex, ey, radius=0.15)
                emergency_obstacle_id = obstacle.id
                emergency_triggered_at = time.time()
                self.state.phase = "emergency_stop"
                self.state.note("Hazard placed inside the emergency clearance radius — raw-distance backstop should PAUSE the vehicle")
                emergency_done = True

            if emergency_obstacle_id is not None and time.time() - (emergency_triggered_at or 0) > 2.5:
                sim.environment.remove_obstacle(emergency_obstacle_id)
                emergency_obstacle_id = None
                self.state.note("Hazard cleared — vehicle resumes toward the goal")

            if sim.goal_reached:
                self.state.status = "SUCCESS"
                self.state.phase = "success"
                self.state.note("Point B reached — mission SUCCESS")
                sim.stop()
                return

            if sim.collided:
                self.state.status = "FAILED"
                self.state.phase = "collided"
                self.state.note("Collision detected — demo failed")
                sim.stop()
                return

            if time.time() - start_time > TIMEOUT_S:
                self.state.status = "TIMEOUT"
                self.state.note("Demo timed out before reaching the goal")
                sim.stop()
                return
