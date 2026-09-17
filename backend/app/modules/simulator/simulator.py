"""Simulation orchestrator: owns one vehicle + one environment, ticks them
forward in real time, and exposes Start/Pause/Stop/Reset control.

`tick(dt)` is a pure, synchronous step — it contains all the actual state
transition logic and is what unit tests call directly. `run_forever()` is a
thin real-time wrapper around it (used by the FastAPI background task) so
the dashboard can watch the vehicle move continuously.

cmd_vel is set directly via `set_cmd_vel()` for now (manual/test control).
From phase 1j onward, the local planner writes into the same slot every
cycle instead — nothing else here changes.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from enum import Enum
from typing import Callable

from app.modules.simulator.environment import Environment
from app.modules.simulator.vehicle import CmdVel, UnicycleVehicle, VehicleState


class SimulationStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class Simulator:
    def __init__(
        self,
        environment: Environment | None = None,
        vehicle: UnicycleVehicle | None = None,
        tick_hz: float = 10.0,
        trajectory_length: int = 1000,
        goal_tolerance: float = 0.4,
    ) -> None:
        self.environment = environment or Environment()
        self.vehicle = vehicle or UnicycleVehicle()
        self.tick_hz = tick_hz
        self.goal_tolerance = goal_tolerance
        self.status = SimulationStatus.IDLE
        self.cmd_vel = CmdVel()
        self.trajectory: deque[tuple[float, float]] = deque(maxlen=trajectory_length)
        self.elapsed_time = 0.0
        self.tick_count = 0
        self.goal_reached = False
        self.collided = False
        self._loop_task: asyncio.Task | None = None

        # Optional hook run at the start of every RUNNING tick, before
        # kinematics are applied — this is how the navigation pipeline
        # (camera -> perception -> ... -> local planner) drives cmd_vel
        # autonomously each cycle. None (the default) preserves plain
        # manual/test control exactly as in phase 1b.
        self.pre_tick_hook: Callable[[float], None] | None = None

        self.trajectory.append((self.vehicle.state.x, self.vehicle.state.y))

    # -- control -----------------------------------------------------
    def start(self) -> None:
        if self.status == SimulationStatus.STOPPED:
            self.status = SimulationStatus.RUNNING
        elif self.status in (SimulationStatus.IDLE, SimulationStatus.PAUSED):
            self.status = SimulationStatus.RUNNING

    def pause(self) -> None:
        if self.status == SimulationStatus.RUNNING:
            self.status = SimulationStatus.PAUSED

    def stop(self) -> None:
        self.status = SimulationStatus.STOPPED
        self.cmd_vel = CmdVel()

    def reset(self, x: float = 1.0, y: float = 1.0, heading: float = 0.0) -> None:
        self.vehicle.reset(x=x, y=y, heading=heading)
        self.cmd_vel = CmdVel()
        self.trajectory.clear()
        self.trajectory.append((x, y))
        self.elapsed_time = 0.0
        self.tick_count = 0
        self.goal_reached = False
        self.collided = False
        self.status = SimulationStatus.IDLE

    def set_cmd_vel(self, linear: float, angular: float) -> None:
        self.cmd_vel = CmdVel(linear=linear, angular=angular)

    # -- stepping ------------------------------------------------------
    def tick(self, dt: float) -> VehicleState:
        """Advance the simulation by dt seconds. No-op unless RUNNING."""
        if self.status != SimulationStatus.RUNNING:
            return self.vehicle.state

        if self.pre_tick_hook is not None:
            self.pre_tick_hook(dt)

        state = self.vehicle.step(self.cmd_vel, dt)
        self.trajectory.append((state.x, state.y))
        self.elapsed_time += dt
        self.tick_count += 1

        if self.environment.distance_to_goal(state.x, state.y) <= self.goal_tolerance:
            self.goal_reached = True

        nearest = self.environment.nearest_obstacle_distance(state.x, state.y)
        if nearest is not None and nearest <= 0.0:
            self.collided = True

        return state

    async def run_forever(self) -> None:
        """Real-time loop driving tick() at ~tick_hz. Runs until STOPPED."""
        dt = 1.0 / self.tick_hz
        while self.status != SimulationStatus.STOPPED:
            loop_start = time.monotonic()
            self.tick(dt)
            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.0, dt - elapsed))

    def ensure_running_task(self) -> None:
        """Start the background real-time loop if it isn't already alive."""
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self.run_forever())

    # -- state ---------------------------------------------------------
    def get_state(self) -> dict:
        return {
            "status": self.status.value,
            "vehicle": self.vehicle.state.as_dict(),
            "cmd_vel": {"linear": self.cmd_vel.linear, "angular": self.cmd_vel.angular},
            "environment": {
                "width": self.environment.width,
                "height": self.environment.height,
                "goal": self.environment.goal,
                "obstacles": [
                    {"id": o.id, "x": o.x, "y": o.y, "radius": o.radius}
                    for o in self.environment.obstacles
                ],
            },
            "trajectory": list(self.trajectory),
            "elapsed_time": self.elapsed_time,
            "tick_count": self.tick_count,
            "goal_reached": self.goal_reached,
            "collided": self.collided,
        }
