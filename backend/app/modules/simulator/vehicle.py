"""Unicycle-model vehicle kinematics.

No motor output, no hardware I/O — this advances a pure kinematic state
(x, y, heading, velocity, angular_velocity) given a commanded velocity,
exactly like a real UGV's low-level controller would consume a "cmd_vel"
message. Swapping this for real motor control later means replacing the
`step()` body only; every module upstream (planner, decision engine) is
unaffected because they only ever see VehicleState.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class CmdVel:
    """A commanded velocity, named after the ROS 2 cmd_vel convention."""

    linear: float = 0.0  # m/s, +forward
    angular: float = 0.0  # rad/s, +counter-clockwise


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # radians, 0 = +x axis
    velocity: float = 0.0  # m/s, last applied linear velocity
    angular_velocity: float = 0.0  # rad/s, last applied angular velocity
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "velocity": self.velocity,
            "angular_velocity": self.angular_velocity,
            "timestamp": self.timestamp,
        }


class UnicycleVehicle:
    """Differential-drive / unicycle kinematic model.

    x' = x + v*cos(heading)*dt
    y' = y + v*sin(heading)*dt
    heading' = heading + omega*dt
    """

    def __init__(
        self,
        max_linear_velocity: float = 1.5,
        max_angular_velocity: float = 1.5,
        initial_state: VehicleState | None = None,
    ) -> None:
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.state = initial_state or VehicleState()

    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> VehicleState:
        self.state = VehicleState(x=x, y=y, heading=heading)
        return self.state

    def step(self, cmd: CmdVel, dt: float) -> VehicleState:
        linear = max(-self.max_linear_velocity, min(self.max_linear_velocity, cmd.linear))
        angular = max(-self.max_angular_velocity, min(self.max_angular_velocity, cmd.angular))

        x = self.state.x + linear * math.cos(self.state.heading) * dt
        y = self.state.y + linear * math.sin(self.state.heading) * dt
        heading = wrap_angle(self.state.heading + angular * dt)

        self.state = VehicleState(
            x=x,
            y=y,
            heading=heading,
            velocity=linear,
            angular_velocity=angular,
            timestamp=time.time(),
        )
        return self.state
