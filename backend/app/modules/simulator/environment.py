"""2D outdoor environment: bounds, static obstacles, goal.

Deliberately minimal — this is the world the vehicle and (later) the risk
field / planners reason about. Units are meters, origin at (0, 0).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field


@dataclass
class Obstacle:
    id: str
    x: float
    y: float
    radius: float = 0.5


@dataclass
class Environment:
    width: float = 20.0
    height: float = 20.0
    goal: tuple[float, float] = (9.0, 9.0)
    obstacles: list[Obstacle] = field(default_factory=list)

    def is_within_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def add_obstacle(self, x: float, y: float, radius: float = 0.5) -> Obstacle:
        obstacle = Obstacle(id=str(uuid.uuid4())[:8], x=x, y=y, radius=radius)
        self.obstacles.append(obstacle)
        return obstacle

    def remove_obstacle(self, obstacle_id: str) -> bool:
        before = len(self.obstacles)
        self.obstacles = [o for o in self.obstacles if o.id != obstacle_id]
        return len(self.obstacles) < before

    def clear_obstacles(self) -> None:
        self.obstacles.clear()

    def set_goal(self, x: float, y: float) -> None:
        self.goal = (x, y)

    def distance_to_goal(self, x: float, y: float) -> float:
        gx, gy = self.goal
        return math.hypot(gx - x, gy - y)

    def nearest_obstacle_distance(self, x: float, y: float) -> float | None:
        """Distance from (x, y) to the nearest obstacle surface, or None if empty."""
        if not self.obstacles:
            return None
        return min(math.hypot(o.x - x, o.y - y) - o.radius for o in self.obstacles)
