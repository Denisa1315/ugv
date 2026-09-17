"""A* over the risk grid, restricted to the safe corridor's reachable mask.

cost(step) = distance_weight*distance_cost + risk_weight*risk_cost
           + turning_weight*turning_cost

All three weights live in AStarCostConfig — not hardcoded — so the
distance/risk/smoothness tradeoff is tunable without touching the search
code. The heuristic (Euclidean distance * distance_weight) is not strictly
admissible once risk/turning costs are added — a documented, practical
compromise typical of hackathon-grade planners, not a claim of optimality.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from app.modules.risk.grid import RiskGrid

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class AStarCostConfig:
    distance_weight: float = 1.0
    risk_weight: float = 4.0
    turning_weight: float = 0.5
    max_traversable_risk: float = 0.95  # cells at/above this are always blocked, even if "reachable"


@dataclass
class GlobalPlanResult:
    success: bool
    path_world: list[tuple[float, float]] = field(default_factory=list)
    reason: str = "OK"  # "OK" | "NO_SAFE_PATH"
    total_cost: float = math.inf


class AStarPlanner:
    def __init__(self, config: AStarCostConfig | None = None) -> None:
        self.config = config or AStarCostConfig()

    def plan(
        self,
        grid: RiskGrid,
        reachable_mask: np.ndarray,
        start_world: tuple[float, float],
        goal_world: tuple[float, float],
    ) -> GlobalPlanResult:
        cfg = self.config
        start = grid.world_to_cell(*start_world)
        goal = grid.world_to_cell(*goal_world)

        if not reachable_mask[goal]:
            return GlobalPlanResult(success=False, reason="NO_SAFE_PATH")

        cell_size = 1.0 / grid.resolution

        def blocked(cell: tuple[int, int]) -> bool:
            return grid.risk[cell] >= cfg.max_traversable_risk or not reachable_mask[cell]

        # The start cell is never treated as blocked even if it reads
        # unsafe/unreachable (e.g. a transient risk spike right under the
        # vehicle) — the search must be able to leave from where the
        # vehicle already is. See the `neighbor != start` check below.

        def heuristic(cell: tuple[int, int]) -> float:
            dr = cell[0] - goal[0]
            dc = cell[1] - goal[1]
            return cfg.distance_weight * cell_size * math.hypot(dr, dc)

        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        incoming_dir: dict[tuple[int, int], tuple[int, int]] = {start: (0, 0)}
        visited: set[tuple[int, int]] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return GlobalPlanResult(
                    success=True,
                    path_world=self._reconstruct_path(grid, came_from, current, start),
                    reason="OK",
                    total_cost=g_score[current],
                )

            for dr, dc in _NEIGHBOR_OFFSETS:
                neighbor = (current[0] + dr, current[1] + dc)
                if not (0 <= neighbor[0] < grid.rows and 0 <= neighbor[1] < grid.cols):
                    continue
                if neighbor != start and blocked(neighbor):
                    continue

                step_len = cell_size * math.hypot(dr, dc)
                distance_cost = cfg.distance_weight * step_len
                risk_cost = cfg.risk_weight * float(grid.risk[neighbor]) * step_len

                prev_dir = incoming_dir[current]
                turning_cost = 0.0
                if prev_dir != (0, 0):
                    prev_angle = math.atan2(prev_dir[0], prev_dir[1])
                    new_angle = math.atan2(dr, dc)
                    turning_cost = cfg.turning_weight * abs(_wrap_angle(new_angle - prev_angle))

                tentative_g = g_score[current] + distance_cost + risk_cost + turning_cost
                if tentative_g < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    incoming_dir[neighbor] = (dr, dc)
                    heapq.heappush(open_heap, (tentative_g + heuristic(neighbor), neighbor))

        return GlobalPlanResult(success=False, reason="NO_SAFE_PATH")

    @staticmethod
    def _reconstruct_path(grid: RiskGrid, came_from, current, start) -> list[tuple[float, float]]:
        cells = [current]
        while cells[-1] != start:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        return [grid.cell_to_world(r, c) for r, c in cells]


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
