"""World-frame risk grid: the map the corridor generator and planners
search over. Cells are square, `resolution_cells_per_meter` per side,
covering the Simulator's Environment bounds.
"""

from __future__ import annotations

import math

import numpy as np

from app.modules.risk.config import RiskFieldConfig
from app.modules.simulator.environment import Environment


class RiskGrid:
    def __init__(self, environment: Environment, config: RiskFieldConfig) -> None:
        self.environment = environment
        self.config = config
        self.resolution = config.resolution_cells_per_meter
        self.rows = max(1, int(round(environment.height * self.resolution)))
        self.cols = max(1, int(round(environment.width * self.resolution)))

        self.terrain_cost = np.full((self.rows, self.cols), config.default_cost, dtype=np.float32)
        self.obstacle_cost = np.full((self.rows, self.cols), config.default_cost, dtype=np.float32)
        self.depth_cost = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.localization_uncertainty_cost = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.risk = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.safe_mask = np.ones((self.rows, self.cols), dtype=bool)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(x * self.resolution, 0, self.cols - 1))
        row = int(np.clip(y * self.resolution, 0, self.rows - 1))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = (col + 0.5) / self.resolution
        y = (row + 0.5) / self.resolution
        return x, y

    def splat(self, layer: np.ndarray, wx: float, wy: float, value: float, radius_m: float) -> None:
        """Stamp `value` (with linear falloff) into `layer` around a world point."""
        if not self.environment.is_within_bounds(wx, wy):
            return
        row0, col0 = self.world_to_cell(wx, wy)
        radius_cells = max(1, int(round(radius_m * self.resolution)))
        r0, r1 = max(0, row0 - radius_cells), min(self.rows, row0 + radius_cells + 1)
        c0, c1 = max(0, col0 - radius_cells), min(self.cols, col0 + radius_cells + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                dist_cells = math.hypot(r - row0, c - col0)
                if dist_cells > radius_cells:
                    continue
                falloff = 1.0 - dist_cells / max(1, radius_cells)
                layer[r, c] = max(layer[r, c], value * falloff)

    def range_cost_from(self, x: float, y: float, growth_per_meter: float) -> np.ndarray:
        rows_idx, cols_idx = np.indices((self.rows, self.cols))
        world_x = (cols_idx + 0.5) / self.resolution
        world_y = (rows_idx + 0.5) / self.resolution
        dist = np.hypot(world_x - x, world_y - y)
        return np.clip(dist * growth_per_meter, 0.0, 1.0).astype(np.float32)

    def combine(self) -> None:
        w = self.config.weights
        self.risk = np.clip(
            w.w1_terrain * self.terrain_cost
            + w.w2_obstacle * self.obstacle_cost
            + w.w3_depth * self.depth_cost
            + w.w4_localization * self.localization_uncertainty_cost,
            0.0,
            1.0,
        ).astype(np.float32)
        self.safe_mask = self.risk <= self.config.safe_risk_threshold
