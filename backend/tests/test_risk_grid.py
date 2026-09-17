import math

import numpy as np
import pytest

from app.modules.risk.config import RiskFieldConfig, RiskWeights
from app.modules.risk.grid import RiskGrid
from app.modules.simulator.environment import Environment


def make_grid(width=10, height=10, resolution=2.0, **config_kwargs):
    env = Environment(width=width, height=height)
    config = RiskFieldConfig(resolution_cells_per_meter=resolution, **config_kwargs)
    return RiskGrid(env, config)


def test_world_to_cell_and_back_roundtrip():
    grid = make_grid()
    row, col = grid.world_to_cell(3.3, 4.7)
    x, y = grid.cell_to_world(row, col)
    assert math.isclose(x, 3.3, abs_tol=1.0 / grid.resolution)
    assert math.isclose(y, 4.7, abs_tol=1.0 / grid.resolution)


def test_world_to_cell_clamps_out_of_bounds():
    grid = make_grid(width=10, height=10)
    row, col = grid.world_to_cell(-5, 500)
    assert 0 <= row < grid.rows
    assert 0 <= col < grid.cols


def test_splat_peaks_at_center_and_falls_off():
    grid = make_grid()
    grid.terrain_cost[:] = 0.0
    grid.splat(grid.terrain_cost, 5.0, 5.0, value=1.0, radius_m=1.0)
    row0, col0 = grid.world_to_cell(5.0, 5.0)
    assert grid.terrain_cost[row0, col0] == 1.0
    # a cell near the edge of the splat radius should be lower than center
    edge_row, edge_col = grid.world_to_cell(5.0 + 0.9, 5.0)
    assert grid.terrain_cost[edge_row, edge_col] < grid.terrain_cost[row0, col0]


def test_splat_outside_environment_bounds_is_ignored():
    grid = make_grid(width=10, height=10)
    grid.terrain_cost[:] = 0.0
    grid.splat(grid.terrain_cost, 500.0, 500.0, value=1.0, radius_m=1.0)
    assert np.all(grid.terrain_cost == 0.0)


def test_range_cost_increases_with_distance():
    grid = make_grid()
    cost = grid.range_cost_from(5.0, 5.0, growth_per_meter=0.05)
    near_row, near_col = grid.world_to_cell(5.0, 5.0)
    far_row, far_col = grid.world_to_cell(0.1, 0.1)
    assert cost[near_row, near_col] < cost[far_row, far_col]


def test_combine_applies_weights_and_threshold():
    grid = make_grid(weights=RiskWeights(w1_terrain=1.0, w2_obstacle=0.0, w3_depth=0.0, w4_localization=0.0))
    grid.terrain_cost[:] = 0.5
    grid.obstacle_cost[:] = 1.0  # should be ignored, weight=0
    grid.combine()
    assert np.allclose(grid.risk, 0.5)


def test_safe_mask_reflects_threshold():
    grid = make_grid(
        weights=RiskWeights(w1_terrain=1.0, w2_obstacle=0.0, w3_depth=0.0, w4_localization=0.0),
        safe_risk_threshold=0.4,
    )
    grid.terrain_cost[:] = 0.3
    grid.combine()
    assert np.all(grid.safe_mask)

    grid.terrain_cost[:] = 0.6
    grid.combine()
    assert not np.any(grid.safe_mask)
