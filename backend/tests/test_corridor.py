import numpy as np

from app.modules.corridor.generator import SafeCorridorGenerator
from app.modules.corridor.types import CorridorConfig
from app.modules.risk.config import RiskFieldConfig, RiskWeights
from app.modules.risk.grid import RiskGrid
from app.modules.simulator.environment import Environment
from app.modules.simulator.vehicle import VehicleState


def make_grid(width=10, height=10, resolution=2.0):
    env = Environment(width=width, height=height)
    config = RiskFieldConfig(
        resolution_cells_per_meter=resolution,
        weights=RiskWeights(w1_terrain=1.0, w2_obstacle=0.0, w3_depth=0.0, w4_localization=0.0),
        default_cost=0.1,
        safe_risk_threshold=0.35,
    )
    grid = RiskGrid(env, config)
    grid.terrain_cost[:] = 0.1
    grid.combine()
    return grid


def test_open_grid_is_fully_reachable_and_goal_reachable():
    grid = make_grid()
    vehicle = VehicleState(x=1.0, y=1.0, heading=0.0)
    generator = SafeCorridorGenerator()
    result = generator.generate(grid, vehicle, goal=(9.0, 9.0))
    assert result.goal_reachable is True
    assert result.reachable_mask.sum() == grid.rows * grid.cols
    assert result.target_waypoint is not None


def test_wall_blocks_goal_reachability():
    grid = make_grid()
    # A full-height high-risk wall between the vehicle and the goal.
    wall_col = grid.cols // 2
    grid.terrain_cost[:, wall_col] = 1.0
    grid.combine()

    vehicle = VehicleState(x=1.0, y=5.0, heading=0.0)
    generator = SafeCorridorGenerator()
    result = generator.generate(grid, vehicle, goal=(9.0, 5.0))
    assert result.goal_reachable is False


def test_gap_in_wall_restores_reachability():
    grid = make_grid()
    wall_col = grid.cols // 2
    grid.terrain_cost[:, wall_col] = 1.0
    gap_row = 2
    grid.terrain_cost[gap_row, wall_col] = 0.1  # a passable gap
    grid.combine()

    vehicle = VehicleState(x=1.0, y=5.0, heading=0.0)
    generator = SafeCorridorGenerator()
    result = generator.generate(grid, vehicle, goal=(9.0, 5.0))
    assert result.goal_reachable is True


def test_clearance_is_lower_near_unsafe_cells():
    grid = make_grid()
    blocked_col = grid.cols - 2
    grid.terrain_cost[:, blocked_col] = 1.0
    grid.combine()

    generator = SafeCorridorGenerator()
    result = generator.generate(grid, VehicleState(x=1.0, y=5.0), goal=(9.0, 5.0))

    near_wall_row, near_wall_col = grid.world_to_cell(8.5, 5.0)
    far_from_wall_row, far_from_wall_col = grid.world_to_cell(1.0, 5.0)
    assert result.clearance_m[near_wall_row, near_wall_col] < result.clearance_m[far_from_wall_row, far_from_wall_col]


def test_target_waypoint_biases_toward_goal_direction():
    grid = make_grid()
    generator = SafeCorridorGenerator(CorridorConfig(distance_weight=5.0, risk_weight=1.0, clearance_weight=0.1, heading_weight=0.1))
    vehicle = VehicleState(x=5.0, y=5.0, heading=0.0)
    result = generator.generate(grid, vehicle, goal=(9.5, 5.0))
    assert result.target_waypoint is not None
    # a goal-seeking waypoint should move roughly toward +x from the vehicle
    assert result.target_waypoint[0] > vehicle.x
