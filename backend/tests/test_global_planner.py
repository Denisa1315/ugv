import math

from app.modules.planning.global_planner import AStarCostConfig, AStarPlanner
from app.modules.risk.config import RiskFieldConfig, RiskWeights
from app.modules.risk.grid import RiskGrid
from app.modules.simulator.environment import Environment


def make_grid(width=10, height=10, resolution=2.0):
    env = Environment(width=width, height=height)
    config = RiskFieldConfig(
        resolution_cells_per_meter=resolution,
        weights=RiskWeights(w1_terrain=1.0, w2_obstacle=0.0, w3_depth=0.0, w4_localization=0.0),
        default_cost=0.1,
    )
    grid = RiskGrid(env, config)
    grid.terrain_cost[:] = 0.1
    grid.combine()
    return grid


def test_finds_direct_path_on_open_grid():
    grid = make_grid()
    reachable = grid.safe_mask
    planner = AStarPlanner()
    result = planner.plan(grid, reachable, start_world=(1.0, 1.0), goal_world=(8.0, 8.0))
    assert result.success is True
    assert result.reason == "OK"
    assert len(result.path_world) >= 2
    assert result.path_world[0] == grid.cell_to_world(*grid.world_to_cell(1.0, 1.0))
    goal_cell_world = grid.cell_to_world(*grid.world_to_cell(8.0, 8.0))
    assert result.path_world[-1] == goal_cell_world


def test_reports_no_safe_path_when_goal_unreachable():
    grid = make_grid()
    reachable = grid.safe_mask.copy()
    reachable[:, :] = False
    goal_row, goal_col = grid.world_to_cell(8.0, 8.0)
    start_row, start_col = grid.world_to_cell(1.0, 1.0)
    reachable[start_row, start_col] = True  # vehicle's own cell only

    planner = AStarPlanner()
    result = planner.plan(grid, reachable, start_world=(1.0, 1.0), goal_world=(8.0, 8.0))
    assert result.success is False
    assert result.reason == "NO_SAFE_PATH"
    assert result.path_world == []


def test_path_avoids_high_risk_region():
    grid = make_grid()
    # A high-risk patch directly on the straight-line route.
    for row in range(grid.rows):
        for col in range(grid.cols):
            wx, wy = grid.cell_to_world(row, col)
            if 3.5 < wx < 6.5 and 3.5 < wy < 6.5:
                grid.terrain_cost[row, col] = 1.0
    grid.combine()

    planner = AStarPlanner(AStarCostConfig(risk_weight=10.0))
    result = planner.plan(grid, grid.safe_mask, start_world=(1.0, 1.0), goal_world=(9.0, 9.0))
    assert result.success is True
    for wx, wy in result.path_world:
        assert not (3.5 < wx < 6.5 and 3.5 < wy < 6.5), "path should route around the high-risk patch"


def test_higher_risk_weight_prefers_longer_lower_risk_route():
    grid = make_grid(width=10, height=6, resolution=2.0)
    for row in range(grid.rows):
        for col in range(grid.cols):
            wx, wy = grid.cell_to_world(row, col)
            if 4.0 < wx < 6.0 and wy < 5.0:  # a moderate-risk band with a gap near the top edge
                grid.terrain_cost[row, col] = 0.6
    grid.combine()

    low_risk_weight = AStarPlanner(AStarCostConfig(risk_weight=0.1)).plan(
        grid, grid.safe_mask, (1.0, 3.0), (9.0, 3.0)
    )
    high_risk_weight = AStarPlanner(AStarCostConfig(risk_weight=20.0)).plan(
        grid, grid.safe_mask, (1.0, 3.0), (9.0, 3.0)
    )
    assert low_risk_weight.success and high_risk_weight.success
    # Penalizing risk heavily should produce a path that picks up less total
    # risk exposure than the distance-only-ish route, even if longer.
    def total_risk(path):
        return sum(grid.risk[grid.world_to_cell(x, y)] for x, y in path)

    assert total_risk(high_risk_weight.path_world) <= total_risk(low_risk_weight.path_world)
