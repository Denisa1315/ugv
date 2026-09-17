import math

from app.modules.planning.global_planner import AStarCostConfig, AStarPlanner
from app.modules.planning.local_planner import LocalPlanner, LocalPlannerConfig
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
    )
    grid = RiskGrid(env, config)
    grid.terrain_cost[:] = 0.1
    grid.combine()
    return grid


def test_goal_reached_stops_vehicle():
    grid = make_grid()
    planner = LocalPlanner(AStarPlanner())
    vehicle = VehicleState(x=8.9, y=9.0, heading=0.0)
    result = planner.step(grid, grid.safe_mask, vehicle, goal=(9.0, 9.0))
    assert result.goal_reached is True
    assert result.cmd_vel.linear == 0.0 and result.cmd_vel.angular == 0.0


def test_first_step_plans_and_produces_forward_motion_toward_goal():
    grid = make_grid()
    planner = LocalPlanner(AStarPlanner())
    vehicle = VehicleState(x=1.0, y=1.0, heading=0.0)  # facing goal direction
    result = planner.step(grid, grid.safe_mask, vehicle, goal=(9.0, 9.0))
    assert result.replanned is True
    assert result.blocked is False
    assert result.active_path
    assert result.cmd_vel.linear > 0.0


def test_no_safe_path_reports_blocked_and_zero_cmd_vel():
    grid = make_grid()
    unreachable = grid.safe_mask.copy()
    unreachable[:, :] = False
    start_row, start_col = grid.world_to_cell(1.0, 1.0)
    unreachable[start_row, start_col] = True

    planner = LocalPlanner(AStarPlanner())
    vehicle = VehicleState(x=1.0, y=1.0, heading=0.0)
    result = planner.step(grid, unreachable, vehicle, goal=(9.0, 9.0))
    assert result.blocked is True
    assert result.reason == "NO_SAFE_PATH"
    assert result.cmd_vel.linear == 0.0 and result.cmd_vel.angular == 0.0


def test_replans_in_real_time_when_new_obstacle_blocks_planned_path():
    """The core 1j requirement: an obstacle appearing on the already-planned
    path must invalidate that segment and trigger an immediate replan
    within the same step() call — no stale path is ever driven through it."""
    grid = make_grid(width=10, height=4, resolution=2.0)
    planner = LocalPlanner(
        AStarPlanner(AStarCostConfig(risk_weight=5.0)),
        LocalPlannerConfig(path_invalidate_risk_threshold=0.5, replan_check_lookahead_m=10.0),
    )
    vehicle = VehicleState(x=1.0, y=2.0, heading=0.0)
    goal = (9.0, 2.0)

    first = planner.step(grid, grid.safe_mask, vehicle, goal)
    assert first.replanned is True
    assert first.blocked is False
    original_path = list(first.active_path)
    assert original_path, "expected an initial straight-ish path across the open grid"

    # Now block the middle of that exact path with a fresh high-risk detection.
    mid_x, mid_y = original_path[len(original_path) // 2]
    row, col = grid.world_to_cell(mid_x, mid_y)
    grid.terrain_cost[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2] = 1.0
    grid.combine()

    second = planner.step(grid, grid.safe_mask, vehicle, goal)
    assert second.replanned is True, "a newly-risky segment on the active path must trigger an immediate replan"
    assert second.blocked is False
    assert second.active_path != original_path
    for wx, wy in second.active_path:
        r, c = grid.world_to_cell(wx, wy)
        assert grid.risk[r, c] < 0.5, "replanned path must route around the newly-blocked cells"
    assert planner.replan_count == 2


def test_no_replan_when_path_stays_clear():
    grid = make_grid()
    planner = LocalPlanner(AStarPlanner())
    vehicle = VehicleState(x=1.0, y=1.0, heading=0.0)
    planner.step(grid, grid.safe_mask, vehicle, goal=(9.0, 9.0))
    assert planner.replan_count == 1

    result = planner.step(grid, grid.safe_mask, vehicle, goal=(9.0, 9.0))
    assert result.replanned is False
    assert planner.replan_count == 1


def test_angular_command_steers_toward_offaxis_target():
    grid = make_grid()
    planner = LocalPlanner(AStarPlanner())
    vehicle = VehicleState(x=1.0, y=1.0, heading=math.pi / 2)  # facing +y, goal is ahead+right
    result = planner.step(grid, grid.safe_mask, vehicle, goal=(9.0, 9.0))
    assert result.cmd_vel.angular != 0.0
