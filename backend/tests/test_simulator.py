import math

from app.modules.simulator.environment import Environment
from app.modules.simulator.renderer import render_topdown
from app.modules.simulator.simulator import SimulationStatus, Simulator


def make_sim():
    env = Environment(width=20, height=20, goal=(5.0, 0.0))
    return Simulator(environment=env, tick_hz=10.0)


def test_tick_is_noop_when_not_running():
    sim = make_sim()
    sim.reset(x=0, y=0, heading=0)
    sim.set_cmd_vel(1.0, 0.0)
    sim.tick(1.0)
    assert sim.vehicle.state.x == 0.0  # still IDLE, no movement


def test_start_moves_vehicle_on_tick():
    sim = make_sim()
    sim.reset(x=0, y=0, heading=0)
    sim.start()
    sim.set_cmd_vel(1.0, 0.0)
    sim.tick(1.0)
    assert sim.vehicle.state.x == 1.0
    assert sim.tick_count == 1
    assert len(sim.trajectory) == 2  # initial position + one tick


def test_pause_stops_movement_but_keeps_state():
    sim = make_sim()
    sim.reset(x=0, y=0, heading=0)
    sim.start()
    sim.set_cmd_vel(1.0, 0.0)
    sim.tick(1.0)
    sim.pause()
    sim.tick(1.0)
    assert sim.status == SimulationStatus.PAUSED
    assert sim.vehicle.state.x == 1.0  # unchanged after pause


def test_goal_reached_flag():
    sim = make_sim()
    sim.reset(x=0, y=0, heading=0)
    sim.start()
    sim.set_cmd_vel(1.0, 0.0)
    for _ in range(60):
        sim.tick(0.1)
        if sim.goal_reached:
            break
    assert sim.goal_reached is True
    assert math.isclose(sim.vehicle.state.x, 5.0, abs_tol=0.5)


def test_collision_flag():
    sim = make_sim()
    sim.environment.add_obstacle(2.0, 0.0, radius=0.3)
    sim.reset(x=0, y=0, heading=0)
    sim.start()
    sim.set_cmd_vel(1.0, 0.0)
    for _ in range(30):
        sim.tick(0.1)
        if sim.collided:
            break
    assert sim.collided is True


def test_reset_clears_flags_and_trajectory():
    sim = make_sim()
    sim.start()
    sim.set_cmd_vel(1.0, 0.0)
    sim.tick(1.0)
    sim.reset(x=2.0, y=3.0, heading=0.0)
    assert sim.status == SimulationStatus.IDLE
    assert sim.vehicle.state.x == 2.0
    assert sim.vehicle.state.y == 3.0
    assert len(sim.trajectory) == 1
    assert sim.goal_reached is False
    assert sim.collided is False


def test_render_topdown_produces_image():
    sim = make_sim()
    sim.start()
    sim.set_cmd_vel(1.0, 0.2)
    for _ in range(5):
        sim.tick(0.1)
    img = render_topdown(sim)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype.name == "uint8"
