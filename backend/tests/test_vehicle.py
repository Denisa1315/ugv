import math

from app.modules.simulator.vehicle import CmdVel, UnicycleVehicle, wrap_angle


def test_straight_line_motion():
    vehicle = UnicycleVehicle()
    vehicle.reset(x=0.0, y=0.0, heading=0.0)
    state = vehicle.step(CmdVel(linear=1.0, angular=0.0), dt=1.0)
    assert math.isclose(state.x, 1.0, abs_tol=1e-9)
    assert math.isclose(state.y, 0.0, abs_tol=1e-9)
    assert math.isclose(state.heading, 0.0, abs_tol=1e-9)


def test_pure_rotation():
    vehicle = UnicycleVehicle(max_angular_velocity=10.0)
    vehicle.reset(x=0.0, y=0.0, heading=0.0)
    state = vehicle.step(CmdVel(linear=0.0, angular=math.pi / 2), dt=1.0)
    assert math.isclose(state.heading, math.pi / 2, abs_tol=1e-9)
    assert math.isclose(state.x, 0.0, abs_tol=1e-9)
    assert math.isclose(state.y, 0.0, abs_tol=1e-9)


def test_circular_arc_returns_near_start():
    """Constant linear + angular velocity traces a circle; a full revolution
    (angular*T = 2*pi) should return close to the starting point."""
    vehicle = UnicycleVehicle(max_linear_velocity=10, max_angular_velocity=10)
    vehicle.reset(x=0.0, y=0.0, heading=0.0)
    dt = 0.001
    angular = 1.0
    steps = int(round(2 * math.pi / angular / dt))
    for _ in range(steps):
        vehicle.step(CmdVel(linear=1.0, angular=angular), dt)
    assert math.isclose(vehicle.state.x, 0.0, abs_tol=0.05)
    assert math.isclose(vehicle.state.y, 0.0, abs_tol=0.05)


def test_velocity_is_clamped_to_limits():
    vehicle = UnicycleVehicle(max_linear_velocity=1.0, max_angular_velocity=1.0)
    vehicle.reset()
    state = vehicle.step(CmdVel(linear=100.0, angular=-100.0), dt=1.0)
    assert state.velocity == 1.0
    assert state.angular_velocity == -1.0


def test_wrap_angle_stays_in_range():
    assert math.isclose(wrap_angle(3 * math.pi), math.pi, abs_tol=1e-9) or math.isclose(
        wrap_angle(3 * math.pi), -math.pi, abs_tol=1e-9
    )
    assert -math.pi < wrap_angle(10.0) <= math.pi
