import math

import numpy as np
import pytest

from app.modules.geometry import CameraGeometryConfig, image_point_to_world, world_point_to_image
from app.modules.simulator.vehicle import VehicleState


def make_depth_map(h, w, geometry):
    horizon_row = int(h * geometry.horizon_fraction)
    depth_map = np.ones((h, w), dtype=np.float32)
    span = max(1, h - 1 - horizon_row)
    rows = np.arange(horizon_row, h)
    t = (rows - horizon_row) / span
    depth_map[horizon_row:, :] = (1.0 - t)[:, None]
    return depth_map


@pytest.mark.parametrize(
    "vehicle,world_point",
    [
        (VehicleState(x=0, y=0, heading=0), (5.0, 0.0)),  # straight ahead
        (VehicleState(x=0, y=0, heading=0), (5.0, 1.5)),  # ahead + left
        (VehicleState(x=0, y=0, heading=0), (5.0, -1.5)),  # ahead + right
        (VehicleState(x=3, y=2, heading=math.pi / 2), (3.0, 7.0)),  # rotated heading
        (VehicleState(x=-1, y=-1, heading=math.pi), (-6.0, -1.0)),  # facing -x
    ],
)
def test_world_to_image_to_world_roundtrip(vehicle, world_point):
    geometry = CameraGeometryConfig()
    h, w = 480, 640
    depth_map = make_depth_map(h, w, geometry)

    projected = world_point_to_image(world_point[0], world_point[1], (h, w), vehicle, geometry)
    assert projected is not None, "test point should be within FOV/range for this setup"
    px, py, _ = projected

    recovered = image_point_to_world(px, py, (h, w), vehicle, geometry, depth_map)
    # Row quantization (integer pixel rows) limits precision; loose tolerance is expected.
    assert math.isclose(recovered[0], world_point[0], abs_tol=0.5)
    assert math.isclose(recovered[1], world_point[1], abs_tol=0.5)


def test_point_behind_vehicle_is_culled():
    geometry = CameraGeometryConfig()
    vehicle = VehicleState(x=0, y=0, heading=0)
    assert world_point_to_image(-5.0, 0.0, (480, 640), vehicle, geometry) is None


def test_point_outside_fov_is_culled():
    geometry = CameraGeometryConfig(horizontal_fov_deg=60)
    vehicle = VehicleState(x=0, y=0, heading=0)
    # far to the side relative to a narrow FOV
    assert world_point_to_image(1.0, 10.0, (480, 640), vehicle, geometry) is None


def test_straight_ahead_projects_to_center_column():
    geometry = CameraGeometryConfig()
    vehicle = VehicleState(x=0, y=0, heading=0)
    projected = world_point_to_image(5.0, 0.0, (480, 640), vehicle, geometry)
    assert projected is not None
    px, py, forward = projected
    assert abs(px - 320) <= 1
    assert math.isclose(forward, 5.0, abs_tol=1e-6)


def test_relative_depth_meters_roundtrip():
    geometry = CameraGeometryConfig(near_m=0.5, far_m=15.0)
    for rd in (0.0, 0.25, 0.5, 0.75, 1.0):
        meters = geometry.relative_depth_to_meters(rd)
        back = geometry.meters_to_relative_depth(meters)
        assert math.isclose(back, rd, abs_tol=1e-6)
