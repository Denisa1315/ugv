import numpy as np

from app.modules.camera.demo_source import DemoFrameSource
from app.modules.depth.factory import create_depth_estimator
from app.modules.localization.schemas import LocalizationResult, RelativePose
from app.modules.perception.factory import create_perception_module
from app.modules.risk.builder import RiskFieldBuilder
from app.modules.risk.config import RiskFieldConfig
from app.modules.risk.grid import RiskGrid
from app.modules.simulator.environment import Environment
from app.modules.simulator.vehicle import VehicleState


def make_localization(confidence=0.8):
    return LocalizationResult(
        timestamp=0.0,
        mode="orb_essential_matrix",
        pose=RelativePose(),
        confidence=confidence,
        tracked_features=100,
        matched_features=50,
        inliers=40,
        status="tracking",
    )


def test_update_produces_nonuniform_risk_from_demo_scene():
    env = Environment(width=20, height=20)
    grid = RiskGrid(env, RiskFieldConfig())
    builder = RiskFieldBuilder()

    cam = DemoFrameSource(width=320, height=240, num_obstacles=3, seed=11)
    cam.start()
    perception = create_perception_module("fallback")
    depth = create_depth_estimator()

    vehicle = VehicleState(x=10.0, y=10.0, heading=0.0)

    for _ in range(10):
        frame = cam.read_frame()
        p = perception.process(frame)
        d = depth.estimate(frame, p)
        loc = make_localization()
        builder.update(grid, vehicle, frame.shape[:2], p, d, loc, dt=0.1)

    # The demo scene's hazard blobs should have raised terrain_cost somewhere
    # above the neutral default, and risk should not be perfectly uniform.
    assert grid.terrain_cost.max() > grid.config.default_cost
    assert grid.risk.std() > 0.0
    assert grid.risk.shape == (grid.rows, grid.cols)


def test_localization_confidence_raises_localization_uncertainty_cost():
    env = Environment(width=20, height=20)
    vehicle = VehicleState(x=10.0, y=10.0, heading=0.0)
    cam = DemoFrameSource(width=320, height=240, num_obstacles=1, seed=2)
    cam.start()
    perception = create_perception_module("fallback")
    depth = create_depth_estimator()
    frame = cam.read_frame()
    p = perception.process(frame)
    d = depth.estimate(frame, p)

    grid_high_conf = RiskGrid(env, RiskFieldConfig())
    RiskFieldBuilder().update(grid_high_conf, vehicle, frame.shape[:2], p, d, make_localization(0.95), dt=0.1)

    grid_low_conf = RiskGrid(env, RiskFieldConfig())
    RiskFieldBuilder().update(grid_low_conf, vehicle, frame.shape[:2], p, d, make_localization(0.05), dt=0.1)

    assert grid_low_conf.localization_uncertainty_cost.mean() > grid_high_conf.localization_uncertainty_cost.mean()
    assert grid_low_conf.risk.mean() > grid_high_conf.risk.mean()


def test_decay_returns_unobserved_cells_toward_default_after_hazard_leaves_view():
    env = Environment(width=20, height=20)
    grid = RiskGrid(env, RiskFieldConfig(decay_per_second=1.0))  # fast decay for a short test
    builder = RiskFieldBuilder()
    vehicle = VehicleState(x=1.0, y=1.0, heading=0.0)

    # Manually spike terrain_cost to simulate a past detection.
    grid.terrain_cost[:] = 0.9

    blank_frame = np.full((240, 320, 3), 90, dtype=np.uint8)
    perception = create_perception_module("fallback")
    depth = create_depth_estimator()
    p = perception.process(blank_frame)
    d = depth.estimate(blank_frame, p)
    loc = make_localization()

    for _ in range(20):
        builder.update(grid, vehicle, blank_frame.shape[:2], p, d, loc, dt=0.5)

    assert grid.terrain_cost.mean() < 0.9
    assert grid.terrain_cost.mean() <= grid.config.default_cost + 0.05
