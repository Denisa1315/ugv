import numpy as np
import pytest

from app.modules.depth.factory import create_depth_estimator
from app.modules.depth.synthetic import SyntheticRelativeDepth
from app.modules.perception.schemas import ObstacleDetection, PerceptionResult


def make_perception(obstacles, confidence=0.8):
    h, w = 100, 100
    return PerceptionResult(
        timestamp=0.0,
        mode="fallback",
        model_name="test",
        obstacles=obstacles,
        free_space_mask=np.ones((h, w), dtype=np.uint8),
        confidence=confidence,
        annotated_frame=np.zeros((h, w, 3), dtype=np.uint8),
    )


def test_depth_map_is_near_zero_at_bottom_and_near_one_at_horizon():
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    estimator = SyntheticRelativeDepth(horizon_fraction=0.5)
    result = estimator.estimate(frame, make_perception([]))

    assert result.relative_depth_map.shape == (100, 100)
    assert result.relative_depth_map[99, 50] < 0.05  # bottom row: near
    assert result.relative_depth_map[50, 50] > 0.95  # horizon row: far


def test_sky_region_is_constant_far():
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    estimator = SyntheticRelativeDepth(horizon_fraction=0.5)
    result = estimator.estimate(frame, make_perception([]))
    assert np.all(result.relative_depth_map[:50, :] == 1.0)


def test_depth_map_is_monotonic_with_row():
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    estimator = SyntheticRelativeDepth(horizon_fraction=0.4)
    result = estimator.estimate(frame, make_perception([]))
    column = result.relative_depth_map[40:, 50]
    assert np.all(np.diff(column) <= 0)  # depth strictly non-increasing toward bottom (near)


def test_obstacle_near_bottom_is_close_and_near_horizon_is_far():
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    obstacles = [
        ObstacleDetection(x1=10, y1=90, x2=20, y2=98, confidence=0.5, label="hazard"),  # near bottom
        ObstacleDetection(x1=60, y1=45, x2=70, y2=52, confidence=0.5, label="hazard"),  # near horizon
    ]
    estimator = SyntheticRelativeDepth(horizon_fraction=0.45)
    result = estimator.estimate(frame, make_perception(obstacles))

    assert len(result.obstacle_relative_depth) == 2
    near_depth, far_depth = result.obstacle_relative_depth
    assert near_depth < far_depth


def test_confidence_combines_perception_confidence_and_frame_quality():
    sharp_frame = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    estimator = SyntheticRelativeDepth()

    high = estimator.estimate(sharp_frame, make_perception([], confidence=1.0))
    low = estimator.estimate(sharp_frame, make_perception([], confidence=0.0))
    assert high.confidence > low.confidence
    assert 0.0 <= high.confidence <= 1.0
    assert 0.0 <= low.confidence <= 1.0


def test_factory_default_and_invalid_mode():
    estimator = create_depth_estimator()
    assert isinstance(estimator, SyntheticRelativeDepth)
    assert estimator.mode == "synthetic"
    with pytest.raises(ValueError):
        create_depth_estimator("real")  # not implemented in this MVP
