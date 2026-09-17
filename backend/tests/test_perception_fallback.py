import cv2
import numpy as np
import pytest

from app.modules.camera.demo_source import DemoFrameSource
from app.modules.perception.fallback import ClassicalPerception
from app.modules.perception.schemas import ObstacleDetection
from app.modules.perception.utils import compute_free_space_mask, frame_quality_score


def test_detects_obstacles_on_demo_stream_after_warmup():
    cam = DemoFrameSource(width=320, height=240, num_obstacles=2, seed=3)
    cam.start()
    perception = ClassicalPerception(warmup_frames=10)

    any_detected = False
    last_result = None
    for _ in range(40):
        frame = cam.read_frame()
        last_result = perception.process(frame)
        if last_result.obstacles:
            any_detected = True

    assert any_detected, "expected the moving demo obstacles to be detected at least once"
    assert last_result.mode == "fallback"
    assert last_result.model_name == "classical-cv-mog2"
    assert last_result.free_space_mask.shape == (240, 320)
    assert last_result.annotated_frame.shape == (240, 320, 3)
    assert 0.0 <= last_result.confidence <= 1.0


def test_static_scene_settles_to_no_detections():
    static_frame = np.full((100, 100, 3), 80, dtype=np.uint8)
    perception = ClassicalPerception(warmup_frames=5)
    result = None
    for _ in range(20):
        result = perception.process(static_frame)
    assert result.obstacles == []


def test_confidence_increases_as_background_model_warms_up():
    static_frame = np.full((100, 100, 3), 80, dtype=np.uint8)
    perception = ClassicalPerception(warmup_frames=10)
    first = perception.process(static_frame)
    for _ in range(15):
        last = perception.process(static_frame)
    assert last.confidence >= first.confidence


def test_is_ready_is_always_true():
    assert ClassicalPerception().is_ready is True
    assert ClassicalPerception().mode == "fallback"


# -- shared utils --------------------------------------------------------


def test_frame_quality_score_bounded():
    frame = np.random.default_rng(0).integers(0, 255, (50, 50, 3), dtype=np.uint8)
    score = frame_quality_score(frame)
    assert 0.0 <= score <= 1.0


def test_frame_quality_score_sharper_scores_higher_than_blurred():
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
    assert frame_quality_score(sharp) >= frame_quality_score(blurred)


def test_compute_free_space_mask_excludes_obstacle_region():
    obstacles = [ObstacleDetection(x1=10, y1=60, x2=30, y2=80, confidence=0.5, label="hazard")]
    mask = compute_free_space_mask((100, 100), obstacles, horizon_fraction=0.45, dilation_px=0)
    assert mask[70, 20] == 0  # inside obstacle bbox, below horizon -> excluded
    assert mask[70, 90] == 1  # below horizon, away from obstacle -> free
    assert mask[10, 20] == 0  # above horizon -> never free
