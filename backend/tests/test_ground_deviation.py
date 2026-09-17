import cv2
import numpy as np

from app.modules.camera.demo_source import DemoFrameSource
from app.modules.depth.ground_deviation import compute_ground_deviation_mask, extract_deviation_regions
from app.modules.depth.synthetic import SyntheticRelativeDepth
from app.modules.perception.schemas import PerceptionResult


def make_perception(obstacles=(), confidence=0.8, h=200, w=200):
    return PerceptionResult(
        timestamp=0.0,
        mode="fallback",
        model_name="test",
        obstacles=list(obstacles),
        free_space_mask=np.ones((h, w), dtype=np.uint8),
        confidence=confidence,
        annotated_frame=np.zeros((h, w, 3), dtype=np.uint8),
    )


def test_flat_gradient_ground_has_no_deviation():
    h, w = 200, 200
    horizon_row = 90
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        frame[row, :] = (40, 90 + row // 4, 40)  # smooth vertical gradient, no localized anomalies
    mask = compute_ground_deviation_mask(frame, horizon_row, color_tolerance=16.0)
    assert mask.sum() == 0


def test_colored_patch_on_uniform_ground_is_flagged():
    # Patch is deliberately small relative to the (width-scaled) blur
    # kernel: the residual-from-local-blur method flags localized anomalies
    # whose extent is well under the kernel size — a much-larger patch
    # would partly blur into its own average at its center, a known,
    # documented limitation of this lightweight single-frame technique
    # (see module docstring).
    h, w = 200, 200
    horizon_row = 90
    frame = np.full((h, w, 3), (40, 100, 40), dtype=np.uint8)  # uniform green ground
    cv2.rectangle(frame, (94, 151), (102, 159), (0, 0, 220), -1)  # a small, distinctly-colored "rock"
    mask = compute_ground_deviation_mask(frame, horizon_row, color_tolerance=16.0)
    assert mask[155, 98] == 1  # inside the patch
    assert mask[150, 20] == 0  # elsewhere on uniform ground


def test_deviation_ignored_above_horizon():
    h, w = 200, 200
    horizon_row = 90
    frame = np.full((h, w, 3), (40, 100, 40), dtype=np.uint8)
    cv2.rectangle(frame, (80, 10), (110, 40), (0, 0, 220), -1)  # bright patch in the "sky" region
    mask = compute_ground_deviation_mask(frame, horizon_row, color_tolerance=16.0)
    assert mask[:horizon_row, :].sum() == 0


def test_extract_deviation_regions_returns_bbox_with_terrain_label():
    h, w = 200, 200
    horizon_row = 90
    frame = np.full((h, w, 3), (40, 100, 40), dtype=np.uint8)
    cv2.rectangle(frame, (80, 140), (110, 170), (0, 0, 220), -1)
    mask = compute_ground_deviation_mask(frame, horizon_row, color_tolerance=16.0)
    regions = extract_deviation_regions(mask, frame_area=h * w)
    assert len(regions) >= 1
    assert regions[0].label == "terrain_deviation"
    assert 0.0 <= regions[0].confidence <= 1.0


def test_synthetic_depth_populates_ground_deviation_fields():
    estimator = SyntheticRelativeDepth(horizon_fraction=0.45)
    cam = DemoFrameSource(width=320, height=240, num_obstacles=3, seed=9)
    cam.start()
    frame = cam.read_frame()
    result = estimator.estimate(frame, make_perception(h=240, w=320))

    assert result.ground_deviation_mask.shape == (240, 320)
    assert result.ground_deviation_mask.dtype == np.uint8
    assert len(result.ground_deviation_regions) == len(result.ground_deviation_relative_depth)
    # DemoFrameSource's hazard blobs are color anomalies against flat
    # sky/ground gradients -> the class-agnostic detector should catch at
    # least one, independent of any perception/YOLO/MOG2 output.
    assert len(result.ground_deviation_regions) >= 1
    for region in result.ground_deviation_regions:
        assert region.label == "terrain_deviation"
