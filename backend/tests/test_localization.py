import cv2
import numpy as np
import pytest

from app.modules.camera.demo_source import DemoFrameSource
from app.modules.geometry import CameraGeometryConfig
from app.modules.localization.orb_localizer import OrbFeatureLocalizer


def _camera_matrix(geometry: CameraGeometryConfig, w: int, h: int) -> np.ndarray:
    f = geometry.focal_px(w)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def _render_synthetic_view(points_3d, colors, K, R, t, image_size, marker_radius=7):
    """Render a set of 3D points (in the reference camera's frame) as
    textured markers from a camera at pose (R, t) relative to that frame —
    a genuine, ground-truth two-view scenario (real parallax), not an image
    warp, so essential-matrix recovery is exactly checkable."""
    h, w = image_size
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    for P, color in zip(points_3d, colors):
        cam_pt = R @ P + t
        if cam_pt[2] <= 0.1:
            continue
        uv = K @ (cam_pt / cam_pt[2])
        u, v = int(uv[0]), int(uv[1])
        if marker_radius <= u < w - marker_radius and marker_radius <= v < h - marker_radius:
            cv2.circle(img, (u, v), marker_radius, color, -1)
            cv2.circle(img, (u, v), marker_radius, (0, 0, 0), 2)
    return img


def _yaw_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _make_scene(rng, n_points=70):
    points = np.stack(
        [rng.uniform(-2.5, 2.5, n_points), rng.uniform(-1.5, 1.5, n_points), rng.uniform(3.0, 10.0, n_points)],
        axis=1,
    )
    colors = [tuple(int(c) for c in rng.integers(50, 255, 3)) for _ in range(n_points)]
    return points, colors


def test_initializing_on_first_frame():
    rng = np.random.default_rng(0)
    points, colors = _make_scene(rng)
    geometry = CameraGeometryConfig()
    K = _camera_matrix(geometry, 640, 480)
    frame = _render_synthetic_view(points, colors, K, np.eye(3), np.zeros(3), (480, 640))

    localizer = OrbFeatureLocalizer(geometry=geometry)
    result = localizer.track(frame, dt=0.1)
    assert result.status == "initializing"
    assert result.pose.x == 0.0 and result.pose.y == 0.0 and result.pose.heading == 0.0


def test_textureless_frame_never_crashes_and_reports_low_confidence():
    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    localizer = OrbFeatureLocalizer()
    r1 = localizer.track(blank, dt=0.1)
    r2 = localizer.track(blank, dt=0.1)
    assert r1.status == "initializing"
    assert r2.status in ("initializing", "lost")
    assert r2.confidence == 0.0


def test_recovers_forward_translation_and_yaw_sign():
    geometry = CameraGeometryConfig()
    K = _camera_matrix(geometry, 640, 480)
    rng = np.random.default_rng(1)
    points, colors = _make_scene(rng)

    theta = 0.15  # ~8.6 degrees yaw
    R2 = _yaw_matrix(theta)
    t2 = np.array([0.05, 0.0, 0.3])  # mostly-forward translation + small lateral, nonzero baseline

    frame1 = _render_synthetic_view(points, colors, K, np.eye(3), np.zeros(3), (480, 640))
    frame2 = _render_synthetic_view(points, colors, K, R2, t2, (480, 640))

    localizer = OrbFeatureLocalizer(geometry=geometry, assumed_forward_speed=1.0)
    localizer.track(frame1, dt=1.0)
    result = localizer.track(frame2, dt=1.0)

    assert result.status == "tracking"
    assert result.matched_features >= 15
    assert result.inliers >= 10
    assert result.confidence > 0.2
    # Forward motion -> positive local x; recovered yaw should share sign(theta).
    assert result.pose.x > 0
    assert result.pose.heading > 0


def test_yaw_sign_flips_with_opposite_rotation():
    geometry = CameraGeometryConfig()
    K = _camera_matrix(geometry, 640, 480)
    rng = np.random.default_rng(2)
    points, colors = _make_scene(rng)
    t2 = np.array([0.0, 0.0, 0.3])

    def run(theta):
        frame1 = _render_synthetic_view(points, colors, K, np.eye(3), np.zeros(3), (480, 640))
        frame2 = _render_synthetic_view(points, colors, K, _yaw_matrix(theta), t2, (480, 640))
        localizer = OrbFeatureLocalizer(geometry=geometry)
        localizer.track(frame1, dt=1.0)
        return localizer.track(frame2, dt=1.0)

    result_pos = run(0.15)
    result_neg = run(-0.15)
    assert result_pos.status == "tracking" and result_neg.status == "tracking"
    assert result_pos.pose.heading > 0 > result_neg.pose.heading


def test_reset_clears_accumulated_pose_and_tracking_state():
    geometry = CameraGeometryConfig()
    K = _camera_matrix(geometry, 640, 480)
    rng = np.random.default_rng(3)
    points, colors = _make_scene(rng)
    frame1 = _render_synthetic_view(points, colors, K, np.eye(3), np.zeros(3), (480, 640))
    frame2 = _render_synthetic_view(points, colors, K, _yaw_matrix(0.1), np.array([0, 0, 0.3]), (480, 640))

    localizer = OrbFeatureLocalizer(geometry=geometry)
    localizer.track(frame1, dt=1.0)
    localizer.track(frame2, dt=1.0)
    localizer.reset()

    result = localizer.track(frame1, dt=1.0)
    assert result.status == "initializing"
    assert result.pose.x == 0.0 and result.pose.y == 0.0 and result.pose.heading == 0.0


def test_runs_against_demo_camera_stream_without_crashing():
    """DemoFrameSource's default (no world context) mode has a mostly-static
    background — no real camera parallax — so near-zero motion / modest
    confidence is the CORRECT, expected result here, not a bug. This just
    confirms the localizer runs cleanly end-to-end on real pipeline frames."""
    cam = DemoFrameSource(width=320, height=240, num_obstacles=2, seed=4)
    cam.start()
    localizer = OrbFeatureLocalizer()
    last = None
    for _ in range(15):
        frame = cam.read_frame()
        last = localizer.track(frame, dt=0.1)
    assert last is not None
    assert last.status in ("tracking", "lost", "initializing")
    assert 0.0 <= last.confidence <= 1.0
