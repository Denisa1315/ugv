import cv2
import numpy as np
import pytest

from app.modules.camera.demo_source import DemoFrameSource
from app.modules.camera.factory import create_camera_source
from app.modules.camera.video_file_source import VideoFileSource
from app.modules.camera.webcam_source import WebcamSource


# -- demo source -----------------------------------------------------------


def test_demo_source_unavailable_frames_before_start():
    src = DemoFrameSource(width=64, height=48)
    assert src.is_available() is True  # synthetic: always "available"
    assert src.read_frame() is None  # but not started yet -> no frames


def test_demo_source_produces_frames_after_start():
    src = DemoFrameSource(width=64, height=48)
    src.start()
    frame = src.read_frame()
    assert frame is not None
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    src.stop()
    assert src.read_frame() is None


def test_demo_source_obstacles_move_between_frames():
    src = DemoFrameSource(width=200, height=150, num_obstacles=1, seed=1)
    src.start()
    positions = [src._obstacles[0]["x"]]
    for _ in range(5):
        src.read_frame()
        positions.append(src._obstacles[0]["x"])
    assert len(set(positions)) > 1  # actually moved, not frozen


def test_demo_source_is_deterministic_given_seed():
    a = DemoFrameSource(width=64, height=48, seed=42)
    b = DemoFrameSource(width=64, height=48, seed=42)
    a.start()
    b.start()
    frame_a = a.read_frame()
    frame_b = b.read_frame()
    assert np.array_equal(frame_a, frame_b)


# -- video file source -------------------------------------------------


def _write_tiny_video(path: str, num_frames: int = 6, size: tuple[int, int] = (32, 24)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_file_source_reads_and_loops(tmp_path):
    video_path = str(tmp_path / "clip.mp4")
    _write_tiny_video(video_path, num_frames=4, size=(32, 24))

    src = VideoFileSource(video_path, loop=True)
    src.start()
    assert src.is_available() is True

    frames_read = 0
    for _ in range(10):  # more than num_frames -> exercises the loop-back path
        frame = src.read_frame()
        assert frame is not None
        frames_read += 1
    assert frames_read == 10
    src.stop()
    assert src.is_available() is False


def test_video_file_source_missing_file_is_graceful():
    src = VideoFileSource("/nonexistent/path/video.mp4", loop=True)
    src.start()  # must not raise
    assert src.is_available() is False
    assert src.read_frame() is None


def test_video_file_source_no_loop_stops_at_end(tmp_path):
    video_path = str(tmp_path / "clip.mp4")
    _write_tiny_video(video_path, num_frames=3, size=(32, 24))

    src = VideoFileSource(video_path, loop=False)
    src.start()
    frames = [src.read_frame() for _ in range(5)]
    assert sum(f is not None for f in frames) == 3
    assert frames[-1] is None


# -- webcam source (best-effort: no camera hardware guaranteed here) ------


def test_webcam_source_never_raises_when_unavailable():
    src = WebcamSource(device_index=99)  # near-certainly nonexistent index
    src.start()  # must not raise even with no such device
    assert isinstance(src.is_available(), bool)
    if not src.is_available():
        assert src.read_frame() is None
    src.stop()
    assert src.is_available() is False


# -- factory -----------------------------------------------------------


def test_factory_creates_expected_types():
    assert isinstance(create_camera_source("demo"), DemoFrameSource)
    assert isinstance(create_camera_source("webcam"), WebcamSource)
    assert isinstance(create_camera_source("video_file", path="x.mp4"), VideoFileSource)


def test_factory_rejects_unknown_mode():
    with pytest.raises(ValueError):
        create_camera_source("not_a_real_mode")
