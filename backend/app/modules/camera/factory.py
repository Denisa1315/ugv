from __future__ import annotations

from typing import Literal

from app.modules.camera.base import CameraSource
from app.modules.camera.demo_source import DemoFrameSource
from app.modules.camera.video_file_source import VideoFileSource
from app.modules.camera.webcam_source import WebcamSource

CameraMode = Literal["webcam", "video_file", "demo"]


def create_camera_source(mode: CameraMode = "demo", **kwargs) -> CameraSource:
    if mode == "webcam":
        return WebcamSource(**kwargs)
    if mode == "video_file":
        return VideoFileSource(**kwargs)
    if mode == "demo":
        return DemoFrameSource(**kwargs)
    raise ValueError(f"Unknown camera mode: {mode!r}")
