from __future__ import annotations

from app.modules.camera.factory import CameraMode, create_camera_source
from app.modules.depth.factory import create_depth_estimator
from app.modules.localization.factory import create_localizer
from app.modules.perception.factory import PerceptionMode, create_perception_module
from app.modules.pipeline.orchestrator import NavigationPipeline
from app.modules.simulator.environment import Environment
from app.modules.telemetry.event_logger import EventLogger


def create_pipeline(
    environment: Environment,
    camera_mode: CameraMode = "demo",
    perception_mode: PerceptionMode = "fallback",
    db_path: str = "ugv_events.db",
) -> NavigationPipeline:
    camera = create_camera_source(camera_mode)
    perception = create_perception_module(perception_mode)
    depth = create_depth_estimator()
    localizer = create_localizer("orb")
    event_logger = EventLogger(db_path)
    return NavigationPipeline(environment, camera, perception, depth, localizer, event_logger=event_logger)
