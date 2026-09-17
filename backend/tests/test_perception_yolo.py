import numpy as np
import pytest

from app.modules.camera.demo_source import DemoFrameSource

try:
    import ultralytics  # noqa: F401

    _ULTRALYTICS_INSTALLED = True
except ImportError:
    _ULTRALYTICS_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not _ULTRALYTICS_INSTALLED,
    reason="requirements-real.txt not installed — real YOLOv8n path is optional (see README)",
)


def test_yolo_perception_loads_and_is_ready():
    from app.modules.perception.yolo import YoloPerception

    module = YoloPerception()
    assert module.is_ready is True
    assert module.mode == "real"


def test_yolo_perception_matches_common_interface_on_demo_frame():
    from app.modules.perception.yolo import YoloPerception

    module = YoloPerception()
    cam = DemoFrameSource(width=320, height=240, num_obstacles=2, seed=5)
    cam.start()
    frame = cam.read_frame()

    result = module.process(frame)

    assert result.mode == "real"
    assert result.model_name == "yolov8n"
    assert result.free_space_mask.shape == (240, 320)
    assert result.annotated_frame.shape == (240, 320, 3)
    assert result.annotated_frame.dtype == np.uint8
    assert 0.0 <= result.confidence <= 1.0
    # COCO has no classes for the synthetic hazard blobs — an empty list here
    # is expected and is exactly why the classical fallback exists for demo mode.
    assert isinstance(result.obstacles, list)


def test_yolo_perception_detects_known_coco_classes_on_a_real_photo(tmp_path):
    """End-to-end sanity check against a real, non-synthetic photo containing
    people and a bus — both are in _HAZARD_CLASSES. Skips if offline."""
    import urllib.error
    import urllib.request

    import cv2

    from app.modules.perception.yolo import YoloPerception

    try:
        data = urllib.request.urlopen("https://ultralytics.com/images/bus.jpg", timeout=10).read()
    except (urllib.error.URLError, TimeoutError):
        pytest.skip("no network access to fetch the sample photo")

    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    module = YoloPerception()
    result = module.process(frame)

    labels = {o.label for o in result.obstacles}
    assert "person" in labels
    assert "bus" in labels
    assert result.confidence > 0.5
