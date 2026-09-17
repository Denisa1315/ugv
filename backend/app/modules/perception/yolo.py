"""Real-model perception: pretrained YOLOv8n (COCO weights, no training).

Optional dependency (see requirements-real.txt). If `ultralytics` isn't
installed, or the pretrained weights can't be fetched (no internet on
first use), is_ready stays False and the factory silently falls back to
ClassicalPerception — this module must never crash the app for that reason.

COCO has no "rock"/"tree"/terrain-hazard classes, so _HAZARD_CLASSES is an
off-the-shelf proxy for outdoor hazards, not a purpose-trained detector.
A real Industrial version would fine-tune on RELLIS-3D/RUGD-style outdoor
hazard classes — explicitly out of scope for this MVP (no training pipeline).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from app.modules.perception.base import PerceptionModule
from app.modules.perception.schemas import ObstacleDetection, PerceptionResult
from app.modules.perception.utils import compute_free_space_mask, frame_quality_score

try:
    from ultralytics import YOLO

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

_HAZARD_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "horse",
    "cow",
    "sheep",
    "bench",
    "fire hydrant",
}


class YoloPerception(PerceptionModule):
    def __init__(self, weights: str = "yolov8n.pt", confidence_threshold: float = 0.35) -> None:
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._ready = False

        if not _ULTRALYTICS_AVAILABLE:
            return
        try:
            self._model = YOLO(weights)
            self._ready = True
        except Exception:
            # Weight download failed / no internet / corrupt cache, etc.
            self._model = None
            self._ready = False

    @property
    def mode(self) -> str:
        return "real"

    @property
    def is_ready(self) -> bool:
        return self._ready

    def process(self, frame: np.ndarray) -> PerceptionResult:
        if not self._ready or self._model is None:
            raise RuntimeError("YoloPerception.process() called while not ready; check is_ready first")

        h, w = frame.shape[:2]
        results = self._model.predict(frame, verbose=False, conf=self.confidence_threshold)[0]

        obstacles: list[ObstacleDetection] = []
        for box in results.boxes:
            label = self._model.names.get(int(box.cls[0]), str(int(box.cls[0])))
            if label not in _HAZARD_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            obstacles.append(ObstacleDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, label=label))

        free_space_mask = compute_free_space_mask((h, w), obstacles)

        # Confidence reflects perception/process quality, not just whether
        # something was detected — an empty result on a sharp, well-exposed
        # frame is trustworthy; on a blurry one it should read as degraded,
        # per the "no detection != safe" rule enforced downstream.
        quality = frame_quality_score(frame)
        detection_component = float(np.mean([o.confidence for o in obstacles])) if obstacles else quality
        confidence = float(np.clip(0.5 * quality + 0.5 * detection_component, 0.0, 1.0))

        annotated = self._annotate(frame, obstacles)

        return PerceptionResult(
            timestamp=time.time(),
            mode="real",
            model_name="yolov8n",
            obstacles=obstacles,
            free_space_mask=free_space_mask,
            confidence=confidence,
            annotated_frame=annotated,
        )

    @staticmethod
    def _annotate(frame: np.ndarray, obstacles: list[ObstacleDetection]) -> np.ndarray:
        annotated = frame.copy()
        for obstacle in obstacles:
            cv2.rectangle(annotated, (obstacle.x1, obstacle.y1), (obstacle.x2, obstacle.y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"{obstacle.label} {obstacle.confidence:.2f}",
                (obstacle.x1, max(0, obstacle.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
        return annotated
