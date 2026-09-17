"""Classical-CV perception: the DEMO MODE / zero-download fallback.

Detects obstacles via background subtraction (MOG2) — anything that moves
against an otherwise static scene. This is what the DemoFrameSource's
drifting hazard blobs are built to trigger. On a real static webcam feed it
degrades to detecting only moving objects, which is an honest, disclosed
limitation (see README "Future Industrial Version": a real terrain/hazard
classifier would replace this).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from app.modules.perception.base import PerceptionModule
from app.modules.perception.schemas import ObstacleDetection, PerceptionResult
from app.modules.perception.utils import compute_free_space_mask, frame_quality_score


class ClassicalPerception(PerceptionModule):
    MIN_CONTOUR_AREA_FRACTION = 0.0015  # of frame area, below which a blob is noise

    def __init__(
        self,
        history: int = 200,
        var_threshold: float = 24.0,
        warmup_frames: int = 15,
        horizon_fraction: float = 0.45,
    ) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self._warmup_frames = warmup_frames
        self._horizon_fraction = horizon_fraction
        self._frames_processed = 0

    @property
    def mode(self) -> str:
        return "fallback"

    @property
    def is_ready(self) -> bool:
        return True  # no model to load — always ready

    def process(self, frame: np.ndarray) -> PerceptionResult:
        h, w = frame.shape[:2]
        fg_mask = self._subtractor.apply(frame)
        self._frames_processed += 1

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.MIN_CONTOUR_AREA_FRACTION * h * w

        obstacles: list[ObstacleDetection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            area_ratio = area / (h * w)
            # Heuristic proxy confidence (classical CV has no calibrated
            # probability): bigger, more solid blobs score higher.
            det_confidence = float(np.clip(0.35 + area_ratio * 25, 0.3, 0.9))
            obstacles.append(
                ObstacleDetection(x1=x, y1=y, x2=x + bw, y2=y + bh, confidence=det_confidence, label="hazard")
            )

        free_space_mask = compute_free_space_mask((h, w), obstacles, self._horizon_fraction)

        # Background model needs a handful of frames to warm up; report that
        # honestly via measured frame count rather than pretending day-one
        # confidence is as good as steady-state.
        warmup_factor = min(1.0, self._frames_processed / self._warmup_frames)
        quality = frame_quality_score(frame)
        detection_component = (
            float(np.mean([o.confidence for o in obstacles])) if obstacles else quality
        )
        confidence = float(np.clip(0.5 * warmup_factor + 0.25 * quality + 0.25 * detection_component, 0.0, 1.0))

        annotated = self._annotate(frame, obstacles, free_space_mask)

        return PerceptionResult(
            timestamp=time.time(),
            mode="fallback",
            model_name="classical-cv-mog2",
            obstacles=obstacles,
            free_space_mask=free_space_mask,
            confidence=confidence,
            annotated_frame=annotated,
        )

    @staticmethod
    def _annotate(frame: np.ndarray, obstacles: list[ObstacleDetection], free_space_mask: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        overlay[free_space_mask.astype(bool)] = (0, 180, 0)
        annotated = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        for obstacle in obstacles:
            cv2.rectangle(annotated, (obstacle.x1, obstacle.y1), (obstacle.x2, obstacle.y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"hazard {obstacle.confidence:.2f}",
                (obstacle.x1, max(0, obstacle.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
        return annotated
