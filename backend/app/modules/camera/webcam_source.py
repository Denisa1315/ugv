from __future__ import annotations

import cv2
import numpy as np

from app.modules.camera.base import CameraSource


class WebcamSource(CameraSource):
    def __init__(
        self,
        device_index: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self.device_index)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if cap.isOpened():
            self._cap = cap
        else:
            # No camera / permission denied / index out of range: report via
            # is_available() rather than raising, so callers can fall back.
            cap.release()
            self._cap = None

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_available(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> np.ndarray | None:
        if not self.is_available():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def source_type(self) -> str:
        return "webcam"
