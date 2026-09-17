from __future__ import annotations

import os

import cv2
import numpy as np

from app.modules.camera.base import CameraSource


class VideoFileSource(CameraSource):
    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        if not os.path.isfile(self.path):
            self._cap = None
            return
        cap = cv2.VideoCapture(self.path)
        if cap.isOpened():
            self._cap = cap
        else:
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
        if not ok:
            if not self.loop:
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                return None
        return frame

    @property
    def source_type(self) -> str:
        return "video_file"
