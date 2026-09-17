"""Common interface every camera/video source implements.

Nothing downstream (perception, depth, ...) is allowed to know or care
whether frames came from a webcam, an uploaded video file, or synthetic
demo frames — it only ever calls this interface. That's what makes the
REAL MODE / DEMO MODE fallback rule possible without branching logic
scattered through the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.modules.simulator.environment import Environment
    from app.modules.simulator.vehicle import VehicleState


class CameraSource(ABC):
    @abstractmethod
    def start(self) -> None:
        """Open/prepare the source. Must not raise if the device is
        unavailable — set internal state so is_available() reports False
        instead, so callers can fall back gracefully."""

    @abstractmethod
    def stop(self) -> None:
        """Release any underlying resources (capture device, file handle)."""

    @abstractmethod
    def read_frame(self) -> np.ndarray | None:
        """Return the next frame as a BGR uint8 HxWx3 array, or None if no
        frame is currently available (source not started, device dropped,
        end of a non-looping file, ...)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this source can currently supply frames."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Short label: 'webcam' | 'video_file' | 'demo'."""

    def update_world_context(self, vehicle: "VehicleState", environment: "Environment") -> None:
        """Optional hook: let a source render content consistent with the
        simulated vehicle's true pose/world (see DemoFrameSource). Real
        sources (webcam/video file) have no notion of a simulated world and
        correctly leave this a no-op."""
