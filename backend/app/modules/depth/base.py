from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.modules.depth.schemas import DepthResult
from app.modules.perception.schemas import PerceptionResult


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray, perception: PerceptionResult) -> DepthResult:
        """Relative depth only — never metric-accurate for this MVP."""

    @property
    @abstractmethod
    def mode(self) -> str: ...
