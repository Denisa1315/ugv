from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.modules.localization.schemas import LocalizationResult


class Localizer(ABC):
    @abstractmethod
    def track(self, frame: np.ndarray, dt: float) -> LocalizationResult:
        """Process one frame, updating and returning the accumulated relative pose."""

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated pose to the origin and drop tracking state."""

    @property
    @abstractmethod
    def mode(self) -> str: ...
