from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.modules.perception.schemas import PerceptionResult


class PerceptionModule(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray) -> PerceptionResult:
        """Run perception on one BGR frame. Only called when is_ready is True."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """'real' | 'fallback'."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether process() can currently be called (model loaded, etc.)."""
