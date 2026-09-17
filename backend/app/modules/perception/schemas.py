"""PerceptionResult: the one output shape both the classical fallback and
the real YOLOv8n path must produce, so nothing downstream (depth, risk
field, ...) branches on which perception mode is active.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObstacleDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str


@dataclass
class PerceptionResult:
    timestamp: float
    mode: str  # "real" | "fallback"
    model_name: str
    obstacles: list[ObstacleDetection]
    free_space_mask: np.ndarray  # (H, W) uint8, 1 = drivable/free, 0 = not
    confidence: float  # measured process-quality/detection confidence, 0..1
    annotated_frame: np.ndarray  # BGR uint8, boxes + free-space overlay drawn
