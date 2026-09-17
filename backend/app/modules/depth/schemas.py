from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.modules.perception.schemas import ObstacleDetection


@dataclass
class DepthResult:
    timestamp: float
    mode: str  # "synthetic" for this MVP; a real model would report "real"
    relative_depth_map: np.ndarray  # (H, W) float32, 0 = near, 1 = far — NOT metric
    # Aligned by index with the PerceptionResult.obstacles passed into estimate().
    obstacle_relative_depth: list[float]
    confidence: float

    # Class-agnostic, motion-independent static-hazard signal (see
    # ground_deviation.py) — this is the PRIMARY signal phase 1g's risk
    # field uses for terrain_cost/obstacle_cost, since neither perception
    # path alone can see un-classified, stationary hazards.
    ground_deviation_mask: np.ndarray  # (H, W) uint8, 1 = deviates from expected ground appearance
    ground_deviation_regions: list[ObstacleDetection]  # label="terrain_deviation"
    ground_deviation_relative_depth: list[float]  # aligned with ground_deviation_regions
