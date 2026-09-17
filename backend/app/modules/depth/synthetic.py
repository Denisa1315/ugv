"""Ground-plane monocular relative-depth heuristic — the only depth mode
in this MVP (no model training/download involved).

Assumes a flat ground plane and fixed camera pitch: pixel row maps
monotonically to distance from the vehicle (rows near the bottom of the
frame are close; rows near the horizon are far). This is a classical,
deterministic approximation — explicitly NOT metric-accurate and NOT a
learned monocular depth model (training one is out of scope for the MVP;
see README "Future Industrial Version").

A real depth model (e.g. a pretrained MiDaS variant) could implement this
same DepthEstimator interface later — it would just return a learned
relative_depth_map instead of this geometric one; nothing downstream
(risk field, planners) would need to change.
"""

from __future__ import annotations

import time

import numpy as np

from app.modules.depth.base import DepthEstimator
from app.modules.depth.ground_deviation import compute_ground_deviation_mask, extract_deviation_regions
from app.modules.depth.schemas import DepthResult
from app.modules.perception.schemas import PerceptionResult
from app.modules.perception.utils import frame_quality_score


class SyntheticRelativeDepth(DepthEstimator):
    def __init__(
        self,
        horizon_fraction: float = 0.45,
        weight_perception_confidence: float = 0.6,
        weight_frame_quality: float = 0.4,
        ground_deviation_color_tolerance: float = 16.0,
    ) -> None:
        self.horizon_fraction = horizon_fraction
        self.weight_perception_confidence = weight_perception_confidence
        self.weight_frame_quality = weight_frame_quality
        self.ground_deviation_color_tolerance = ground_deviation_color_tolerance

    @property
    def mode(self) -> str:
        return "synthetic"

    def estimate(self, frame: np.ndarray, perception: PerceptionResult) -> DepthResult:
        h, w = frame.shape[:2]
        horizon_row = int(h * self.horizon_fraction)

        # Sky (above horizon) has no ground-plane distance — treated as
        # maximally far/undefined since it's irrelevant to ground navigation.
        depth_map = np.ones((h, w), dtype=np.float32)
        span = max(1, h - 1 - horizon_row)
        rows_below = np.arange(horizon_row, h)
        t = (rows_below - horizon_row) / span  # 0 at horizon (far) .. 1 at bottom (near)
        depth_map[horizon_row:, :] = (1.0 - t)[:, None]

        obstacle_relative_depth: list[float] = []
        for obstacle in perception.obstacles:
            contact_x = int(np.clip((obstacle.x1 + obstacle.x2) // 2, 0, w - 1))
            contact_y = int(np.clip(obstacle.y2, 0, h - 1))  # bbox bottom = ground contact point
            obstacle_relative_depth.append(float(depth_map[contact_y, contact_x]))

        confidence = float(
            np.clip(
                self.weight_perception_confidence * perception.confidence
                + self.weight_frame_quality * frame_quality_score(frame),
                0.0,
                1.0,
            )
        )

        deviation_mask = compute_ground_deviation_mask(
            frame, horizon_row, color_tolerance=self.ground_deviation_color_tolerance
        )
        deviation_regions = extract_deviation_regions(deviation_mask, frame_area=h * w)
        deviation_relative_depth = []
        for region in deviation_regions:
            contact_x = int(np.clip((region.x1 + region.x2) // 2, 0, w - 1))
            contact_y = int(np.clip(region.y2, 0, h - 1))
            deviation_relative_depth.append(float(depth_map[contact_y, contact_x]))

        return DepthResult(
            timestamp=time.time(),
            mode="synthetic",
            relative_depth_map=depth_map,
            obstacle_relative_depth=obstacle_relative_depth,
            confidence=confidence,
            ground_deviation_mask=deviation_mask,
            ground_deviation_regions=deviation_regions,
            ground_deviation_relative_depth=deviation_relative_depth,
        )
