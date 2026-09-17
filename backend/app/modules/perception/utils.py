"""Helpers shared by every perception implementation, so free-space
heuristics and confidence scoring stay identical regardless of which
detector produced the obstacle list.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.modules.perception.schemas import ObstacleDetection


def frame_quality_score(frame: np.ndarray, scale: float = 300.0) -> float:
    """Laplacian-variance sharpness proxy, normalized to ~[0, 1].

    Not a calibrated probability — a genuinely-measured signal (computed
    from actual pixel data) for how much a blurry/degenerate frame should
    pull perception confidence down. This is what lets an empty detection
    list on a bad frame read as low-confidence instead of silently "safe".
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.clip(variance / scale, 0.0, 1.0))


def compute_free_space_mask(
    frame_shape: tuple[int, int],
    obstacles: list[ObstacleDetection],
    horizon_fraction: float = 0.45,
    dilation_px: int = 8,
) -> np.ndarray:
    """Heuristic proxy for drivable free space: everything below the
    horizon row, minus a dilated margin around each detected obstacle.

    This is NOT a learned terrain-segmentation model (out of scope for the
    MVP — see README "Future Industrial Version"); it is a cheap geometric
    stand-in that gives the risk field and planner something concrete to
    consume today, with an interface a real segmentation model could
    replace later without changing any downstream code.
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    horizon_row = int(h * horizon_fraction)
    mask[horizon_row:, :] = 1

    for obstacle in obstacles:
        x1 = max(0, obstacle.x1 - dilation_px)
        y1 = max(0, obstacle.y1 - dilation_px)
        x2 = min(w, obstacle.x2 + dilation_px)
        y2 = min(h, obstacle.y2 + dilation_px)
        mask[y1:y2, x1:x2] = 0

    return mask
