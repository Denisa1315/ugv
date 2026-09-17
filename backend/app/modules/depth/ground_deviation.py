"""Class-agnostic, motion-independent static-hazard detection.

Neither perception path alone can see the PS's named static hazards:
MOG2 (phase 1d fallback) only fires on motion, and YOLOv8n's COCO classes
have no rock/ditch/tree/stump categories. This looks for local appearance
anomalies against the smoothly-varying expected ground appearance instead —
a single-RGB-frame proxy for "this patch doesn't look like the ground plane
around it", which is what a real height-above-ground-plane check (stereo/
LiDAR) would give more rigorously (that's the Future Industrial Version
upgrade; see README).

Technique: a large-kernel Gaussian blur in LAB space approximates the
"expected" locally-smooth ground appearance (gradients survive, small
objects get averaged away); the per-pixel residual against that expectation
highlights localized anomalies — a classical background/expectation
residual filter, fully vectorized (no Python pixel loops), fast enough for
the per-cycle real-time budget.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.modules.perception.schemas import ObstacleDetection

MIN_REGION_AREA_FRACTION = 0.001  # of frame area
MIN_SOLIDITY = 0.4  # area / bbox_area — rejects thin lines/edges (e.g. terrain boundaries),
# which are real appearance discontinuities but not hazard blobs; a filled
# circle/rock is ~0.78, a thin diagonal stripe is well under 0.4.


def _default_blur_kernel(width: int) -> tuple[int, int]:
    # Scaled to frame width (not a fixed pixel count) so the "expected
    # local ground appearance" estimate stays proportionate across camera
    # resolutions — a fixed 41x41 kernel is reasonable at 640px wide but
    # oversized (blends a hazard's own interior into its "expected" value)
    # at something like 160px wide.
    size = max(15, int(round(width * 0.065)))
    if size % 2 == 0:
        size += 1
    return (size, size)


def compute_ground_deviation_mask(
    frame: np.ndarray,
    horizon_row: int,
    color_tolerance: float = 16.0,
    blur_kernel: tuple[int, int] | None = None,
) -> np.ndarray:
    if blur_kernel is None:
        blur_kernel = _default_blur_kernel(frame.shape[1])
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    expected = cv2.GaussianBlur(lab, blur_kernel, 0)
    residual = np.linalg.norm(lab - expected, axis=2)

    mask = (residual > color_tolerance).astype(np.uint8)
    mask[:horizon_row, :] = 0  # only evaluate the ground region

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def extract_deviation_regions(mask: np.ndarray, frame_area: int) -> list[ObstacleDetection]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_REGION_AREA_FRACTION * frame_area

    regions: list[ObstacleDetection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox_area = max(1, bw * bh)
        if (area / bbox_area) < MIN_SOLIDITY:
            continue

        area_ratio = area / frame_area
        confidence = float(np.clip(0.3 + area_ratio * 20, 0.25, 0.85))
        regions.append(
            ObstacleDetection(x1=x, y1=y, x2=x + bw, y2=y + bh, confidence=confidence, label="terrain_deviation")
        )
    return regions
