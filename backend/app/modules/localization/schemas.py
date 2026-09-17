from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RelativePose:
    """Pose accumulated since the localizer was last reset() — NOT tied to
    any absolute/GPS coordinate frame. Origin is wherever tracking started."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0


@dataclass
class LocalizationResult:
    timestamp: float
    mode: str  # "orb_essential_matrix"
    pose: RelativePose  # explicitly RELATIVE — never GPS-accurate
    confidence: float  # 0..1, from tracked feature count + essential-matrix inlier ratio
    tracked_features: int
    matched_features: int
    inliers: int
    status: str  # "initializing" | "tracking" | "lost"
