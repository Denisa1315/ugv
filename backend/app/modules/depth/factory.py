from __future__ import annotations

from typing import Literal

from app.modules.depth.base import DepthEstimator
from app.modules.depth.synthetic import SyntheticRelativeDepth

DepthMode = Literal["synthetic"]


def create_depth_estimator(mode: DepthMode = "synthetic", **kwargs) -> DepthEstimator:
    if mode == "synthetic":
        return SyntheticRelativeDepth(**kwargs)
    raise ValueError(
        f"Unknown depth mode: {mode!r} (only 'synthetic' is implemented for this "
        "MVP — a real monocular depth model would plug in here later via the "
        "same DepthEstimator interface)"
    )
