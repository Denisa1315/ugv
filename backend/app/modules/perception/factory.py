from __future__ import annotations

from typing import Literal

from app.modules.perception.base import PerceptionModule
from app.modules.perception.fallback import ClassicalPerception

PerceptionMode = Literal["auto", "real", "fallback"]


def create_perception_module(
    mode: PerceptionMode = "auto",
    real_kwargs: dict | None = None,
    fallback_kwargs: dict | None = None,
) -> PerceptionModule:
    """FALLBACK RULE at the perception layer.

    "fallback" -> always classical CV (zero downloads, always works).
    "real"     -> YOLOv8n, or raise if it genuinely can't be loaded.
    "auto"     -> try YOLOv8n; silently use classical CV if unavailable.
    """
    real_kwargs = real_kwargs or {}
    fallback_kwargs = fallback_kwargs or {}

    if mode == "fallback":
        return ClassicalPerception(**fallback_kwargs)

    if mode in ("real", "auto"):
        real: PerceptionModule | None = None
        try:
            from app.modules.perception.yolo import YoloPerception

            real = YoloPerception(**real_kwargs)
        except Exception:
            real = None

        if real is not None and real.is_ready:
            return real

        if mode == "real":
            raise RuntimeError(
                "Real perception mode requested but YOLOv8n could not be loaded "
                "(ultralytics not installed, or weights unavailable). Install "
                "requirements-real.txt and ensure internet access, or use mode="
                "'fallback'/'auto'."
            )
        return ClassicalPerception(**fallback_kwargs)

    raise ValueError(f"Unknown perception mode: {mode!r}")
