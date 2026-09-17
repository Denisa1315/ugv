from __future__ import annotations

from typing import Literal

from app.modules.localization.base import Localizer
from app.modules.localization.orb_localizer import OrbFeatureLocalizer

LocalizationMode = Literal["orb"]


def create_localizer(mode: LocalizationMode = "orb", **kwargs) -> Localizer:
    if mode == "orb":
        return OrbFeatureLocalizer(**kwargs)
    raise ValueError(f"Unknown localization mode: {mode!r}")
