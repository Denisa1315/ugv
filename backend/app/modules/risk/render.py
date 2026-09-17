from __future__ import annotations

import cv2
import numpy as np

from app.modules.risk.grid import RiskGrid


def render_risk_heatmap(grid: RiskGrid, upscale: int = 8) -> np.ndarray:
    risk_u8 = np.clip(grid.risk * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(risk_u8, cv2.COLORMAP_JET)
    heatmap[~grid.safe_mask] = np.clip(heatmap[~grid.safe_mask].astype(np.int16) + 40, 0, 255).astype(np.uint8)
    if upscale > 1:
        heatmap = cv2.resize(
            heatmap, (grid.cols * upscale, grid.rows * upscale), interpolation=cv2.INTER_NEAREST
        )
    return heatmap


def render_risk_heatmap_png(grid: RiskGrid, upscale: int = 8) -> bytes:
    img = render_risk_heatmap(grid, upscale=upscale)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode risk heatmap as PNG")
    return buf.tobytes()
