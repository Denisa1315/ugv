"""Top-down PNG renderer for the 2D simulation.

Used right now to visually verify the simulator moves without a frontend
yet. Reused as-is by GET /simulator/render.png and, later, by the React
dashboard's 2D map panel (obstacles/goal/path/risk-heatmap overlay slots
are deliberately drawn as separate layers here so that reuse is a
straight extension, not a rewrite).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.modules.simulator.simulator import Simulator

PX_PER_METER = 40
MARGIN_PX = 20

_BG_COLOR = (30, 30, 30)
_GRID_COLOR = (55, 55, 55)
_GOAL_COLOR = (0, 215, 255)
_OBSTACLE_COLOR = (60, 60, 230)
_TRAJECTORY_COLOR = (120, 200, 120)
_VEHICLE_COLOR = (255, 200, 60)


def _to_px(x: float, y: float, height_px: int) -> tuple[int, int]:
    """Environment (meters, y-up) -> image pixels (origin top-left, y-down)."""
    px = MARGIN_PX + int(round(x * PX_PER_METER))
    py = height_px - MARGIN_PX - int(round(y * PX_PER_METER))
    return px, py


def render_topdown(sim: Simulator) -> np.ndarray:
    env = sim.environment
    width_px = int(env.width * PX_PER_METER) + 2 * MARGIN_PX
    height_px = int(env.height * PX_PER_METER) + 2 * MARGIN_PX

    img = np.full((height_px, width_px, 3), _BG_COLOR, dtype=np.uint8)

    # 1m grid
    for gx in range(0, int(env.width) + 1):
        p1 = _to_px(gx, 0, height_px)
        p2 = _to_px(gx, env.height, height_px)
        cv2.line(img, p1, p2, _GRID_COLOR, 1)
    for gy in range(0, int(env.height) + 1):
        p1 = _to_px(0, gy, height_px)
        p2 = _to_px(env.width, gy, height_px)
        cv2.line(img, p1, p2, _GRID_COLOR, 1)

    # obstacles
    for obstacle in env.obstacles:
        center = _to_px(obstacle.x, obstacle.y, height_px)
        radius_px = max(2, int(obstacle.radius * PX_PER_METER))
        cv2.circle(img, center, radius_px, _OBSTACLE_COLOR, -1)

    # goal
    goal_px = _to_px(env.goal[0], env.goal[1], height_px)
    cv2.drawMarker(img, goal_px, _GOAL_COLOR, cv2.MARKER_STAR, 18, 2)

    # trajectory trail
    points = [_to_px(x, y, height_px) for x, y in sim.trajectory]
    for p1, p2 in zip(points, points[1:]):
        cv2.line(img, p1, p2, _TRAJECTORY_COLOR, 2)

    # vehicle (triangle pointing along heading)
    state = sim.vehicle.state
    center = _to_px(state.x, state.y, height_px)
    nose_len = 0.5 * PX_PER_METER
    back = 0.3 * PX_PER_METER
    heading = state.heading
    nose = (
        int(center[0] + nose_len * math.cos(heading)),
        int(center[1] - nose_len * math.sin(heading)),
    )
    left = (
        int(center[0] + back * math.cos(heading + 2.5)),
        int(center[1] - back * math.sin(heading + 2.5)),
    )
    right = (
        int(center[0] + back * math.cos(heading - 2.5)),
        int(center[1] - back * math.sin(heading - 2.5)),
    )
    cv2.fillPoly(img, [np.array([nose, left, right])], _VEHICLE_COLOR)

    status_text = f"{sim.status.value.upper()}  t={sim.elapsed_time:0.1f}s"
    if sim.goal_reached:
        status_text += "  GOAL REACHED"
    if sim.collided:
        status_text += "  COLLISION"
    cv2.putText(img, status_text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    return img


def render_topdown_png(sim: Simulator) -> bytes:
    img = render_topdown(sim)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode top-down render as PNG")
    return buf.tobytes()
