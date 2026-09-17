"""Synthetic frame generator: the DEMO MODE camera source.

Produces frames that look enough like an outdoor scene (sky, textured
ground, a lighter drivable "path" corridor, drifting colored hazard blobs)
to exercise the entire downstream pipeline identically to a real camera —
this is what makes a reliable, cameraless live demo possible. is_available()
is always True: there is no hardware to fail.

Two rendering modes:
  - DEFAULT (no world context set): obstacles drift independently across
    the frame — simple, self-contained, exactly the phase 1c/1d behavior.
  - POSE-CONSISTENT (after update_world_context()): obstacles are instead
    projected from the Simulator's actual Environment.obstacles using the
    vehicle's true pose (see app.modules.geometry) — this is what lets
    phases 1g-1j's risk field/planner meaningfully navigate around the
    same obstacles the vehicle can actually collide with. The sky/ground
    gradient itself intentionally stays static — no full 3D scene renderer
    (out of scope). A set of small, fixed, non-hazardous "landmark" points
    scattered across world space ARE also projected pose-consistently
    (same projection as obstacles), purely so phase 1f's visual localizer
    has genuine, geometrically-correct texture/parallax to track as the
    vehicle moves — without them, a scene with no obstacle currently in
    view has zero real parallax, and honest zero localization confidence
    would otherwise permanently block phase 1k's decision engine from ever
    allowing motion. Landmarks are low-contrast and reliably smaller than
    the hazard-detection area/color thresholds, so they cannot be mistaken
    for hazards by perception or ground-deviation detection.

The hazard blobs are deliberately colored/shaped to also be detectable by
the classical-CV fallback perception path (phase 1d) via background-
subtraction motion detection when they move (drifting in default mode, or
via parallax as the vehicle moves in pose-consistent mode).
"""

from __future__ import annotations

import cv2
import numpy as np

from app.modules.camera.base import CameraSource
from app.modules.geometry import CameraGeometryConfig, world_point_to_image
from app.modules.simulator.environment import Environment
from app.modules.simulator.vehicle import VehicleState

_SKY_TOP = np.array([120, 70, 30], dtype=np.float32)  # BGR
_SKY_HORIZON = np.array([180, 150, 120], dtype=np.float32)
_GROUND_NEAR_HORIZON = np.array([50, 90, 55], dtype=np.float32)
_GROUND_BOTTOM = np.array([35, 70, 40], dtype=np.float32)
_PATH_COLOR = np.array([120, 120, 120], dtype=np.float32)

_HAZARD_COLORS = [
    (30, 60, 220),  # orange-red
    (20, 100, 230),  # orange
    (60, 40, 200),  # red-purple
]


class DemoFrameSource(CameraSource):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        num_obstacles: int = 3,
        seed: int = 7,
        camera_geometry: CameraGeometryConfig | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.camera_geometry = camera_geometry or CameraGeometryConfig(horizon_fraction=0.45)
        self._rng = np.random.default_rng(seed)
        self._active = False
        self._frame_index = 0
        self._horizon_row = int(height * 0.45)
        self._background = self._build_background()
        self._obstacles = self._init_obstacles(num_obstacles)
        self._landmarks = self._init_landmarks()
        self._world_context: tuple[VehicleState, Environment] | None = None

    # -- construction ----------------------------------------------------
    def _build_background(self) -> np.ndarray:
        h, w = self.height, self.width
        img = np.zeros((h, w, 3), dtype=np.float32)
        horizon = self._horizon_row

        for row in range(horizon):
            t = row / max(1, horizon - 1)
            img[row, :] = (1 - t) * _SKY_TOP + t * _SKY_HORIZON
        for row in range(horizon, h):
            t = (row - horizon) / max(1, h - horizon - 1)
            img[row, :] = (1 - t) * _GROUND_NEAR_HORIZON + t * _GROUND_BOTTOM

        # Drivable path corridor: a trapezoid narrowing toward the horizon,
        # giving the perception layer's free-space heuristic something
        # concrete (a lighter, more uniform region) to key off later.
        top_half_width = w * 0.08
        bottom_half_width = w * 0.30
        cx = w / 2
        path_pts = np.array(
            [
                [cx - top_half_width, horizon],
                [cx + top_half_width, horizon],
                [cx + bottom_half_width, h],
                [cx - bottom_half_width, h],
            ],
            dtype=np.int32,
        )
        path_layer = img.copy()
        cv2.fillPoly(path_layer, [path_pts], _PATH_COLOR.tolist())
        img = 0.55 * img + 0.45 * path_layer

        noise = self._rng.normal(0, 6, size=img.shape)
        img = np.clip(img + noise, 0, 255)
        return img.astype(np.uint8)

    def _init_obstacles(self, n: int) -> list[dict]:
        obstacles = []
        for i in range(n):
            speed = self._rng.uniform(25, 55) * self._rng.choice([-1.0, 1.0])
            obstacles.append(
                {
                    "x": float(self._rng.uniform(0.15, 0.85) * self.width),
                    "y": float(self._rng.uniform(0.55, 0.88) * self.height),
                    "vx": speed,
                    "radius": int(self._rng.integers(14, 30)),
                    "color": _HAZARD_COLORS[i % len(_HAZARD_COLORS)],
                }
            )
        return obstacles

    def _init_landmarks(self, n: int = 220) -> list[dict]:
        """Fixed, non-hazardous world-space texture points — see module
        docstring. Scattered over a generous area so they're visible
        across a wide range of vehicle positions/headings, not just one
        specific Environment's bounds."""
        landmarks = []
        for _ in range(n):
            shade = int(self._rng.integers(-18, 18))  # fixed per-landmark contrast, small enough to stay subtle
            landmarks.append(
                {
                    "x": float(self._rng.uniform(-15.0, 35.0)),
                    "y": float(self._rng.uniform(-15.0, 35.0)),
                    "shade": shade,
                }
            )
        return landmarks

    # -- CameraSource interface ------------------------------------------
    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

    def is_available(self) -> bool:
        return True

    @property
    def source_type(self) -> str:
        return "demo"

    def update_world_context(self, vehicle: VehicleState, environment: Environment) -> None:
        self._world_context = (vehicle, environment)

    def read_frame(self) -> np.ndarray | None:
        if not self._active:
            return None

        frame = self._background.copy()
        if self._world_context is not None:
            self._draw_landmarks(frame, self._world_context[0])
            self._draw_world_obstacles(frame, *self._world_context)
        else:
            self._draw_drifting_obstacles(frame)

        self._frame_index += 1
        return frame

    def _draw_drifting_obstacles(self, frame: np.ndarray) -> None:
        for obstacle in self._obstacles:
            obstacle["x"] += obstacle["vx"] * 0.05
            margin = obstacle["radius"] + 5
            if obstacle["x"] < -margin:
                obstacle["x"] = self.width + margin
            elif obstacle["x"] > self.width + margin:
                obstacle["x"] = -margin

            center = (int(obstacle["x"]), int(obstacle["y"]))
            cv2.circle(frame, center, obstacle["radius"], obstacle["color"], -1)
            cv2.circle(frame, center, obstacle["radius"], (10, 10, 10), 2)

    def _draw_landmarks(self, frame: np.ndarray, vehicle: VehicleState) -> None:
        for landmark in self._landmarks:
            projected = world_point_to_image(
                landmark["x"], landmark["y"], (self.height, self.width), vehicle, self.camera_geometry
            )
            if projected is None:
                continue
            px, py, forward_distance = projected
            radius = int(np.clip(4.0 / max(forward_distance, 0.5), 1, 3))
            # Small + low-contrast by construction: stays well under the
            # ground-deviation detector's area/color thresholds and MOG2's
            # foreground sensitivity, so it reads as texture, not a hazard.
            base = frame[max(0, py - radius), px].astype(np.int16) if 0 <= py < self.height and 0 <= px < self.width else np.array([40, 70, 40])
            color = tuple(int(c) for c in np.clip(base + landmark["shade"], 0, 255))
            cv2.circle(frame, (px, max(0, py - radius)), radius, color, -1)

    def _draw_world_obstacles(self, frame: np.ndarray, vehicle: VehicleState, environment: Environment) -> None:
        for i, obstacle in enumerate(environment.obstacles):
            projected = world_point_to_image(
                obstacle.x, obstacle.y, (self.height, self.width), vehicle, self.camera_geometry
            )
            if projected is None:
                continue
            px, py, forward_distance = projected
            focal = self.camera_geometry.focal_px(self.width)
            pixel_radius = int(np.clip((obstacle.radius * focal) / max(forward_distance, 0.1), 3, self.height))

            # Anchor the circle's BASE (not center) at the ground-projected
            # point (px, py) — the depth module treats a detection's bottom
            # edge as its ground-contact point, so the rendering must agree,
            # or depth estimates systematically undershoot for large/near
            # obstacles (whose bottom edge would otherwise sit well past
            # the true ground-plane row).
            color = _HAZARD_COLORS[i % len(_HAZARD_COLORS)]
            base_center = (px, max(0, py - pixel_radius))
            cv2.circle(frame, base_center, pixel_radius, color, -1)
            cv2.circle(frame, base_center, pixel_radius, (10, 10, 10), 2)
