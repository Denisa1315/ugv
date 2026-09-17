"""Monocular visual odometry: ORB features + BFMatcher + essential matrix.

Frame-to-frame pipeline (classical, no learned model, no SLAM backend):
  1. ORB keypoints + descriptors on the current frame.
  2. Match against the previous frame's descriptors (BFMatcher, Hamming,
     Lowe's ratio test).
  3. cv2.findEssentialMat + cv2.recoverPose on the matched point pairs,
     using an ASSUMED pinhole intrinsic matrix (see geometry.CameraGeometryConfig
     — this is not a calibrated camera; a real deployment would calibrate).
  4. The recovered translation is a unit vector — monocular vision cannot
     recover true metric scale from two views alone (a fundamental, disclosed
     limitation, not a bug). This MVP converts it to a metric estimate via a
     configurable assumed_forward_speed heuristic. A real system would fuse
     wheel odometry, an IMU, or stereo to recover true scale (see README
     "Future Industrial Version").

Output pose is relative to wherever tracking started (reset()) — never an
absolute/GPS coordinate. No ORB-SLAM3 / g2o / loop closure / global map:
this is intentionally just the frame-to-frame tracking front-end.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from app.modules.geometry import CameraGeometryConfig
from app.modules.localization.base import Localizer
from app.modules.localization.schemas import LocalizationResult, RelativePose
from app.modules.simulator.vehicle import wrap_angle


class OrbFeatureLocalizer(Localizer):
    def __init__(
        self,
        geometry: CameraGeometryConfig | None = None,
        n_features: int = 500,
        min_tracked_features: int = 30,
        min_good_matches: int = 15,
        ratio_test_threshold: float = 0.75,
        assumed_forward_speed: float = 1.0,  # m/s — scale heuristic, see module docstring
        target_feature_count: int = 200,
        target_match_count: int = 80,
    ) -> None:
        self.geometry = geometry or CameraGeometryConfig()
        self.min_tracked_features = min_tracked_features
        self.min_good_matches = min_good_matches
        self.ratio_test_threshold = ratio_test_threshold
        self.assumed_forward_speed = assumed_forward_speed
        self.target_feature_count = target_feature_count
        self.target_match_count = target_match_count

        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._prev_gray: np.ndarray | None = None
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._pose = RelativePose()

    @property
    def mode(self) -> str:
        return "orb_essential_matrix"

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._pose = RelativePose()

    def track(self, frame: np.ndarray, dt: float) -> LocalizationResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        n_tracked = len(keypoints) if keypoints else 0

        if (
            self._prev_descriptors is None
            or descriptors is None
            or n_tracked < self.min_tracked_features
        ):
            self._prev_gray, self._prev_keypoints, self._prev_descriptors = gray, keypoints, descriptors
            status = "initializing" if self._prev_descriptors is not None else "initializing"
            return self._result(status=status, tracked=n_tracked, matched=0, inliers=0, confidence_scale=0.0)

        raw_matches = self._matcher.knnMatch(self._prev_descriptors, descriptors, k=2)
        good = [m for m, n in raw_matches if n is not None and m.distance < self.ratio_test_threshold * n.distance]

        if len(good) < self.min_good_matches:
            self._prev_gray, self._prev_keypoints, self._prev_descriptors = gray, keypoints, descriptors
            return self._result(status="lost", tracked=n_tracked, matched=len(good), inliers=0, confidence_scale=0.0)

        pts_prev = np.float32([self._prev_keypoints[m.queryIdx].pt for m in good])
        pts_curr = np.float32([keypoints[m.trainIdx].pt for m in good])

        h, w = gray.shape[:2]
        focal = self.geometry.focal_px(w)
        K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)

        E, mask = cv2.findEssentialMat(pts_prev, pts_curr, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None or E.shape != (3, 3):
            self._prev_gray, self._prev_keypoints, self._prev_descriptors = gray, keypoints, descriptors
            return self._result(status="lost", tracked=n_tracked, matched=len(good), inliers=0, confidence_scale=0.0)

        inlier_count, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, K, mask=mask)

        yaw = float(np.arctan2(R[0, 2], R[0, 0]))
        local_forward = self.assumed_forward_speed * dt * float(t[2, 0])
        local_lateral = self.assumed_forward_speed * dt * float(t[0, 0])

        theta = self._pose.heading
        self._pose = RelativePose(
            x=self._pose.x + local_forward * np.cos(theta) + local_lateral * np.sin(theta),
            y=self._pose.y + local_forward * np.sin(theta) - local_lateral * np.cos(theta),
            heading=wrap_angle(theta + yaw),
        )

        self._prev_gray, self._prev_keypoints, self._prev_descriptors = gray, keypoints, descriptors

        inlier_ratio = inlier_count / max(1, len(good))
        feature_component = float(np.clip(n_tracked / self.target_feature_count, 0.0, 1.0))
        match_component = float(np.clip(len(good) / self.target_match_count, 0.0, 1.0))
        confidence_scale = 0.3 * feature_component + 0.3 * match_component + 0.4 * inlier_ratio

        return self._result(
            status="tracking",
            tracked=n_tracked,
            matched=len(good),
            inliers=int(inlier_count),
            confidence_scale=confidence_scale,
        )

    def _result(self, status: str, tracked: int, matched: int, inliers: int, confidence_scale: float) -> LocalizationResult:
        return LocalizationResult(
            timestamp=time.time(),
            mode=self.mode,
            pose=RelativePose(x=self._pose.x, y=self._pose.y, heading=self._pose.heading),
            confidence=float(np.clip(confidence_scale, 0.0, 1.0)),
            tracked_features=tracked,
            matched_features=matched,
            inliers=inliers,
            status=status,
        )
