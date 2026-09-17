from __future__ import annotations

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import get_settings
from app.modules.camera.factory import CameraMode
from app.modules.perception.factory import PerceptionMode
from app.modules.pipeline.factory import create_pipeline
from app.modules.pipeline.orchestrator import NavigationPipeline
from app.modules.risk.render import render_risk_heatmap_png
from app.modules.simulator.simulator import Simulator

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class EnableAutonomousRequest(BaseModel):
    camera_mode: CameraMode = "demo"
    perception_mode: PerceptionMode = "fallback"


class SimulateLowConfidenceRequest(BaseModel):
    duration_s: float = 5.0
    forced_value: float = 0.1


def get_simulator(request: Request) -> Simulator:
    return request.app.state.simulator


@router.post("/enable_autonomous")
def enable_autonomous(request: Request, body: EnableAutonomousRequest = EnableAutonomousRequest()) -> dict:
    """Wires camera -> perception -> depth -> localization -> risk ->
    corridor -> planners into the simulator's real-time tick loop."""
    sim = get_simulator(request)

    pipeline = create_pipeline(
        sim.environment,
        camera_mode=body.camera_mode,
        perception_mode=body.perception_mode,
        db_path=get_settings().db_path,
    )
    request.app.state.pipeline = pipeline
    sim.pre_tick_hook = pipeline.as_pre_tick_hook(sim)

    return {"status": "autonomous_enabled", "camera_mode": body.camera_mode, "perception_mode": body.perception_mode}


@router.post("/disable_autonomous")
def disable_autonomous(request: Request) -> dict:
    sim = get_simulator(request)
    sim.pre_tick_hook = None
    sim.set_cmd_vel(0.0, 0.0)
    return {"status": "autonomous_disabled"}


def _get_pipeline(request: Request) -> NavigationPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=409, detail="Autonomous pipeline not enabled — call /pipeline/enable_autonomous first")
    return pipeline


@router.get("/state")
def get_state(request: Request) -> dict:
    pipeline = _get_pipeline(request)
    cycle = pipeline.last_cycle
    if cycle is None:
        return {"status": "no_cycle_yet"}
    return {
        "timestamp": cycle.timestamp,
        "perception": {"confidence": cycle.perception_confidence, "mode": cycle.perception_mode, "num_obstacles": cycle.num_obstacles},
        "depth": {"confidence": cycle.depth_confidence, "num_terrain_deviations": cycle.num_terrain_deviations},
        "localization": {
            "confidence": cycle.localization_confidence,
            "status": cycle.localization_status,
            "pose": {"x": cycle.localization_pose.x, "y": cycle.localization_pose.y, "heading": cycle.localization_pose.heading},
        },
        "planning": {
            "goal_reachable": cycle.goal_reachable,
            "replanned_this_cycle": cycle.replanned,
            "blocked": cycle.blocked,
            "goal_reached": cycle.goal_reached,
            "reason": cycle.plan_reason,
            "active_path_length": cycle.active_path_length,
            "active_path": pipeline.last_active_path,
            "total_replans": cycle.replan_count_total,
        },
        "decision": {
            "state": cycle.system_state.value,
            "action": cycle.action.value,
            "message": cycle.decision_message,
            "emergency_triggered": cycle.emergency_triggered,
            "raw_min_clearance_m": cycle.raw_min_clearance_m,
            "local_risk": cycle.local_risk,
        },
        "cmd_vel": {"linear": cycle.cmd_vel.linear, "angular": cycle.cmd_vel.angular},
    }


@router.post("/simulate_low_confidence")
def simulate_low_confidence(request: Request, body: SimulateLowConfidenceRequest = SimulateLowConfidenceRequest()) -> dict:
    pipeline = _get_pipeline(request)
    pipeline.simulate_low_confidence(duration_s=body.duration_s, forced_value=body.forced_value)
    return {"status": "fault_injected", "duration_s": body.duration_s, "forced_value": body.forced_value}


@router.get("/risk_heatmap.png")
def risk_heatmap(request: Request) -> Response:
    pipeline = _get_pipeline(request)
    png_bytes = render_risk_heatmap_png(pipeline.risk_grid)
    return Response(content=png_bytes, media_type="image/png")


@router.get("/camera_frame.png")
def camera_frame(request: Request) -> Response:
    """The latest annotated perception frame — segmentation/detection
    overlay drawn on the actual camera input (real or demo)."""
    pipeline = _get_pipeline(request)
    if pipeline.last_annotated_frame is None:
        raise HTTPException(status_code=409, detail="No frame processed yet")
    ok, buf = cv2.imencode(".png", pipeline.last_annotated_frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=buf.tobytes(), media_type="image/png")
