"""Real-time telemetry: a polling WebSocket broadcast (simple or, for a
single-process hackathon demo, pub/sub) plus REST access to the durable
SQLite event log.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["telemetry"])

BROADCAST_INTERVAL_S = 0.1  # ~10Hz, matches the default simulator tick rate


def build_telemetry_payload(app_state) -> dict:
    simulator = app_state.simulator
    payload: dict = {
        "timestamp": simulator.vehicle.state.timestamp,
        "simulator": simulator.get_state(),
    }

    pipeline = getattr(app_state, "pipeline", None)
    if pipeline is not None and pipeline.last_cycle is not None:
        cycle = pipeline.last_cycle
        payload["pipeline"] = {
            "perception": {
                "confidence": cycle.perception_confidence,
                "mode": cycle.perception_mode,
                "num_obstacles": cycle.num_obstacles,
            },
            "depth": {"confidence": cycle.depth_confidence, "num_terrain_deviations": cycle.num_terrain_deviations},
            "localization": {
                "confidence": cycle.localization_confidence,
                "status": cycle.localization_status,
                "pose": {
                    "x": cycle.localization_pose.x,
                    "y": cycle.localization_pose.y,
                    "heading": cycle.localization_pose.heading,
                },
            },
            "planning": {
                "goal_reachable": cycle.goal_reachable,
                "replanned_this_cycle": cycle.replanned,
                "blocked": cycle.blocked,
                "goal_reached": cycle.goal_reached,
                "reason": cycle.plan_reason,
                "total_replans": cycle.replan_count_total,
                "active_path": pipeline.last_active_path,
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
    return payload


@router.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(build_telemetry_payload(websocket.app.state))
            await asyncio.sleep(BROADCAST_INTERVAL_S)
    except WebSocketDisconnect:
        pass


@router.get("/telemetry/events")
def get_events(request: Request, limit: int = Query(default=50, le=500)) -> dict:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return {"events": []}
    events = pipeline.event_logger.recent(limit=limit)
    return {"events": [e.as_dict() for e in events]}
