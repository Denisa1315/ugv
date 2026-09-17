from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.modules.simulator.renderer import render_topdown_png
from app.modules.simulator.simulator import Simulator

router = APIRouter(prefix="/simulator", tags=["simulator"])


class CmdVelRequest(BaseModel):
    linear: float = Field(..., description="m/s, +forward")
    angular: float = Field(..., description="rad/s, +counter-clockwise")


class AddObstacleRequest(BaseModel):
    x: float
    y: float
    radius: float = 0.5


class ResetRequest(BaseModel):
    x: float = 1.0
    y: float = 1.0
    heading: float = 0.0


def get_simulator(request: Request) -> Simulator:
    return request.app.state.simulator


@router.get("/state")
def get_state(request: Request) -> dict:
    return get_simulator(request).get_state()


@router.post("/start")
async def start(request: Request) -> dict:
    # Must run on the event-loop thread (not FastAPI's sync threadpool) so
    # asyncio.create_task() in ensure_running_task() has a loop to attach to.
    sim = get_simulator(request)
    sim.start()
    sim.ensure_running_task()
    return sim.get_state()


@router.post("/pause")
def pause(request: Request) -> dict:
    sim = get_simulator(request)
    sim.pause()
    return sim.get_state()


@router.post("/stop")
def stop(request: Request) -> dict:
    sim = get_simulator(request)
    sim.stop()
    return sim.get_state()


@router.post("/reset")
def reset(request: Request, body: ResetRequest = ResetRequest()) -> dict:
    sim = get_simulator(request)
    sim.reset(x=body.x, y=body.y, heading=body.heading)
    return sim.get_state()


@router.post("/cmd_vel")
def set_cmd_vel(request: Request, body: CmdVelRequest) -> dict:
    """Manual velocity override for testing without a planner.

    From phase 1j (local planner) onward, the planner writes here every
    cycle instead of a human/test client.
    """
    sim = get_simulator(request)
    sim.set_cmd_vel(body.linear, body.angular)
    return sim.get_state()


@router.post("/obstacle")
def add_obstacle(request: Request, body: AddObstacleRequest) -> dict:
    sim = get_simulator(request)
    obstacle = sim.environment.add_obstacle(body.x, body.y, body.radius)
    return {"id": obstacle.id, "x": obstacle.x, "y": obstacle.y, "radius": obstacle.radius}


@router.delete("/obstacles")
def clear_obstacles(request: Request) -> dict:
    sim = get_simulator(request)
    sim.environment.clear_obstacles()
    return sim.get_state()


@router.get("/render.png")
def render_png(request: Request) -> Response:
    sim = get_simulator(request)
    png_bytes = render_topdown_png(sim)
    return Response(content=png_bytes, media_type="image/png")
