from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import get_settings
from app.modules.demo.scenario import DemoScenario

router = APIRouter(prefix="/demo", tags=["demo"])


@router.post("/start")
async def start_demo(request: Request) -> dict:
    # Must run on the event-loop thread (not FastAPI's sync threadpool) so
    # asyncio.create_task() inside DemoScenario.start() has a loop to attach to.
    if getattr(request.app.state, "demo", None) is None:
        request.app.state.demo = DemoScenario(request.app.state, db_path=get_settings().db_path)
    request.app.state.demo.start()
    return {"status": "started"}


@router.get("/state")
def demo_state(request: Request) -> dict:
    demo = getattr(request.app.state, "demo", None)
    if demo is None:
        return {"status": "IDLE", "phase": "idle", "log": []}
    return {"status": demo.state.status, "phase": demo.state.phase, "log": demo.state.log}
