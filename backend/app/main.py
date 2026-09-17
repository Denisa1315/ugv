from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.demo import router as demo_router
from app.api.health import router as health_router
from app.api.pipeline import router as pipeline_router
from app.api.simulator import router as simulator_router
from app.api.telemetry import router as telemetry_router
from app.config import get_settings
from app.modules.simulator.environment import Environment
from app.modules.simulator.simulator import Simulator

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.simulator = Simulator(
        environment=Environment(
            width=settings.env_width,
            height=settings.env_height,
            goal=tuple(settings.env_goal),
        ),
        tick_hz=settings.simulator_tick_hz,
    )
    app.state.simulator.reset(x=1.0, y=1.0, heading=0.0)
    app.state.pipeline = None
    app.state.demo = None
    yield


app = FastAPI(
    title="UGV Autonomous Navigation Backend",
    description=(
        "Vision-based autonomous navigation pipeline for an outdoor UGV "
        "(SIH 2026, PS 26126). Single-process FastAPI backend; internal "
        "data channels are named after their eventual ROS 2 topic names "
        "(e.g. slam/pose, navigation/risk_map, cmd_vel) as documentation "
        "only — there is no pub/sub or ROS 2 runtime here."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(simulator_router)
app.include_router(pipeline_router)
app.include_router(telemetry_router)
app.include_router(demo_router)
