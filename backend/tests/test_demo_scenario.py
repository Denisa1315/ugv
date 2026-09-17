import asyncio
import time
from types import SimpleNamespace

import pytest

from app.modules.demo.scenario import DemoScenario
from app.modules.simulator.environment import Environment
from app.modules.simulator.simulator import Simulator


@pytest.mark.asyncio
async def test_demo_scenario_runs_all_scripted_moments_and_succeeds():
    """Slow, real end-to-end run (~90s wall-clock) of the actual scripted
    SIH demo — the whole point is to prove the milestone sequence really
    fires and the mission really completes, not just that the code parses."""
    sim = Simulator(environment=Environment(width=20, height=20, goal=(18.0, 18.0)), tick_hz=10.0)
    sim.reset(x=1.0, y=1.0, heading=0.0)
    app_state = SimpleNamespace(simulator=sim, pipeline=None)
    demo = DemoScenario(app_state, db_path="/tmp/ugv_test_demo_events.db")

    demo.start()
    deadline = time.time() + 150
    while demo.state.status == "RUNNING" and time.time() < deadline:
        await asyncio.sleep(1.0)

    assert demo.state.status == "SUCCESS", f"log: {demo.state.log}"

    joined_log = "\n".join(demo.state.log)
    assert "Obstacle appeared" in joined_log
    assert "confidence degradation" in joined_log
    assert "emergency clearance" in joined_log
    assert "Hazard cleared" in joined_log
    assert "Point B reached" in joined_log
    assert sim.collided is False
    assert sim.goal_reached is True
