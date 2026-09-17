import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_demo_state_idle_before_start(client):
    resp = client.get("/demo/state")
    assert resp.status_code == 200
    assert resp.json()["status"] == "IDLE"


def test_demo_start_transitions_to_running(client):
    resp = client.post("/demo/start")
    assert resp.status_code == 200
    time.sleep(0.5)

    state = client.get("/demo/state").json()
    assert state["status"] == "RUNNING"
    assert len(state["log"]) >= 1
    assert "mission started" in state["log"][0]

    # Don't wait for the full ~90s scripted run in this fast API test —
    # the full run is covered by test_demo_scenario.py. Just confirm the
    # simulator is actually being driven by it.
    sim_state = client.get("/simulator/state").json()
    assert sim_state["status"] == "running"
