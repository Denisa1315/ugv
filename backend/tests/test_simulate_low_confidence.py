import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_simulate_low_confidence_forces_perception_confidence_down(client):
    client.post("/simulator/reset", json={"x": 1.0, "y": 1.0, "heading": 0.0})
    client.post("/pipeline/enable_autonomous", json={"camera_mode": "demo", "perception_mode": "fallback"})
    client.post("/simulator/start")
    time.sleep(0.3)  # let a cycle run so perception has a real (higher) baseline

    resp = client.post("/pipeline/simulate_low_confidence", json={"duration_s": 2.0, "forced_value": 0.05})
    assert resp.status_code == 200

    time.sleep(0.3)
    state = client.get("/pipeline/state").json()
    assert state["perception"]["confidence"] <= 0.05

    client.post("/simulator/stop")


def test_confidence_recovers_after_fault_window_expires(client):
    client.post("/pipeline/enable_autonomous", json={"camera_mode": "demo", "perception_mode": "fallback"})
    client.post("/simulator/start")

    client.post("/pipeline/simulate_low_confidence", json={"duration_s": 0.3, "forced_value": 0.02})
    time.sleep(0.2)
    degraded_state = client.get("/pipeline/state").json()
    assert degraded_state["perception"]["confidence"] <= 0.02

    time.sleep(1.0)  # fault window has expired
    recovered_state = client.get("/pipeline/state").json()
    assert recovered_state["perception"]["confidence"] > 0.02

    client.post("/simulator/stop")
