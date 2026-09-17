import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_state_before_enabling_returns_409(client):
    resp = client.get("/pipeline/state")
    assert resp.status_code == 409


def test_enable_autonomous_and_drive(client):
    client.post("/simulator/reset", json={"x": 1.0, "y": 1.0, "heading": 0.0})
    resp = client.post("/pipeline/enable_autonomous", json={"camera_mode": "demo", "perception_mode": "fallback"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "autonomous_enabled"

    client.post("/simulator/start")
    time.sleep(1.0)
    client.post("/simulator/stop")

    state = client.get("/pipeline/state").json()
    assert "cmd_vel" in state
    assert 0.0 <= state["perception"]["confidence"] <= 1.0
    assert 0.0 <= state["localization"]["confidence"] <= 1.0

    sim_state = client.get("/simulator/state").json()
    assert sim_state["tick_count"] > 0


def test_risk_heatmap_is_a_real_png(client):
    client.post("/pipeline/enable_autonomous", json={"camera_mode": "demo", "perception_mode": "fallback"})
    resp = client.get("/pipeline/risk_heatmap.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_disable_autonomous_stops_pipeline_control(client):
    client.post("/pipeline/enable_autonomous", json={"camera_mode": "demo", "perception_mode": "fallback"})
    resp = client.post("/pipeline/disable_autonomous")
    assert resp.status_code == 200
    state = client.get("/simulator/state").json()
    assert state["cmd_vel"] == {"linear": 0.0, "angular": 0.0}
