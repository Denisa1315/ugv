import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_events_endpoint_empty_before_autonomous_enabled(client):
    resp = client.get("/telemetry/events")
    assert resp.status_code == 200
    assert resp.json() == {"events": []}


def test_events_populate_after_running(client, tmp_path, monkeypatch):
    client.post("/simulator/reset", json={"x": 1.0, "y": 1.0, "heading": 0.0})
    client.post("/pipeline/enable_autonomous", json={"camera_mode": "demo", "perception_mode": "fallback"})
    client.post("/simulator/start")
    time.sleep(1.0)
    client.post("/simulator/stop")

    resp = client.get("/telemetry/events")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert len(events) >= 1
    first = events[0]
    assert "event_type" in first
    assert "severity" in first
    assert "position" in first
    assert "risk" in first
    assert "confidence" in first
    assert "description" in first


def test_camera_frame_png_available_after_running(client):
    client.post("/pipeline/enable_autonomous", json={"camera_mode": "demo", "perception_mode": "fallback"})
    client.post("/simulator/start")
    time.sleep(0.5)
    client.post("/simulator/stop")

    resp = client.get("/pipeline/camera_frame.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_camera_frame_before_any_cycle_is_409(client):
    client.post("/pipeline/enable_autonomous", json={"camera_mode": "demo", "perception_mode": "fallback"})
    resp = client.get("/pipeline/camera_frame.png")
    assert resp.status_code == 409


def test_websocket_streams_simulator_and_pipeline_state(client):
    client.post("/simulator/reset", json={"x": 1.0, "y": 1.0, "heading": 0.0})
    client.post("/pipeline/enable_autonomous", json={"camera_mode": "demo", "perception_mode": "fallback"})
    client.post("/simulator/start")

    with client.websocket_connect("/ws/telemetry") as ws:
        message = ws.receive_json()
        assert "simulator" in message
        assert "vehicle" in message["simulator"]
        assert "x" in message["simulator"]["vehicle"]

        # A second message should reflect the pipeline once a cycle has run.
        second = ws.receive_json()
        assert "pipeline" in second or "pipeline" in message
        payload_with_pipeline = second if "pipeline" in second else message
        assert "decision" in payload_with_pipeline["pipeline"]
        assert "state" in payload_with_pipeline["pipeline"]["decision"]

    client.post("/simulator/stop")
