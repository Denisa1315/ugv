import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    # Context-manager form so FastAPI's lifespan (startup) actually runs and
    # populates app.state.simulator before any request hits the app.
    with TestClient(app) as c:
        yield c


def test_state_endpoint_returns_initial_state(client):
    resp = client.get("/simulator/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "idle"
    assert body["vehicle"]["x"] == 1.0
    assert body["vehicle"]["y"] == 1.0


def test_add_obstacle_and_clear(client):
    resp = client.post("/simulator/obstacle", json={"x": 4.0, "y": 4.0, "radius": 0.6})
    assert resp.status_code == 200
    obstacle_id = resp.json()["id"]
    state = client.get("/simulator/state").json()
    assert any(o["id"] == obstacle_id for o in state["environment"]["obstacles"])

    resp = client.delete("/simulator/obstacles")
    assert resp.json()["environment"]["obstacles"] == []


def test_cmd_vel_validation_rejects_missing_fields(client):
    resp = client.post("/simulator/cmd_vel", json={"linear": 1.0})
    assert resp.status_code == 422


def test_render_png_is_a_real_png(client):
    resp = client.get("/simulator/render.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(resp.content) > 500


def test_start_actually_advances_state_over_wall_clock_time(client):
    client.post("/simulator/reset", json={"x": 0.0, "y": 0.0, "heading": 0.0})
    client.post("/simulator/cmd_vel", json={"linear": 1.0, "angular": 0.0})
    client.post("/simulator/start")
    time.sleep(0.8)
    state = client.get("/simulator/state").json()
    client.post("/simulator/stop")

    assert state["status"] == "running"
    assert state["vehicle"]["x"] > 0.3  # ~1 m/s for ~0.8s, background loop actually ran
    assert state["tick_count"] > 0
