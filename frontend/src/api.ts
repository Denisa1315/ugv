import axios from "axios";
import type { DemoState, EventLogEntry, SimulatorState } from "./types";

export const BACKEND_HTTP = "http://127.0.0.1:8000";
export const BACKEND_WS = "ws://127.0.0.1:8000";

const client = axios.create({ baseURL: BACKEND_HTTP });

export const api = {
  health: () => client.get("/health"),

  simulatorState: () => client.get<SimulatorState>("/simulator/state"),
  start: () => client.post("/simulator/start"),
  pause: () => client.post("/simulator/pause"),
  stop: () => client.post("/simulator/stop"),
  reset: (x = 1, y = 1, heading = 0) => client.post("/simulator/reset", { x, y, heading }),
  addObstacle: (x: number, y: number, radius = 0.5) => client.post("/simulator/obstacle", { x, y, radius }),
  clearObstacles: () => client.delete("/simulator/obstacles"),

  enableAutonomous: (camera_mode = "demo", perception_mode = "fallback") =>
    client.post("/pipeline/enable_autonomous", { camera_mode, perception_mode }),
  disableAutonomous: () => client.post("/pipeline/disable_autonomous"),
  simulateLowConfidence: (duration_s = 5.0, forced_value = 0.08) =>
    client.post("/pipeline/simulate_low_confidence", { duration_s, forced_value }),

  events: (limit = 50) => client.get<{ events: EventLogEntry[] }>("/telemetry/events", { params: { limit } }),

  startDemo: () => client.post("/demo/start"),
  demoState: () => client.get<DemoState>("/demo/state"),

  cameraFrameUrl: () => `${BACKEND_HTTP}/pipeline/camera_frame.png?t=${Date.now()}`,
  riskHeatmapUrl: () => `${BACKEND_HTTP}/pipeline/risk_heatmap.png?t=${Date.now()}`,
};
