export interface VehicleState {
  x: number;
  y: number;
  heading: number;
  velocity: number;
  angular_velocity: number;
  timestamp: number;
}

export interface ObstacleState {
  id: string;
  x: number;
  y: number;
  radius: number;
}

export interface SimulatorState {
  status: "idle" | "running" | "paused" | "stopped";
  vehicle: VehicleState;
  cmd_vel: { linear: number; angular: number };
  environment: {
    width: number;
    height: number;
    goal: [number, number];
    obstacles: ObstacleState[];
  };
  trajectory: [number, number][];
  elapsed_time: number;
  tick_count: number;
  goal_reached: boolean;
  collided: boolean;
}

export type SystemStateName = "SAFE" | "WARNING" | "DEGRADED" | "CRITICAL";
export type ActionName = "RUN" | "SLOW" | "REPLAN" | "PAUSE";

export interface PipelineState {
  timestamp: number;
  perception: { confidence: number; mode: string; num_obstacles: number };
  depth: { confidence: number; num_terrain_deviations: number };
  localization: {
    confidence: number;
    status: string;
    pose: { x: number; y: number; heading: number };
  };
  planning: {
    goal_reachable: boolean;
    replanned_this_cycle: boolean;
    blocked: boolean;
    goal_reached: boolean;
    reason: string;
    active_path_length: number;
    active_path: [number, number][];
    total_replans: number;
  };
  decision: {
    state: SystemStateName;
    action: ActionName;
    message: string;
    emergency_triggered: boolean;
    raw_min_clearance_m: number | null;
    local_risk: number;
  };
  cmd_vel: { linear: number; angular: number };
}

export interface TelemetryMessage {
  timestamp: number;
  simulator: SimulatorState;
  pipeline?: PipelineState;
}

export interface EventLogEntry {
  id: number;
  timestamp: number;
  event_type: string;
  severity: "INFO" | "WARNING" | "CRITICAL";
  position: { x: number; y: number };
  risk: number;
  confidence: number;
  description: string;
}

export interface DemoState {
  status: "IDLE" | "RUNNING" | "SUCCESS" | "FAILED" | "TIMEOUT";
  phase: string;
  log: string[];
}
