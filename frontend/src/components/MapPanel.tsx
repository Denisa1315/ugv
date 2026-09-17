import { useEffect, useState } from "react";
import { api } from "../api";
import type { PipelineState, SimulatorState } from "../types";
import { Panel } from "./Panel";

interface Props {
  simulator: SimulatorState | null;
  pipeline: PipelineState | null;
  autonomousEnabled: boolean;
}

/** Risk heatmap PNG as the background layer, with an SVG overlay (vehicle,
 * goal, obstacles, planned path, trajectory) drawn from live telemetry —
 * every element here comes from the current WebSocket frame, nothing static. */
export function MapPanel({ simulator, pipeline, autonomousEnabled }: Props) {
  const [heatmapSrc, setHeatmapSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!autonomousEnabled) return;
    const interval = setInterval(() => setHeatmapSrc(api.riskHeatmapUrl()), 500);
    return () => clearInterval(interval);
  }, [autonomousEnabled]);

  if (!simulator) {
    return (
      <Panel title="2D Map">
        <div className="flex aspect-square items-center justify-center text-xs text-slate-500">Waiting for simulator…</div>
      </Panel>
    );
  }

  const { width, height, goal, obstacles } = simulator.environment;
  const flipY = (y: number) => height - y;
  const vehicle = simulator.vehicle;
  const path = pipeline?.planning.active_path ?? [];
  const trajectory = simulator.trajectory;

  return (
    <Panel title="2D Map — UGV / Goal / Obstacles / Risk / Path">
      <div className="relative aspect-square w-full overflow-hidden rounded bg-slate-950">
        {autonomousEnabled && heatmapSrc && (
          <img
            src={heatmapSrc}
            alt="Risk heatmap"
            className="absolute inset-0 h-full w-full object-fill opacity-70"
            style={{ transform: "scaleY(-1)" }}
          />
        )}
        <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full">
          {/* grid */}
          {Array.from({ length: Math.floor(width) + 1 }).map((_, i) => (
            <line key={`vx${i}`} x1={i} y1={0} x2={i} y2={height} stroke="#1e293b" strokeWidth={0.02} />
          ))}
          {Array.from({ length: Math.floor(height) + 1 }).map((_, i) => (
            <line key={`hz${i}`} x1={0} y1={i} x2={width} y2={i} stroke="#1e293b" strokeWidth={0.02} />
          ))}

          {/* trajectory trail */}
          {trajectory.length > 1 && (
            <polyline
              points={trajectory.map(([x, y]) => `${x},${flipY(y)}`).join(" ")}
              fill="none"
              stroke="#4ade80"
              strokeWidth={0.08}
              opacity={0.7}
            />
          )}

          {/* planned path */}
          {path.length > 1 && (
            <polyline
              points={path.map(([x, y]) => `${x},${flipY(y)}`).join(" ")}
              fill="none"
              stroke="#38bdf8"
              strokeWidth={0.1}
              strokeDasharray="0.3,0.2"
            />
          )}

          {/* obstacles */}
          {obstacles.map((o) => (
            <circle key={o.id} cx={o.x} cy={flipY(o.y)} r={o.radius} fill="#ef4444" stroke="#7f1d1d" strokeWidth={0.05} />
          ))}

          {/* goal */}
          <g transform={`translate(${goal[0]},${flipY(goal[1])})`}>
            <path d="M0,-0.5 L0.15,-0.15 L0.5,0 L0.15,0.15 L0,0.5 L-0.15,0.15 L-0.5,0 L-0.15,-0.15 Z" fill="#facc15" />
          </g>

          {/* vehicle */}
          <g transform={`translate(${vehicle.x},${flipY(vehicle.y)}) rotate(${-((vehicle.heading * 180) / Math.PI)})`}>
            <path d="M0.35,0 L-0.2,0.2 L-0.2,-0.2 Z" fill="#f472b6" stroke="#831843" strokeWidth={0.03} />
          </g>
        </svg>
      </div>
      <p className="mt-1 text-[10px] text-slate-500">
        goal ({goal[0].toFixed(1)}, {goal[1].toFixed(1)}) · {obstacles.length} obstacle(s) · path {path.length} pts
      </p>
    </Panel>
  );
}
