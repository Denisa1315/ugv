import type { PipelineState, SimulatorState } from "../types";
import { STATE_BG, confidenceColor } from "../statusColors";
import { Panel } from "./Panel";

function Stat({ label, value, colorClass = "text-slate-200" }: { label: string; value: string; colorClass?: string }) {
  return (
    <div className="flex items-center justify-between border-b border-slate-800 py-1 text-sm last:border-0">
      <span className="text-slate-400">{label}</span>
      <span className={`font-mono ${colorClass}`}>{value}</span>
    </div>
  );
}

export function StatusPanel({ simulator, pipeline }: { simulator: SimulatorState | null; pipeline: PipelineState | null }) {
  const decision = pipeline?.decision;

  return (
    <Panel title="System Status">
      {decision && (
        <div className={`mb-3 rounded border px-3 py-2 text-center ${STATE_BG[decision.state]}`}>
          <div className="text-lg font-bold tracking-wide">{decision.state}</div>
          <div className="text-xs opacity-80">action: {decision.action}</div>
          <div className="mt-1 text-[11px] opacity-90">{decision.message}</div>
        </div>
      )}

      <Stat label="GPS" value="DENIED" colorClass="text-red-400" />
      <Stat
        label="Localization"
        value={pipeline ? `${(pipeline.localization.confidence * 100).toFixed(0)}% (${pipeline.localization.status})` : "—"}
        colorClass={pipeline ? confidenceColor(pipeline.localization.confidence) : undefined}
      />
      <Stat
        label="Perception"
        value={pipeline ? `${(pipeline.perception.confidence * 100).toFixed(0)}% (${pipeline.perception.mode})` : "—"}
        colorClass={pipeline ? confidenceColor(pipeline.perception.confidence) : undefined}
      />
      <Stat
        label="Depth"
        value={pipeline ? `${(pipeline.depth.confidence * 100).toFixed(0)}%` : "—"}
        colorClass={pipeline ? confidenceColor(pipeline.depth.confidence) : undefined}
      />
      <Stat
        label="Local Risk"
        value={pipeline ? pipeline.decision.local_risk.toFixed(2) : "—"}
        colorClass={pipeline ? confidenceColor(1 - pipeline.decision.local_risk) : undefined}
      />
      <Stat
        label="Raw Clearance"
        value={pipeline?.decision.raw_min_clearance_m != null ? `${pipeline.decision.raw_min_clearance_m.toFixed(2)} m` : "—"}
      />
      <Stat label="Speed" value={simulator ? `${simulator.vehicle.velocity.toFixed(2)} m/s` : "—"} />
      <Stat label="Heading" value={simulator ? `${((simulator.vehicle.heading * 180) / Math.PI).toFixed(0)}°` : "—"} />
      <Stat label="Obstacles seen" value={pipeline ? `${pipeline.perception.num_obstacles}` : "—"} />
      <Stat label="Terrain deviations" value={pipeline ? `${pipeline.depth.num_terrain_deviations}` : "—"} />
      <Stat label="Replans" value={pipeline ? `${pipeline.planning.total_replans}` : "—"} />
      <Stat
        label="Sim status"
        value={simulator ? simulator.status : "—"}
        colorClass={simulator?.goal_reached ? "text-green-400" : simulator?.collided ? "text-red-400" : undefined}
      />
      {simulator?.goal_reached && <div className="mt-2 text-center text-sm font-bold text-green-400">✔ GOAL REACHED</div>}
      {simulator?.collided && <div className="mt-2 text-center text-sm font-bold text-red-400">✖ COLLISION</div>}
    </Panel>
  );
}
