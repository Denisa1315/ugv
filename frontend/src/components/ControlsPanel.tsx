import { useState } from "react";
import { api } from "../api";
import { Panel } from "./Panel";

const buttonBase =
  "rounded px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40";

interface Props {
  autonomousEnabled: boolean;
  onAutonomousChange: (enabled: boolean) => void;
}

export function ControlsPanel({ autonomousEnabled, onAutonomousChange }: Props) {
  const [obstacleX, setObstacleX] = useState("10");
  const [obstacleY, setObstacleY] = useState("10");
  const [busy, setBusy] = useState(false);

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel title="Controls">
      <div className="mb-3 grid grid-cols-2 gap-2">
        <button
          className={`${buttonBase} bg-green-600 hover:bg-green-500`}
          disabled={busy}
          onClick={() => run(api.start)}
        >
          ▶ Start
        </button>
        <button className={`${buttonBase} bg-yellow-600 hover:bg-yellow-500`} disabled={busy} onClick={() => run(api.pause)}>
          ⏸ Pause
        </button>
        <button className={`${buttonBase} bg-red-600 hover:bg-red-500`} disabled={busy} onClick={() => run(api.stop)}>
          ⏹ Stop
        </button>
        <button
          className={`${buttonBase} bg-slate-600 hover:bg-slate-500`}
          disabled={busy}
          onClick={() => run(() => api.reset())}
        >
          ↺ Reset
        </button>
      </div>

      <div className="mb-3 border-t border-slate-800 pt-2">
        {!autonomousEnabled ? (
          <button
            className={`${buttonBase} w-full bg-sky-600 hover:bg-sky-500`}
            disabled={busy}
            onClick={() =>
              run(async () => {
                await api.enableAutonomous();
                onAutonomousChange(true);
              })
            }
          >
            Enable Autonomous Mode
          </button>
        ) : (
          <button
            className={`${buttonBase} w-full bg-slate-700 hover:bg-slate-600`}
            disabled={busy}
            onClick={() =>
              run(async () => {
                await api.disableAutonomous();
                onAutonomousChange(false);
              })
            }
          >
            Disable Autonomous Mode
          </button>
        )}
      </div>

      <div className="mb-3 border-t border-slate-800 pt-2">
        <div className="mb-1 text-xs text-slate-400">Add Obstacle (x, y)</div>
        <div className="flex gap-2">
          <input
            className="w-16 rounded bg-slate-800 px-2 py-1 text-sm"
            value={obstacleX}
            onChange={(e) => setObstacleX(e.target.value)}
          />
          <input
            className="w-16 rounded bg-slate-800 px-2 py-1 text-sm"
            value={obstacleY}
            onChange={(e) => setObstacleY(e.target.value)}
          />
          <button
            className={`${buttonBase} flex-1 bg-indigo-600 hover:bg-indigo-500`}
            disabled={busy}
            onClick={() => run(() => api.addObstacle(parseFloat(obstacleX), parseFloat(obstacleY), 0.6))}
          >
            Add
          </button>
        </div>
      </div>

      <div className="border-t border-slate-800 pt-2">
        <button
          className={`${buttonBase} w-full bg-orange-600 hover:bg-orange-500`}
          disabled={busy || !autonomousEnabled}
          onClick={() => run(() => api.simulateLowConfidence(5.0, 0.08))}
          title={!autonomousEnabled ? "Enable autonomous mode first" : ""}
        >
          ⚠ Simulate Low Confidence
        </button>
      </div>
    </Panel>
  );
}
