import { useEffect, useState } from "react";
import { api } from "../api";
import type { DemoState } from "../types";
import { Panel } from "./Panel";

const STATUS_COLOR: Record<string, string> = {
  IDLE: "text-slate-400",
  RUNNING: "text-sky-400",
  SUCCESS: "text-green-400",
  FAILED: "text-red-400",
  TIMEOUT: "text-orange-400",
};

export function DemoPanel({ onStarted }: { onStarted: () => void }) {
  const [demo, setDemo] = useState<DemoState | null>(null);

  useEffect(() => {
    const poll = () => api.demoState().then((r) => setDemo(r.data)).catch(() => {});
    poll();
    const interval = setInterval(poll, 800);
    return () => clearInterval(interval);
  }, []);

  const running = demo?.status === "RUNNING";

  return (
    <Panel title="SIH Demo Mode" className="border-sky-700">
      <button
        className="w-full rounded bg-gradient-to-r from-sky-600 to-indigo-600 px-3 py-2 text-sm font-bold tracking-wide hover:from-sky-500 hover:to-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
        disabled={running}
        onClick={async () => {
          await api.startDemo();
          onStarted();
        }}
      >
        {running ? "Demo Running…" : "▶ Start SIH Demo"}
      </button>

      {demo && demo.status !== "IDLE" && (
        <div className="mt-2">
          <div className={`text-sm font-bold ${STATUS_COLOR[demo.status]}`}>
            {demo.status} — {demo.phase}
          </div>
          <div className="mt-1 max-h-40 space-y-0.5 overflow-y-auto text-[11px] text-slate-400">
            {demo.log.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}
