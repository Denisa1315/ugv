import { useEffect, useState } from "react";
import { api } from "../api";
import type { EventLogEntry } from "../types";
import { Panel } from "./Panel";

const SEVERITY_COLOR: Record<string, string> = {
  INFO: "text-slate-300",
  WARNING: "text-yellow-400",
  CRITICAL: "text-red-400",
};

export function EventLogPanel() {
  const [events, setEvents] = useState<EventLogEntry[]>([]);

  useEffect(() => {
    const poll = () => api.events(50).then((r) => setEvents(r.data.events)).catch(() => {});
    poll();
    const interval = setInterval(poll, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <Panel title="Event Log">
      <div className="max-h-64 space-y-1 overflow-y-auto text-xs">
        {events.length === 0 && <div className="text-slate-500">No events yet</div>}
        {events.map((e) => (
          <div key={e.id} className="border-b border-slate-800 py-1 last:border-0">
            <div className="flex items-center justify-between">
              <span className={`font-semibold ${SEVERITY_COLOR[e.severity] ?? ""}`}>{e.event_type}</span>
              <span className="text-slate-500">{new Date(e.timestamp * 1000).toLocaleTimeString()}</span>
            </div>
            <div className="text-slate-400">
              {e.description} — pos ({e.position.x.toFixed(1)}, {e.position.y.toFixed(1)}) risk {e.risk.toFixed(2)} conf{" "}
              {e.confidence.toFixed(2)}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
