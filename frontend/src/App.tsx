import { useState } from "react";
import { CameraPanel } from "./components/CameraPanel";
import { ControlsPanel } from "./components/ControlsPanel";
import { DemoPanel } from "./components/DemoPanel";
import { EventLogPanel } from "./components/EventLogPanel";
import { MapPanel } from "./components/MapPanel";
import { StatusPanel } from "./components/StatusPanel";
import { useTelemetry } from "./hooks/useTelemetry";

export default function App() {
  const { telemetry, connected } = useTelemetry();
  const [autonomousEnabled, setAutonomousEnabled] = useState(false);

  const simulator = telemetry?.simulator ?? null;
  const pipeline = telemetry?.pipeline ?? null;

  return (
    <div className="min-h-screen bg-slate-950 p-4 text-slate-100">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <h1 className="text-lg font-bold tracking-tight">UGV Autonomous Navigation — PS 26126</h1>
          <p className="text-xs text-slate-500">
            Vision-based navigation for an outdoor Unmanned Ground Vehicle · Bharat Electronics Limited · SIH 2026
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={`h-2 w-2 rounded-full ${connected ? "bg-green-400" : "bg-red-500"}`} />
          <span className="text-slate-400">{connected ? "backend connected" : "backend disconnected"}</span>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="space-y-4 lg:col-span-5">
          <CameraPanel autonomousEnabled={autonomousEnabled} />
          <DemoPanel onStarted={() => setAutonomousEnabled(true)} />
        </div>

        <div className="lg:col-span-4">
          <MapPanel simulator={simulator} pipeline={pipeline} autonomousEnabled={autonomousEnabled} />
        </div>

        <div className="space-y-4 lg:col-span-3">
          <StatusPanel simulator={simulator} pipeline={pipeline} />
          <ControlsPanel autonomousEnabled={autonomousEnabled} onAutonomousChange={setAutonomousEnabled} />
        </div>

        <div className="lg:col-span-12">
          <EventLogPanel />
        </div>
      </div>
    </div>
  );
}
