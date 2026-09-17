import { useEffect, useState } from "react";
import { api } from "../api";
import { Panel } from "./Panel";

/** Polls the backend's latest annotated perception frame (real camera or
 * demo synthetic feed — whichever mode is active) as a live-refreshing
 * image. Never a canned/mock image. */
export function CameraPanel({ autonomousEnabled }: { autonomousEnabled: boolean }) {
  const [src, setSrc] = useState<string | null>(null);
  const [everLoaded, setEverLoaded] = useState(false);

  useEffect(() => {
    if (!autonomousEnabled) {
      setSrc(null);
      setEverLoaded(false);
      return;
    }
    const interval = setInterval(() => {
      setSrc(api.cameraFrameUrl());
    }, 200);
    return () => clearInterval(interval);
  }, [autonomousEnabled]);

  return (
    <Panel title="Live Camera / Perception View">
      <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded bg-black">
        {!autonomousEnabled && (
          <span className="text-xs text-slate-500">Enable autonomous mode to start the feed</span>
        )}
        {autonomousEnabled && src && (
          // Always mounted (never unmounted on a failed attempt) so each
          // new polled URL gets a fresh chance to load — the very first
          // frame can 409 before the pipeline has processed anything yet.
          <img
            src={src}
            alt="Annotated camera feed"
            className="h-full w-full object-contain"
            style={{ visibility: everLoaded ? "visible" : "hidden" }}
            onLoad={() => setEverLoaded(true)}
          />
        )}
        {autonomousEnabled && !everLoaded && (
          <span className="absolute text-xs text-slate-500">Waiting for first frame…</span>
        )}
      </div>
    </Panel>
  );
}
