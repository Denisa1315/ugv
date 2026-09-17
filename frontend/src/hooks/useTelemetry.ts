import { useEffect, useRef, useState } from "react";
import { BACKEND_WS } from "../api";
import type { TelemetryMessage } from "../types";

/** Live telemetry over the backend's /ws/telemetry stream. Reconnects
 * automatically on drop (e.g. backend restart) — no mock/fallback data is
 * ever substituted; the UI simply reflects "disconnected" until real data
 * resumes. */
export function useTelemetry() {
  const [telemetry, setTelemetry] = useState<TelemetryMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      if (cancelled) return;
      const ws = new WebSocket(`${BACKEND_WS}/ws/telemetry`);
      socketRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onmessage = (event) => {
        try {
          setTelemetry(JSON.parse(event.data));
        } catch {
          // ignore malformed frame
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) retryTimer = setTimeout(connect, 1000);
      };
      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      cancelled = true;
      clearTimeout(retryTimer);
      socketRef.current?.close();
    };
  }, []);

  return { telemetry, connected };
}
