import type { SystemStateName } from "./types";

export const STATE_COLORS: Record<SystemStateName, string> = {
  SAFE: "#22c55e",
  WARNING: "#eab308",
  DEGRADED: "#f97316",
  CRITICAL: "#ef4444",
};

export const STATE_BG: Record<SystemStateName, string> = {
  SAFE: "bg-green-500/15 text-green-400 border-green-500/40",
  WARNING: "bg-yellow-500/15 text-yellow-400 border-yellow-500/40",
  DEGRADED: "bg-orange-500/15 text-orange-400 border-orange-500/40",
  CRITICAL: "bg-red-500/15 text-red-400 border-red-500/40",
};

export function confidenceColor(value: number): string {
  if (value >= 0.6) return "text-green-400";
  if (value >= 0.4) return "text-yellow-400";
  if (value >= 0.2) return "text-orange-400";
  return "text-red-400";
}
