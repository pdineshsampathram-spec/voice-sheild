"use client";

import type { RiskLevel } from "@/types";

const CONFIG: Record<RiskLevel, { color: string; bg: string; border: string; glow: string; label: string }> = {
  LOW: { color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30", glow: "glow-green", label: "SAFE" },
  MEDIUM: { color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/30", glow: "glow-amber", label: "MONITORING" },
  HIGH: { color: "text-orange-400", bg: "bg-orange-500/10", border: "border-orange-500/30", glow: "glow-amber", label: "HIGH RISK" },
  CRITICAL: { color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/30", glow: "glow-red", label: "CRITICAL" },
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  const cfg = CONFIG[level];
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${cfg.bg} ${cfg.color} border ${cfg.border}`}>
      <span className={`w-2 h-2 rounded-full ${level === "CRITICAL" ? "animate-pulse-glow" : ""} ${cfg.color.replace("text-", "bg-")}`} />
      {cfg.label}
    </span>
  );
}

export function ThreatGauge({ score, level }: { score: number; level: RiskLevel }) {
  const cfg = CONFIG[level];
  const clampedScore = Math.max(0, Math.min(1, isNaN(score) ? 0 : score));
  const pct = Math.round(clampedScore * 100);
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - clampedScore);

  return (
    <div className={`glass-card p-6 ${cfg.glow} flex flex-col items-center justify-center text-center`}>
      <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-3">
        Overall Risk
      </div>

      {/* Bounded circular container: inset-0 is strictly contained within 160x160 box */}
      <div className="relative w-[160px] h-[160px] flex items-center justify-center flex-shrink-0">
        <svg width="160" height="160" viewBox="0 0 160 160" className="transform -rotate-90 block">
          <circle cx="80" cy="80" r={radius} fill="none" stroke="#1f2937" strokeWidth="10" />
          <circle
            cx="80" cy="80" r={radius} fill="none"
            stroke={level === "CRITICAL" ? "#ef4444" : level === "HIGH" ? "#f97316" : level === "MEDIUM" ? "#f59e0b" : "#22c55e"}
            strokeWidth="10" strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            className="transition-all duration-700 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none select-none">
          <span className={`text-4xl font-extrabold tracking-tight tabular-nums ${cfg.color}`}>
            {pct}%
          </span>
          <span className="text-[11px] text-gray-400 font-medium mt-0.5">Risk Score</span>
        </div>
      </div>

      {/* Risk Badge separated cleanly in regular flex layout */}
      <div className="mt-4 flex flex-col items-center gap-1">
        <RiskBadge level={level} />
      </div>
    </div>
  );
}

export function ScoreCard({
  label,
  value,
  suffix,
  color,
  sublabel,
}: {
  label: string;
  value: number | null | undefined;
  suffix?: string;
  color?: string;
  sublabel?: string;
}) {
  const hasValue = value !== null && value !== undefined && !isNaN(value);
  const pct = hasValue ? Math.round(value * 100) : null;
  const barColor = color || (pct !== null && pct > 70 ? "bg-red-500" : pct !== null && pct > 40 ? "bg-amber-500" : "bg-emerald-500");

  return (
    <div className="glass-card p-4 flex flex-col justify-between min-h-[105px]">
      <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold truncate" title={label}>
        {label}
      </div>
      <div className="my-1.5 flex items-baseline gap-2 flex-wrap">
        <span className="text-2xl font-bold text-white tabular-nums">
          {hasValue ? `${pct}%` : "N/A"}
        </span>
        {suffix && <span className="text-xs text-gray-400 font-medium truncate">{suffix}</span>}
      </div>
      {hasValue ? (
        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden w-full">
          <div
            className={`h-full rounded-full ${barColor} animate-fill transition-all duration-500`}
            style={{ width: `${Math.max(0, Math.min(100, pct || 0))}%` }}
          />
        </div>
      ) : (
        <div className="text-[11px] text-gray-500 italic truncate">
          {sublabel || "Evidence unavailable"}
        </div>
      )}
    </div>
  );
}

export function ModelStatusIndicator({ name, status, message }: { name: string; status: string; message?: string }) {
  const isOnline = status === "ONLINE";
  const isError = status === "ERROR";

  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-gray-300">{name}</span>
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${isOnline ? "bg-emerald-400" : isError ? "bg-red-400 animate-pulse" : "bg-gray-500"}`} />
        <span className={`text-xs font-mono ${isOnline ? "text-emerald-400" : isError ? "text-red-400" : "text-gray-500"}`}>
          {status}
        </span>
      </div>
    </div>
  );
}

export function AlertPanel({ title, message, level, reason, onDismiss, onVerify, onBlock }: {
  title: string;
  message: string;
  level: RiskLevel;
  reason: string;
  onDismiss?: () => void;
  onVerify?: () => void;
  onBlock?: () => void;
}) {
  const cfg = CONFIG[level];
  const isCritical = level === "CRITICAL" || level === "HIGH";

  return (
    <div className={`${cfg.bg} border ${cfg.border} rounded-xl p-4 ${isCritical ? cfg.glow : ""}`}>
      <div className="flex items-start gap-3">
        <div className={`text-2xl ${cfg.color}`}>
          {level === "CRITICAL" ? "🔴" : level === "HIGH" ? "🔶" : "⚠️"}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className={`font-bold ${cfg.color}`}>{title}</h3>
          <p className="text-sm text-gray-300 mt-1">{message}</p>
          <p className="text-xs text-gray-500 mt-1">{reason}</p>
          {isCritical && (
            <div className="flex gap-2 mt-3">
              {onVerify && (
                <button onClick={onVerify} className="px-3 py-1.5 text-xs font-bold rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white transition-colors">
                  VERIFY
                </button>
              )}
              {onDismiss && (
                <button onClick={onDismiss} className="px-3 py-1.5 text-xs font-bold rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors">
                  DISMISS
                </button>
              )}
              {onBlock && (
                <button onClick={onBlock} className="px-3 py-1.5 text-xs font-bold rounded-lg bg-red-600 hover:bg-red-500 text-white transition-colors">
                  END / BLOCK
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function Timeline({ segments }: { segments: { start: number; end: number; spoof_score: number }[] }) {
  if (!segments.length) return null;
  const maxEnd = Math.max(...segments.map((s) => s.end));

  return (
    <div className="glass-card p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">Audio Timeline</div>
      <div className="relative h-8 bg-gray-800 rounded-lg overflow-hidden">
        {segments.map((seg, i) => {
          const left = (seg.start / maxEnd) * 100;
          const width = ((seg.end - seg.start) / maxEnd) * 100;
          const color = seg.spoof_score > 0.7 ? "bg-red-500/70" : seg.spoof_score > 0.4 ? "bg-amber-500/60" : "bg-emerald-500/50";

          return (
            <div
              key={i}
              className={`absolute top-0 h-full ${color} border-r border-gray-700/50`}
              style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
              title={`Segment ${i + 1}: spoof ${Math.round(seg.spoof_score * 100)}%`}
            />
          );
        })}
      </div>
      <div className="flex justify-between mt-1">
        <span className="text-xs text-gray-600">0s</span>
        <span className="text-xs text-gray-600">{maxEnd.toFixed(1)}s</span>
      </div>
    </div>
  );
}
