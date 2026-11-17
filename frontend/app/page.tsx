"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { SystemHealth, AlertPayload } from "@/types";
import { ModelStatusIndicator, RiskBadge } from "@/components/ui";

export default function DashboardPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [alerts, setAlerts] = useState<AlertPayload[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await api.health();
        setHealth(data);
        setError(null);
      } catch {
        setError("Cannot connect to backend");
      }
    };

    const fetchAlerts = async () => {
      try {
        const data = await api.getRecentAlerts(10);
        setAlerts(data.alerts || []);
      } catch {
        /* ignore — alerts are optional */
      }
    };

    fetchHealth();
    fetchAlerts();
    const interval = setInterval(() => {
      fetchHealth();
      fetchAlerts();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          Security Dashboard
        </h1>
        <p className="text-gray-500 mt-1">AI-Powered Voice Cloning Detection & Prevention</p>
      </div>

      {error && (
        <div className="glass-card p-4 border-red-500/30 bg-red-500/5">
          <p className="text-red-400 text-sm">
            ⚠ {error}. Make sure the backend is running:{" "}
            <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded">cd backend && source venv/bin/activate && uvicorn app.main:app --port 8000</code>
          </p>
        </div>
      )}

      {/* Quick Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <a href="/live" className="glass-card p-6 hover:border-cyan-500/30 transition-colors group cursor-pointer">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center group-hover:bg-cyan-500/20 transition-colors">
              <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <h3 className="font-bold text-white">Live Protection</h3>
          </div>
          <p className="text-sm text-gray-400">Monitor your call in real-time for voice cloning attacks.</p>
        </a>

        <a href="/analyze" className="glass-card p-6 hover:border-blue-500/30 transition-colors group cursor-pointer">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
              <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <h3 className="font-bold text-white">Analyze Recording</h3>
          </div>
          <p className="text-sm text-gray-400">Upload and analyze a recording for synthetic voice indicators.</p>
        </a>

        <a href="/enroll" className="glass-card p-6 hover:border-purple-500/30 transition-colors group cursor-pointer">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center group-hover:bg-purple-500/20 transition-colors">
              <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <h3 className="font-bold text-white">Enroll Speaker</h3>
          </div>
          <p className="text-sm text-gray-400">Create a voice profile for speaker verification.</p>
        </a>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Status */}
        <div className="glass-card p-6">
          <h2 className="text-lg font-bold text-white mb-4">AI Model Status</h2>
          {health ? (
            <div className="divide-y divide-gray-800">
              {health.models.map((m) => (
                <div key={m.name} className="py-2">
                  <ModelStatusIndicator name={m.name} status={m.status} message={m.message} />
                  {m.load_time_ms && (
                    <div className="text-xs text-gray-600 ml-0 mt-0.5">
                      Loaded in {Math.round(m.load_time_ms)}ms
                    </div>
                  )}
                </div>
              ))}
              <div className="flex items-center justify-between py-2">
                <span className="text-sm text-gray-300">Session Store</span>
                <span className={`text-xs font-mono ${health.database === "IN_MEMORY" ? "text-amber-400" : health.database === "ONLINE" ? "text-emerald-400" : "text-gray-500"}`}>
                  {health.database}
                </span>
              </div>
              <div className="flex items-center justify-between py-2">
                <span className="text-sm text-gray-300">Redis Cache</span>
                <span className={`text-xs font-mono ${health.redis === "ONLINE" ? "text-emerald-400" : "text-gray-500"}`}>
                  {health.redis}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-gray-500 text-sm">Loading...</div>
          )}
          {health && (
            <div className="mt-4 pt-3 border-t border-gray-800">
              <span className="text-xs text-gray-500">
                System: <span className={health.status === "healthy" ? "text-emerald-400" : "text-amber-400"}>{health.status}</span>
                {" · "}Uptime: {Math.round(health.uptime_seconds)}s
              </span>
            </div>
          )}
        </div>

        {/* Recent Alerts */}
        <div className="glass-card p-6">
          <h2 className="text-lg font-bold text-white mb-4">Recent Alerts</h2>
          {alerts.length > 0 ? (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {alerts.map((alert) => (
                <div key={alert.alert_id} className="flex items-start gap-2 py-2 border-b border-gray-800/50 last:border-0">
                  <RiskBadge level={alert.level} />
                  <div className="min-w-0">
                    <p className="text-sm text-gray-300 truncate">{alert.title}</p>
                    <p className="text-xs text-gray-500">{new Date(alert.timestamp).toLocaleString()}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-gray-500 text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              No recent alerts. System is clear.
            </div>
          )}
        </div>
      </div>

      {/* System Info */}
      <div className="glass-card p-4">
        <div className="flex flex-wrap gap-6 text-xs text-gray-500">
          <span>PS-104: AI-Powered Voice Cloning Detection</span>
          <span>Models: Spectra-AASIST · ECAPA-TDNN · Silero VAD</span>
          <span>SIH 2026 Prototype</span>
        </div>
      </div>
    </div>
  );
}
