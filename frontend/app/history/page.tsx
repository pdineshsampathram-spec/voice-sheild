"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { RiskBadge } from "@/components/ui";

interface SecurityEvent {
  id: string;
  session_id: string;
  timestamp: string;
  spoof_score: number;
  speaker_similarity?: number;
  risk_score: number;
  risk_level: string;
  decision: string;
  reason: string;
  source: string;
}

export default function HistoryPage() {
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getSessions(100)
      .then((data) => setEvents(data.sessions || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Security History</h1>
        <p className="text-gray-500 mt-1">Past analysis sessions and security events</p>
      </div>

      <div className="glass-card overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-gray-500">Loading...</div>
        ) : events.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            <div className="text-4xl mb-3 opacity-30">🛡️</div>
            <p>No security events recorded yet.</p>
            <p className="text-xs mt-1">Events will appear here after analyzing audio.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-left">
                  <th className="px-4 py-3 text-xs text-gray-500 uppercase tracking-wider">Time</th>
                  <th className="px-4 py-3 text-xs text-gray-500 uppercase tracking-wider">Source</th>
                  <th className="px-4 py-3 text-xs text-gray-500 uppercase tracking-wider">Spoof</th>
                  <th className="px-4 py-3 text-xs text-gray-500 uppercase tracking-wider">Speaker</th>
                  <th className="px-4 py-3 text-xs text-gray-500 uppercase tracking-wider">Risk</th>
                  <th className="px-4 py-3 text-xs text-gray-500 uppercase tracking-wider">Decision</th>
                  <th className="px-4 py-3 text-xs text-gray-500 uppercase tracking-wider">Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/50">
                {events.map((evt) => (
                  <tr key={evt.id} className="hover:bg-gray-800/30 transition-colors">
                    <td className="px-4 py-3 text-gray-400 whitespace-nowrap">
                      {new Date(evt.timestamp).toLocaleString()}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`text-xs px-2 py-0.5 rounded ${evt.source === "live" ? "bg-cyan-500/10 text-cyan-400" : "bg-blue-500/10 text-blue-400"}`}>
                        {evt.source}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-300 font-mono">
                      {Math.round(evt.spoof_score * 100)}%
                    </td>
                    <td className="px-4 py-3 text-gray-300 font-mono">
                      {evt.speaker_similarity != null ? `${Math.round(evt.speaker_similarity * 100)}%` : "—"}
                    </td>
                    <td className="px-4 py-3">
                      <RiskBadge level={evt.risk_level as any} />
                    </td>
                    <td className="px-4 py-3 text-gray-400 text-xs">{evt.decision}</td>
                    <td className="px-4 py-3 text-gray-500 text-xs max-w-xs truncate">{evt.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
