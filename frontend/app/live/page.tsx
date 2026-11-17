"use client";

import { useEffect, useRef, useState } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAudioCapture } from "@/hooks/useAudioCapture";
import { ThreatGauge, ScoreCard, AlertPanel, RiskBadge } from "@/components/ui";
import type { RiskLevel } from "@/types";

/** Canvas-based real-time waveform visualizer */
function WaveformVisualizer({
  data,
  isActive,
}: {
  data: Float32Array | null;
  isActive: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, w, h);

    if (!data || !isActive) {
      // Flat line when idle
      ctx.strokeStyle = "#374151";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      return;
    }

    // Draw waveform
    const gradient = ctx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, "#06b6d4");
    gradient.addColorStop(0.5, "#3b82f6");
    gradient.addColorStop(1, "#06b6d4");
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.beginPath();

    const sliceWidth = w / data.length;
    let x = 0;

    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const y = (v + 1) / 2 * h;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    ctx.stroke();

    // Glow effect
    ctx.shadowBlur = 8;
    ctx.shadowColor = "#06b6d4";
    ctx.strokeStyle = "rgba(6, 182, 212, 0.3)";
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [data, isActive]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-20 rounded-lg"
      style={{ imageRendering: "auto" }}
    />
  );
}

export default function LiveProtectionPage() {
  const ws = useWebSocket();
  const audio = useAudioCapture();
  const [isActive, setIsActive] = useState(false);

  const handleStart = async () => {
    ws.start();
    await audio.startCapture((data) => {
      ws.sendAudio(data);
    });
    setIsActive(true);
  };

  const handleStop = () => {
    audio.stopCapture();
    ws.stop();
    setIsActive(false);
  };

  const currentLevel: RiskLevel = ws.lastResult?.risk_level || "LOW";
  const currentScore = ws.lastResult?.risk_score || 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Live Protection</h1>
          <p className="text-gray-500 mt-1">
            Real-time voice cloning detection via browser microphone
          </p>
        </div>
        <div className="flex items-center gap-3">
          {ws.isConnected && (
            <span className="flex items-center gap-1.5 text-xs text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Connected
            </span>
          )}
          {!isActive ? (
            <button
              onClick={handleStart}
              className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold transition-all shadow-lg shadow-cyan-500/20"
            >
              Start Protection
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="px-6 py-2.5 rounded-lg bg-red-600 hover:bg-red-500 text-white font-bold transition-colors"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {audio.error && (
        <div className="glass-card p-4 border-red-500/30 bg-red-500/5">
          <p className="text-red-400 text-sm">⚠ {audio.error}</p>
        </div>
      )}

      {/* Waveform Visualizer */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-gray-500 uppercase tracking-wider">
            Audio Input
          </span>
          {isActive && (
            <span className="flex items-center gap-1.5 text-xs text-cyan-400">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              Listening
            </span>
          )}
        </div>
        <WaveformVisualizer data={audio.waveformData} isActive={isActive} />
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Threat Gauge */}
        <div className="lg:col-span-1">
          <ThreatGauge score={currentScore} level={currentLevel} />

          {/* Microphone Status */}
          <div className="glass-card p-4 mt-4">
            <div className="flex items-center gap-3">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  audio.isRecording
                    ? "bg-cyan-500/20 animate-pulse-glow"
                    : "bg-gray-800"
                }`}
              >
                <svg
                  className={`w-5 h-5 ${
                    audio.isRecording ? "text-cyan-400" : "text-gray-500"
                  }`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                  />
                </svg>
              </div>
              <div>
                <div className="text-sm font-medium text-white">
                  {audio.isRecording ? "Listening..." : "Microphone Off"}
                </div>
                <div className="text-xs text-gray-500">
                  {isActive
                    ? `${ws.results.length} windows analyzed`
                    : "Press Start to begin"}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Scores & Info */}
        <div className="lg:col-span-2 space-y-4">
          {/* Score Cards */}
          <div className="grid grid-cols-2 gap-4">
            <ScoreCard
              label="Raw Spoof Score"
              value={ws.lastResult?.spoof.spoof_score || 0}
              color={
                ws.lastResult && ws.lastResult.spoof.spoof_score > 0.7
                  ? "bg-red-500"
                  : undefined
              }
            />
            <ScoreCard
              label="Speaker Match"
              value={ws.lastResult?.speaker?.speaker_similarity || 0}
              suffix={ws.lastResult?.speaker ? "" : "N/A"}
              color="bg-blue-500"
            />
          </div>

          {/* Confidence indicator */}
          {ws.results.length > 0 && ws.results.length < 3 && (
            <div className="glass-card p-3 border-amber-500/20 bg-amber-500/5">
              <p className="text-xs text-amber-400">
                ⏳ Low confidence — {ws.results.length}/3 windows analyzed.
                Scores may shift as more data arrives.
              </p>
            </div>
          )}

          {/* Current Decision */}
          {ws.lastResult && (
            <div className="glass-card p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-gray-500 uppercase tracking-wider">
                    Current Decision
                  </div>
                  <div className="text-lg font-bold text-white mt-1">
                    {ws.lastResult.decision}
                  </div>
                </div>
                <RiskBadge level={ws.lastResult.risk_level} />
              </div>
              <p className="text-sm text-gray-400 mt-2">
                {ws.lastResult.reason}
              </p>
              <div className="text-xs text-gray-600 mt-2">
                Window #{ws.lastResult.window_index} ·{" "}
                {new Date(ws.lastResult.timestamp).toLocaleTimeString()}
              </div>
            </div>
          )}

          {/* Active Alert */}
          {ws.lastAlert && (
            <AlertPanel
              title={ws.lastAlert.title}
              message={ws.lastAlert.message}
              level={ws.lastAlert.level}
              reason={ws.lastAlert.reason}
              onDismiss={ws.dismiss}
              onVerify={ws.requestChallenge}
              onBlock={ws.block}
            />
          )}

          {/* Challenge Panel */}
          {ws.challenge && (
            <div className="glass-card p-4 border-cyan-500/30 bg-cyan-500/5">
              <div className="text-xs text-cyan-400 uppercase tracking-wider mb-2">
                🎤 Verification Challenge
              </div>
              <p className="text-white font-medium text-lg">
                &ldquo;{ws.challenge.phrase}&rdquo;
              </p>
              <p className="text-sm text-gray-400 mt-2">
                Please repeat this phrase clearly into the microphone. The system
                will compare your live voice against the suspected synthetic
                audio.
              </p>
            </div>
          )}

          {/* Recent Windows */}
          <div className="glass-card p-4">
            <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">
              Analysis History
            </div>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {ws.results.length === 0 ? (
                <p className="text-gray-600 text-sm">Waiting for audio...</p>
              ) : (
                ws.results
                  .slice(-20)
                  .reverse()
                  .map((r, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between py-1.5 border-b border-gray-800/30 last:border-0"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-600 font-mono w-8">
                          #{r.window_index}
                        </span>
                        <span className="text-sm text-gray-300">
                          Spoof: {Math.round(r.spoof.spoof_score * 100)}%
                        </span>
                      </div>
                      <RiskBadge level={r.risk_level} />
                    </div>
                  ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
