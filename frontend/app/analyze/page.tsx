"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import type { AnalysisResult } from "@/types";
import { ThreatGauge, ScoreCard, RiskBadge, Timeline } from "@/components/ui";
import { AudioPlayer } from "@/components/AudioPlayer";

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [speakerId, setSpeakerId] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState(0);

  const steps = ["Upload", "VAD", "Spoof Detection", "Speaker Verify", "Risk Assessment", "Complete"];

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    // Simulate progress steps
    for (let i = 1; i <= 4; i++) {
      setStep(i);
      await new Promise((r) => setTimeout(r, 300));
    }

    try {
      const data = await api.analyzeUpload(file, speakerId || undefined);
      setResult(data);
      setStep(5);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      setStep(0);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Analyze Recording</h1>
        <p className="text-gray-500 mt-1">Upload an audio file to detect synthetic voice indicators</p>
      </div>

      {/* Upload Area */}
      {!result && (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className={`glass-card p-12 border-2 border-dashed text-center cursor-pointer transition-colors ${
            file ? "border-cyan-500/50 bg-cyan-500/5" : "border-gray-700 hover:border-gray-600"
          }`}
          onClick={() => document.getElementById("file-input")?.click()}
        >
          <input
            id="file-input"
            type="file"
            accept=".wav,.mp3,.m4a,.webm,.ogg,.flac"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          {file ? (
            <div>
              <div className="text-4xl mb-3">🎵</div>
              <p className="text-white font-medium">{file.name}</p>
              <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          ) : (
            <div>
              <div className="text-4xl mb-3 opacity-50">📁</div>
              <p className="text-gray-400">Drop an audio file here, or click to browse</p>
              <p className="text-xs text-gray-600 mt-2">WAV, MP3, M4A, WebM, OGG, FLAC (max 50MB)</p>
            </div>
          )}
        </div>
      )}

      {/* Speaker ID (optional) */}
      {!result && file && (
        <div className="glass-card p-4">
          <label className="text-xs text-gray-500 uppercase tracking-wider">Speaker ID (optional)</label>
          <input
            type="text"
            value={speakerId}
            onChange={(e) => setSpeakerId(e.target.value)}
            placeholder="Enter enrolled speaker ID for verification"
            className="w-full mt-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none"
          />
        </div>
      )}

      {/* Analyze Button */}
      {!result && file && (
        <button
          onClick={handleAnalyze}
          disabled={isAnalyzing}
          className="w-full py-3 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold transition-all"
        >
          {isAnalyzing ? "Analyzing..." : "Analyze Recording"}
        </button>
      )}

      {/* Progress */}
      {isAnalyzing && (
        <div className="glass-card p-4">
          <div className="flex items-center gap-2 mb-3">
            {steps.map((s, i) => (
              <div key={i} className="flex items-center gap-1">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                  i < step ? "bg-cyan-500 text-white" : i === step ? "bg-cyan-500/30 text-cyan-400 animate-pulse" : "bg-gray-800 text-gray-600"
                }`}>
                  {i < step ? "✓" : i + 1}
                </div>
                {i < steps.length - 1 && <div className={`w-8 h-0.5 ${i < step ? "bg-cyan-500" : "bg-gray-800"}`} />}
              </div>
            ))}
          </div>
          <p className="text-sm text-gray-400">{steps[step]}...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="glass-card p-4 border-red-500/30 bg-red-500/5">
          <p className="text-red-400 text-sm">⚠ {error}</p>
          <button onClick={() => { setError(null); setResult(null); setStep(0); }} className="text-xs text-gray-400 mt-2 underline">Try again</button>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <h2 className="text-xl font-bold text-white">Forensic Analysis Results</h2>
              <p className="text-xs text-gray-400 mt-0.5">
                Session: <span className="font-mono text-gray-300">{result.session_id.slice(0, 8)}</span>
                {file && <span className="ml-2 text-gray-500 font-mono truncate max-w-xs inline-block align-bottom">({file.name})</span>}
              </p>
            </div>
            <button
              onClick={() => { setResult(null); setFile(null); setStep(0); }}
              className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm text-gray-300 hover:text-white transition-colors border border-gray-700"
            >
              ← Analyze another recording
            </button>
          </div>

          {/* EXACT ANALYZED RECORDING PLAYBACK PLAYER */}
          {result.recording && (
            <AudioPlayer
              title="PLAY ANALYZED RECORDING"
              playbackUrl={result.recording.playback_url}
              playbackAvailable={result.recording.playback_available}
              originalFilename={result.recording.original_filename}
              mimeType={result.recording.mime_type}
              durationSeconds={result.recording.duration_seconds}
              sha256={result.recording.sha256}
              unavailableMessage="RECORDING NO LONGER AVAILABLE"
            />
          )}

          {/* Top Hero: Threat Gauge + Security Verdict */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-1 flex flex-col justify-stretch">
              <ThreatGauge score={result.risk.risk_score} level={result.risk.risk_level} />
            </div>

            <div className="md:col-span-2 glass-card p-6 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
                  <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">
                    Final Security Verdict
                  </span>
                  <RiskBadge level={result.risk.risk_level} />
                </div>
                <div className="p-4 rounded-lg bg-gray-900/70 border border-gray-800/80 mb-4">
                  <p className="text-base text-gray-100 font-medium leading-relaxed">
                    {result.verdict?.reason || result.risk.reason}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-gray-800/60 text-xs">
                <div>
                  <span className="text-gray-500 block">Duration</span>
                  <span className="text-gray-300 font-mono font-semibold">{result.duration_seconds}s</span>
                </div>
                <div>
                  <span className="text-gray-500 block">Speech Segments</span>
                  <span className="text-gray-300 font-mono font-semibold">{result.segments.length}</span>
                </div>
                <div>
                  <span className="text-gray-500 block">Suspicious Segments</span>
                  <span className={`font-mono font-semibold ${result.risk.suspicious_window_count > 0 ? "text-amber-400" : "text-emerald-400"}`}>
                    {result.risk.suspicious_window_count}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500 block">Latency</span>
                  <span className="text-gray-300 font-mono font-semibold">{result.processing_time_ms.toFixed(0)}ms</span>
                </div>
              </div>
            </div>
          </div>

          {/* Three Semantic Pillars Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Pillar 1: Voice Authenticity (Anti-Spoof Pipeline) */}
            <div className="glass-card p-5 space-y-3 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-2 border-b border-gray-800">
                  <span className="text-xs text-cyan-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
                    🛡️ Voice Authenticity
                  </span>
                  <span className={`text-[11px] font-mono px-2 py-0.5 rounded font-bold ${
                    result.risk.risk_level in { HIGH: 1, CRITICAL: 1 }
                      ? "bg-red-500/20 text-red-400 border border-red-500/40"
                      : result.risk.risk_level === "MEDIUM"
                      ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                      : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                  }`}>
                    {result.risk.risk_level in { HIGH: 1, CRITICAL: 1 }
                      ? "SYNTHETIC DETECTED"
                      : result.risk.risk_level === "MEDIUM"
                      ? "ANOMALOUS"
                      : "AUTHENTIC SPEECH"}
                  </span>
                </div>

                <div className="mt-4 space-y-3">
                  <div>
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span>Raw Logit Margin (Δ)</span>
                      <span className="font-mono text-white font-semibold">
                        {result.spoof?.raw_margin !== undefined && result.spoof?.raw_margin !== null
                          ? `${result.spoof.raw_margin > 0 ? "+" : ""}${result.spoof.raw_margin.toFixed(2)}`
                          : "N/A"}
                      </span>
                    </div>
                    <div className="text-[11px] text-gray-500">
                      Δ = logit(spoof) - logit(bonafide). Positive values indicate synthetic speech characteristics.
                    </div>
                  </div>

                  <div className="pt-2 border-t border-gray-800/40">
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span>Calibrated Probability</span>
                      <span className="font-mono text-white font-semibold">
                        {result.spoof?.calibrated_probability !== null && result.spoof?.calibrated_probability !== undefined
                          ? `${(result.spoof.calibrated_probability * 100).toFixed(1)}%`
                          : "UNAVAILABLE"}
                      </span>
                    </div>
                    <div className="text-[11px] text-gray-500">
                      {result.spoof?.status === "CALIBRATED"
                        ? "Empirically calibrated under WebRTC/Opus distribution."
                        : "Calibration unavailable; displaying raw logit evidence."}
                    </div>
                  </div>
                </div>
              </div>

              <div className="text-[11px] text-gray-500 pt-2 border-t border-gray-800">
                AASIST spoof detector evaluates synthetic audio artifacts independently from identity.
              </div>
            </div>

            {/* Pillar 2: Speaker Identification (Biometrics) */}
            <div className="glass-card p-5 space-y-3 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-2 border-b border-gray-800">
                  <span className="text-xs text-purple-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
                    👤 Speaker Match
                  </span>
                  <span className={`text-[11px] font-mono px-2 py-0.5 rounded font-bold ${
                    result.speaker?.best_match?.status === "MATCH"
                      ? "bg-purple-500/20 text-purple-300 border border-purple-500/40"
                      : "bg-gray-800 text-gray-400 border border-gray-700"
                  }`}>
                    {result.speaker?.best_match?.status || "NO MATCH"}
                  </span>
                </div>

                <div className="mt-4 space-y-3">
                  {result.speaker?.best_match ? (
                    <div>
                      <div className="text-base font-bold text-white break-words">
                        {result.speaker.best_match.speaker_name}
                      </div>
                      <div className="text-xs text-gray-500 font-mono truncate" title={result.speaker.best_match.speaker_id}>
                        ID: {result.speaker.best_match.speaker_id}
                      </div>

                      <div className="mt-3 p-2.5 rounded bg-gray-900/60 border border-gray-800/60">
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-gray-400">Measured Cosine Similarity</span>
                          <span className="font-mono text-purple-300 font-bold text-sm">
                            {result.speaker.best_match.similarity.toFixed(3)}
                          </span>
                        </div>
                        <div className="mt-1.5 h-1 bg-gray-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-purple-500 rounded-full"
                            style={{ width: `${Math.max(0, Math.min(100, Math.round(result.speaker.best_match.similarity * 100)))}%` }}
                          />
                        </div>
                      </div>

                      {result.speaker.alternatives && result.speaker.alternatives.length > 0 && (
                        <div className="mt-3">
                          <span className="text-[11px] text-gray-500 uppercase tracking-wider font-semibold block mb-1">
                            Ranked Alternatives
                          </span>
                          <div className="space-y-1">
                            {result.speaker.alternatives.slice(0, 2).map((alt) => (
                              <div key={alt.speaker_id} className="flex justify-between items-center text-xs text-gray-400">
                                <span className="truncate max-w-[140px]">{alt.speaker_name}</span>
                                <span className="font-mono text-gray-500">{alt.similarity.toFixed(2)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="py-6 text-center text-gray-500 text-xs">
                      No enrolled speaker profiles matched with confidence.
                    </div>
                  )}
                </div>
              </div>

              <div className="text-[11px] text-purple-400/80 bg-purple-950/20 p-2 rounded border border-purple-900/30">
                ⚠ Security Notice: Speaker resemblance does not prove genuine speech. Synthetic clones match enrolled profiles.
              </div>
            </div>

            {/* Pillar 3: Signal Quality & Calibration */}
            <div className="glass-card p-5 space-y-3 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-2 border-b border-gray-800">
                  <span className="text-xs text-emerald-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
                    📶 Signal Quality
                  </span>
                  <span className={`text-[11px] font-mono px-2 py-0.5 rounded font-bold ${
                    result.quality?.status === "VALID"
                      ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                      : "bg-red-500/20 text-red-400 border border-red-500/40"
                  }`}>
                    {result.quality?.status || "VALID"}
                  </span>
                </div>

                <div className="mt-4 space-y-3 text-xs">
                  <div className="flex justify-between py-1 border-b border-gray-800/40">
                    <span className="text-gray-400">Acoustic Energy (RMS)</span>
                    <span className="font-mono text-white font-semibold">
                      {result.quality?.rms !== undefined ? result.quality.rms.toFixed(4) : "Nominal"}
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-gray-800/40">
                    <span className="text-gray-400">Voice Activity Detection</span>
                    <span className="font-mono text-emerald-400 font-semibold">
                      {result.segments.length > 0 ? "Speech Detected" : "No Speech"}
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-gray-800/40">
                    <span className="text-gray-400">Calibration Status</span>
                    <span className="font-mono text-white font-semibold">
                      {result.spoof?.status || "CALIBRATION_UNAVAILABLE"}
                    </span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Audio Decoder</span>
                    <span className="font-mono text-cyan-400 font-semibold">PyAV (16 kHz PCM)</span>
                  </div>
                </div>
              </div>

              <div className="text-[11px] text-gray-500 pt-2 border-t border-gray-800">
                {result.quality?.reason || "Audio signal passes energy and temporal quality gates."}
              </div>
            </div>
          </div>

          {/* Timeline */}
          <Timeline
            segments={result.segments.map((s) => ({
              start: s.start_time,
              end: s.end_time,
              spoof_score: s.spoof.spoof_score,
            }))}
          />

          {/* Segment Details */}
          <div className="glass-card p-5">
            <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-3">
              Speech Segment Analysis ({result.segments.length} segments)
            </div>
            <div className="space-y-2">
              {result.segments.map((seg) => (
                <div
                  key={seg.segment_index}
                  className="flex items-center justify-between py-2.5 px-3 rounded-lg bg-gray-900/40 border border-gray-800/40 flex-wrap gap-2"
                >
                  <div className="flex items-center gap-4 flex-wrap">
                    <span className="text-xs text-cyan-400 font-mono">
                      {seg.start_time.toFixed(1)}s - {seg.end_time.toFixed(1)}s
                    </span>
                    <span className="text-xs text-gray-300">
                      Spoof Score: <span className="font-mono font-bold">{Math.round(seg.spoof.spoof_score * 100)}%</span>
                    </span>
                    {seg.spoof.delta !== undefined && seg.spoof.delta !== null && (
                      <span className="text-xs text-gray-400 font-mono">
                        Δ: {seg.spoof.delta > 0 ? "+" : ""}{seg.spoof.delta.toFixed(2)}
                      </span>
                    )}
                    {seg.speaker && (
                      <span className="text-xs text-purple-300">
                        Speaker Sim: <span className="font-mono font-bold">{seg.speaker.speaker_similarity.toFixed(2)}</span>
                      </span>
                    )}
                  </div>
                  <span
                    className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                      seg.spoof.label === "spoof"
                        ? "bg-red-500/20 text-red-400 border border-red-500/30"
                        : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                    }`}
                  >
                    {seg.spoof.label.toUpperCase()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

