"use client";

import { useState } from "react";
import {
  useMonitorSocket,
  type DetectionEvent,
  type TranscriptEntry,
  type TelemetryData,
} from "@/hooks/useMonitorSocket";

/* ────────────────────────────────────────────
   Helper: format sample offset → readable time
   ──────────────────────────────────────────── */
function samplesToTime(samples: number, sr = 16000): string {
  const secs = samples / sr;
  const m = Math.floor(secs / 60);
  const s = (secs % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

/* ────────────────────────────────────────────
   Panel 1 — Live Speech (VAD + Audio Telemetry)
   ──────────────────────────────────────────── */
function SpeechPanel({
  telemetry,
  webrtcState,
  browserDiagnostics,
  uiSpeechStatus,
  audioDevices,
  selectedDeviceId,
  setSelectedDeviceId,
}: {
  telemetry: ReturnType<typeof useMonitorSocket>["telemetry"];
  webrtcState: string;
  browserDiagnostics: ReturnType<typeof useMonitorSocket>["browserDiagnostics"];
  uiSpeechStatus: ReturnType<typeof useMonitorSocket>["uiSpeechStatus"];
  audioDevices: MediaDeviceInfo[];
  selectedDeviceId: string;
  setSelectedDeviceId: (id: string) => void;
}) {
  const webrtc = telemetry?.webrtc;
  const at = telemetry?.audio_truth;
  const stateColor =
    webrtcState === "connected"
      ? "text-emerald-400"
      : webrtcState === "connecting" || webrtcState === "checking"
        ? "text-amber-400"
        : "text-red-400";

  // Visual status badge
  let statusBadgeColor = "bg-gray-800 text-gray-400 border-gray-700";
  let statusText = "OFFLINE";
  let dotColor = "bg-gray-600";

  switch (uiSpeechStatus) {
    case "SPEECH_DETECTED":
      statusBadgeColor = "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
      statusText = "🎙 SPEECH ACTIVE (VAD ONSET)";
      dotColor = "bg-emerald-400 animate-pulse shadow-lg shadow-emerald-400/50";
      break;
    case "SILENCE":
      statusBadgeColor = "bg-blue-500/10 text-blue-300 border-blue-500/30";
      statusText = "— SILENCE (Verified VAD Silence)";
      dotColor = "bg-blue-400";
      break;
    case "AUDIO_RECEIVED":
      statusBadgeColor = "bg-cyan-500/10 text-cyan-300 border-cyan-500/30";
      statusText = "🔊 AUDIO RECEIVING (Awaiting VAD)";
      dotColor = "bg-cyan-400 animate-pulse";
      break;
    case "CONNECTING_WEBRTC":
      statusBadgeColor = "bg-amber-500/10 text-amber-300 border-amber-500/30";
      statusText = "🔄 CONNECTING WEBRTC...";
      dotColor = "bg-amber-400 animate-ping";
      break;
    case "NO_MIC_INPUT":
      statusBadgeColor = "bg-red-500/20 text-red-300 border-red-500/40";
      statusText = "⚠️ NO MIC INPUT / NEAR-ZERO (< 0.0001)";
      dotColor = "bg-red-400 animate-ping";
      break;
    case "VAD_UNAVAILABLE":
      statusBadgeColor = "bg-rose-500/20 text-rose-300 border-rose-500/40";
      statusText = "❌ VAD MODEL UNAVAILABLE";
      dotColor = "bg-rose-500";
      break;
  }

  // Pre-WebRTC browser mic level percentage
  const browserMicPct = Math.min(100, Math.max(0, (browserDiagnostics?.currentRms ?? 0) * 800));

  return (
    <div className="rounded-xl border border-gray-800/60 bg-gray-900/50 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Live Speech & Microphone Ingest
        </h3>
        <span className={`text-xs font-mono ${stateColor}`}>
          WebRTC: {webrtcState}
        </span>
      </div>

      {/* Unambiguous Resolved Speech Status */}
      <div className={`flex items-center justify-between p-3 rounded-lg border ${statusBadgeColor}`}>
        <div className="flex items-center gap-3">
          <div className={`w-3.5 h-3.5 rounded-full ${dotColor}`} />
          <span className="text-sm font-bold tracking-wide">{statusText}</span>
        </div>
        {browserDiagnostics && (
          <span className="text-[10px] font-mono opacity-80">
            {browserDiagnostics.speechActive ? "VOICE DETECTED" : "QUIET / NO VOICE"}
          </span>
        )}
      </div>

      {/* Near-zero input warning if mic is dead/muted */}
      {browserDiagnostics?.nearZeroInputWarning && (
        <div className="p-2.5 rounded-lg bg-red-950/40 border border-red-500/30 text-xs text-red-300 space-y-1">
          <p className="font-bold flex items-center gap-1.5">
            <span>⚠️</span> MICROPHONE INPUT UNAVAILABLE
          </p>
          <p className="text-[11px] text-red-400">
            Audio RMS is near-zero (&lt; 0.0001) for over 3s. Check your browser mic permissions, macOS Voice Isolation, or select a different physical input device below.
          </p>
        </div>
      )}

      {/* Phase 1: Browser-Side Diagnostic Meter (BEFORE WebRTC) */}
      <div className="p-3 rounded-lg bg-gray-950/70 border border-gray-800/80 space-y-2">
        <div className="flex items-center justify-between text-xs text-gray-400">
          <span className="font-mono text-[11px] text-cyan-400 font-semibold">
            BROWSER MIC LEVEL (Pre-WebRTC AudioContext)
          </span>
          <span className="font-mono text-[11px]">
            RMS: {(browserDiagnostics?.currentRms ?? 0).toFixed(5)} | Peak: {(browserDiagnostics?.peakRms ?? 0).toFixed(5)}
          </span>
        </div>
        <div className="w-full h-2.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-100 rounded-full ${
              (browserDiagnostics?.currentRms ?? 0) > 0.006
                ? "bg-gradient-to-r from-emerald-500 to-cyan-400"
                : (browserDiagnostics?.currentRms ?? 0) > 0.001
                  ? "bg-amber-400"
                  : "bg-gray-600"
            }`}
            style={{ width: `${browserMicPct}%` }}
          />
        </div>
      </div>

      {/* Phase 2: Audio Device Selector & Lineage */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        <div>
          <label className="block text-[10px] uppercase font-mono text-gray-500 mb-1">
            Input Audio Device
          </label>
          <select
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            disabled={webrtcState === "connected"}
            className="w-full p-1.5 rounded bg-gray-950 border border-gray-800 text-gray-300 text-xs font-mono focus:border-cyan-500 focus:outline-none disabled:opacity-60"
          >
            {audioDevices.length === 0 ? (
              <option value="">Default System Microphone</option>
            ) : (
              audioDevices.map((d) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label || `Microphone ${d.deviceId.substring(0, 8)}...`}
                </option>
              ))
            )}
          </select>
        </div>

        <div>
          <label className="block text-[10px] uppercase font-mono text-gray-500 mb-1">
            Track Lineage & Sender Proof
          </label>
          <div className="p-1.5 rounded bg-gray-950 border border-gray-800 text-[11px] font-mono text-gray-400 flex items-center justify-between">
            <span className="truncate max-w-[150px]">
              {browserDiagnostics?.trackLabel || "MacBook/System Mic"}
            </span>
            <span className={browserDiagnostics?.senderIdentityVerified ? "text-emerald-400 font-bold" : "text-amber-400"}>
              {browserDiagnostics?.senderIdentityVerified ? "✓ SENDER MATCHED" : "—"}
            </span>
          </div>
        </div>
      </div>

      {/* Backend Canonical Audio Telemetry */}
      {webrtc && webrtc.status === "receiving" && (
        <div className="pt-2 border-t border-gray-800/60 grid grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono text-gray-500">
          <span>Codec: {webrtc.codec || "UNKNOWN"}</span>
          <span>Source SR: {webrtc.source_sample_rate ? `${webrtc.source_sample_rate} Hz` : "NOT_REPORTED"}</span>
          <span>Canonical: 16 kHz mono float32</span>
          <span>Packets: {(webrtc.packets_received ?? 0).toLocaleString()}</span>
          <span>Canonical RMS: {(at?.rms ?? webrtc.rms ?? 0).toFixed(5)}</span>
          <span>Canonical Peak: {(at?.peak ?? webrtc.peak ?? 0).toFixed(5)}</span>
          <span>Zero%: {(webrtc.zero_pct ?? 0).toFixed(1)}%</span>
          <span>Duration: {(webrtc.total_duration_s ?? 0).toFixed(1)}s</span>
        </div>
      )}
    </div>
  );
}

/* ────────────────────────────────────────────
   Panel 2 — Transcript (ASR)
   ──────────────────────────────────────────── */
function TranscriptPanel({
  transcripts,
  telemetry,
}: {
  transcripts: TranscriptEntry[];
  telemetry?: TelemetryData | null;
}) {
  const asrLabel = telemetry?.models?.asr_model
    ? `${telemetry.models.asr_provider || "HF Whisper"} (${telemetry.models.asr_model})`
    : "HF Whisper";

  return (
    <div className="rounded-xl border border-gray-800/60 bg-gray-900/50 p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Transcript
        </h3>
        <span className="text-[10px] text-gray-500 font-mono">
          {asrLabel}
        </span>
      </div>

      <div className="space-y-1 max-h-56 overflow-y-auto scrollbar-thin">
        {transcripts.length === 0 ? (
          <p className="text-gray-600 text-sm italic">
            Waiting for speech...
          </p>
        ) : (
          transcripts.map((t, i) => (
            <div key={`${t.sample_start}-${i}`} className="flex gap-3">
              <span className="text-xs font-mono text-gray-600 shrink-0 mt-0.5">
                {samplesToTime(t.sample_start)}
              </span>
              <span
                className={`text-sm ${
                  t.is_partial
                    ? "text-gray-500 italic"
                    : "text-gray-200"
                }`}
              >
                {t.text}
              </span>
              {!t.is_partial && t.latency_ms > 0 && (
                <span className="text-xs text-gray-600 shrink-0 mt-0.5">
                  {t.latency_ms.toFixed(0)}ms
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

/* ────────────────────────────────────────────
   Panel 3 — Anti-Spoof Evidence (AASIST)
   ──────────────────────────────────────────── */
function DetectionPanel({
  detections,
  windowProgress,
}: {
  detections: DetectionEvent[];
  windowProgress: ReturnType<typeof useMonitorSocket>["windowProgress"];
}) {
  const latest = detections.length > 0 ? detections[detections.length - 1] : null;

  return (
    <div className="rounded-xl border border-gray-800/60 bg-gray-900/50 p-5">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
        Anti-Spoof Evidence
      </h3>

      {/* Window progress bar */}
      {windowProgress && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
            <span>
              {windowProgress.progress_pct < 100
                ? `Collecting window #${windowProgress.next_window_index}...`
                : "Window ready"}
            </span>
            <span>{windowProgress.progress_pct.toFixed(0)}%</span>
          </div>
          <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-600 to-blue-600 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(windowProgress.progress_pct, 100)}%` }}
            />
          </div>
          <p className="text-xs text-gray-600 mt-1 font-mono">
            {windowProgress.samples_collected.toLocaleString()} / {windowProgress.samples_needed.toLocaleString()} samples
          </p>
        </div>
      )}

      {/* AASIST results */}
      {latest ? (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className={`text-sm font-medium ${latest.window_valid ? "text-emerald-400" : "text-amber-400"}`}>
              {latest.window_valid ? "✓" : "⚠"} AASIST WINDOW #{latest.window_index}
            </span>
            {!latest.window_valid && (
              <span className="text-xs text-amber-500 bg-amber-500/10 px-2 py-0.5 rounded">
                {latest.invalidation_reason}
              </span>
            )}
          </div>

          {/* Raw model output & Calibrated Contract */}
          <div className="bg-gray-950/50 rounded-lg p-3 border border-gray-800/40 space-y-3">
            <div>
              <p className="text-xs text-gray-500 uppercase mb-1">AASIST Raw Output</p>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div>
                  <span className="text-gray-500">spoof logit: </span>
                  <span className="text-gray-300">{latest.raw_logit_spoof.toFixed(4)}</span>
                </div>
                <div>
                  <span className="text-gray-500">bonafide logit: </span>
                  <span className="text-gray-300">{latest.raw_logit_bonafide.toFixed(4)}</span>
                </div>
                <div>
                  <span className="text-gray-500">raw spoof score: </span>
                  <span className="text-gray-300">{latest.spoof_score.toFixed(4)}</span>
                </div>
                <div>
                  <span className="text-gray-500">raw bonafide: </span>
                  <span className="text-gray-300">{latest.bonafide_score.toFixed(4)}</span>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-gray-800/60 grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500 block text-[11px]">Calibrated Spoof Probability</span>
                <div className="flex items-baseline gap-1.5">
                  <span className="font-mono text-sm font-semibold text-gray-200">
                    {latest.calibrated_probability !== null && latest.calibrated_probability !== undefined
                      ? `${(latest.calibrated_probability * 100).toFixed(1)}%`
                      : "N/A"}
                  </span>
                  {latest.calibrated_probability !== null && (
                    <span className="text-[9px] text-amber-400/80 font-mono uppercase">
                      (Experimental)
                    </span>
                  )}
                </div>
                <span className="text-[10px] text-gray-500 block">
                  {latest.calibration_version !== "none" ? `[${latest.calibration_version}]` : "(Not Calibrated)"}
                </span>
              </div>
              <div>
                <span className="text-gray-500 block text-[11px]">Operating Decision</span>
                <span className={`font-mono text-sm font-bold ${
                  latest.decision === "SUSPICIOUS"
                    ? "text-red-400"
                    : latest.decision === "GENUINE"
                      ? "text-emerald-400"
                      : latest.decision === "UNCERTAIN"
                        ? "text-amber-400"
                        : "text-slate-300"
                }`}>
                  {latest.decision || "UNCALIBRATED"}
                </span>
                <span className="text-[10px] text-gray-500 block">
                  Status: {latest.calibration_status}
                </span>
              </div>
            </div>

            {/* Scientific Caveat / Evidence context */}
            <div className="pt-2 border-t border-gray-800/60 text-[11px] text-gray-400 space-y-1">
              <div className="flex justify-between items-center">
                <span>Evidence: <strong className="text-gray-200">{detections.filter((d) => d.window_valid).length} valid windows</strong></span>
                <span className="text-[10px] text-amber-400/90 font-mono">Calibration: Moderate (ECE 0.19)</span>
              </div>
              <p className="text-[10px] text-gray-500 italic">
                * Single-window probabilities are provisional estimates undergoing multi-condition testing, not definitive voice clone percentages.
              </p>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className={`px-2 py-0.5 rounded font-medium ${
              latest.calibration_status === "CALIBRATED"
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                : latest.calibration_status === "NOT_APPLICABLE_SILENCE"
                  ? "bg-slate-700/20 text-slate-400 border border-slate-700/30"
                  : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
            }`}>
              Calibration: {
                latest.calibration_status === "NOT_APPLICABLE_SILENCE"
                  ? "SILENCE (PAUSE)"
                  : latest.calibration_status
              }
            </span>
            <span className="text-gray-500 font-mono">{latest.latency_ms.toFixed(0)}ms</span>
          </div>

          {/* Timeline sample range */}
          <p className="text-xs text-gray-600 font-mono">
            samples {latest.sample_start.toLocaleString()} → {latest.sample_end.toLocaleString()}
            {" "}({samplesToTime(latest.sample_start)} – {samplesToTime(latest.sample_end)})
          </p>
        </div>
      ) : (
        <p className="text-gray-600 text-sm italic">
          Waiting for first AASIST window (~4s of audio)...
        </p>
      )}

      {/* Detection history (Rule 1: Neutral styling when uncalibrated) */}
      {detections.length > 1 && (
        <div className="mt-4 pt-4 border-t border-gray-800/40">
          <p className="text-xs text-gray-500 mb-2">
            History ({detections.length} windows) — {
              latest?.calibration_status === "CALIBRATED"
                ? "Calibrated Decisions"
                : "Raw Windows (Neutral - Uncalibrated)"
            }
          </p>
          <div className="flex gap-1.5 flex-wrap">
            {detections.map((d) => {
              let colorClass = "bg-slate-800 text-slate-300 border border-slate-700/60";
              if (d.decision === "ABSTAIN" || d.calibration_status === "NOT_APPLICABLE_SILENCE") {
                colorClass = "bg-slate-800/80 text-slate-400 border border-slate-700/40";
              } else if (!d.window_valid) {
                colorClass = "bg-amber-500/20 text-amber-400 border border-amber-500/30";
              } else if (d.decision === "SUSPICIOUS") {
                colorClass = "bg-red-500/20 text-red-400 border border-red-500/30";
              } else if (d.decision === "GENUINE") {
                colorClass = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
              } else if (d.decision === "UNCERTAIN") {
                colorClass = "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30";
              }

              return (
                <div
                  key={d.window_index}
                  className={`w-7 h-7 rounded-md text-xs flex items-center justify-center font-mono ${colorClass}`}
                  title={`Window #${d.window_index}: Raw spoof=${d.spoof_score.toFixed(3)}, Logits=[sp:${d.raw_logit_spoof.toFixed(2)}, bf:${d.raw_logit_bonafide.toFixed(2)}], Decision=${d.decision || "UNCALIBRATED"}`}
                >
                  {d.window_index}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/* ────────────────────────────────────────────
   Panel 4 — Live Speaker Identity (ECAPA-TDNN)
   ──────────────────────────────────────────── */
function SpeakerPanel({
  speakerEvents,
  telemetry,
}: {
  speakerEvents: ReturnType<typeof useMonitorSocket>["speakerEvents"];
  telemetry?: ReturnType<typeof useMonitorSocket>["telemetry"] | null;
}) {
  const latest = speakerEvents.length > 0 ? speakerEvents[speakerEvents.length - 1] : null;

  // Find last evaluated speaker match/candidate in this session (continuity state)
  const lastEvaluated = [...speakerEvents]
    .reverse()
    .find((e) => e.status === "MATCH" || e.status === "NO_CONFIDENT_MATCH") || null;

  // If latest is silence/insufficient evidence, but we have a prior valid match in the session,
  // enter UI continuity state instead of flickering
  const isContinuityState = Boolean(
    lastEvaluated &&
    latest &&
    (latest.status === "INSUFFICIENT_EVIDENCE" || latest.status === "UNAVAILABLE")
  );

  const displayEvent = isContinuityState ? lastEvaluated : latest;

  // Derive active status
  let status = displayEvent?.status;
  if (!status) {
    if (telemetry?.models?.ecapa_loaded === false) {
      status = "UNAVAILABLE";
    } else if (telemetry && (telemetry as any).enrolled_profiles_count === 0) {
      status = "NO_ENROLLED_PROFILES";
    } else {
      status = "INSUFFICIENT_EVIDENCE";
    }
  }

  // Format updated time
  let updatedDisplay = "—";
  if (displayEvent?.updated_at) {
    try {
      const d = new Date(displayEvent.updated_at);
      const m = d.getMinutes().toString().padStart(2, "0");
      const s = d.getSeconds().toString().padStart(2, "0");
      const ms = Math.floor(d.getMilliseconds() / 100);
      updatedDisplay = `${m}:${s}.${ms}`;
    } catch {
      updatedDisplay = displayEvent.updated_at;
    }
  } else if (displayEvent?.sample_end) {
    updatedDisplay = samplesToTime(displayEvent.sample_end);
  }

  // Resolve best match name and similarity
  const bestCandidateName = displayEvent?.best_match?.speaker_name || displayEvent?.speaker_name || null;
  const similarityVal = displayEvent?.best_match?.similarity ?? displayEvent?.speaker_similarity ?? null;
  const profilesChecked = displayEvent?.profiles_checked ?? (telemetry as any)?.enrolled_profiles_count ?? 0;
  const windowId = displayEvent?.speech_window_id
    ? (displayEvent.speech_window_id.startsWith("win_") ? `#${displayEvent.speech_window_id.slice(4)}` : displayEvent.speech_window_id)
    : "—";

  // Badge styling per state
  let badgeText = "INSUFFICIENT EVIDENCE";
  let badgeColor = "bg-slate-800 text-slate-400 border-slate-700";
  let dotColor = "bg-slate-500";

  switch (status) {
    case "MATCH":
      badgeText = isContinuityState ? "MATCH (CONTINUITY)" : "MATCH";
      badgeColor = "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
      dotColor = "bg-emerald-400";
      break;
    case "NO_CONFIDENT_MATCH":
      badgeText = isContinuityState ? "NO CONFIDENT MATCH (CONTINUITY)" : "NO CONFIDENT MATCH";
      badgeColor = "bg-amber-500/20 text-amber-300 border-amber-500/40";
      dotColor = "bg-amber-400";
      break;
    case "NO_ENROLLED_PROFILES":
      badgeText = "NO ENROLLED PROFILES";
      badgeColor = "bg-blue-500/10 text-blue-300 border-blue-500/30";
      dotColor = "bg-blue-400";
      break;
    case "INSUFFICIENT_EVIDENCE":
      badgeText = "INSUFFICIENT EVIDENCE";
      badgeColor = "bg-gray-800 text-gray-400 border-gray-700";
      dotColor = "bg-gray-500";
      break;
    case "UNAVAILABLE":
      badgeText = "SPEAKER IDENTIFICATION UNAVAILABLE";
      badgeColor = "bg-rose-500/20 text-rose-300 border-rose-500/40";
      dotColor = "bg-rose-500";
      break;
  }

  return (
    <div className="rounded-xl border border-gray-800/60 bg-gray-900/50 p-5 space-y-4">
      <div className="flex items-center justify-between pb-1 border-b border-gray-800/60">
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-purple-400" />
            LIVE SPEAKER IDENTITY
          </h3>
          <p className="text-[10px] text-gray-500 mt-0.5">
            Zero-Trust: Speaker resemblance does not prove voice authenticity.
          </p>
        </div>
        <span className={`px-2 py-0.5 rounded text-xs font-mono font-semibold border flex items-center gap-1.5 ${badgeColor}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
          {badgeText}
        </span>
      </div>

      {/* Main Identity Display */}
      {status === "MATCH" && (
        <div className="p-3.5 rounded-lg bg-gray-950/80 border border-emerald-500/30 space-y-3">
          <div className="flex items-baseline justify-between flex-wrap gap-2">
            <div>
              <span className="text-[10px] text-gray-500 uppercase tracking-wider block">Identified Enrolled Speaker</span>
              <span className="text-lg font-bold text-white tracking-wide">{bestCandidateName || "Enrolled Speaker"}</span>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-gray-500 uppercase tracking-wider block">Similarity</span>
              <span className="text-lg font-mono font-bold text-emerald-400">
                {similarityVal !== null ? similarityVal.toFixed(4) : "—"}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-gray-800/60 text-xs font-mono text-gray-400">
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">Profiles Checked</span>
              <span className="text-gray-200 font-semibold">{profilesChecked}</span>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">Speech Window</span>
              <span className="text-cyan-400">{windowId}</span>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">Updated</span>
              <span className="text-gray-200">{updatedDisplay}</span>
            </div>
          </div>

          {isContinuityState && (
            <div className="flex items-center justify-between px-2.5 py-1.5 rounded bg-gray-900/90 border border-gray-800 text-[11px] font-mono mt-2">
              <span className="text-cyan-400 flex items-center gap-1.5 font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                Monitoring...
              </span>
              <span className="text-gray-400">
                Last speech evidence: <strong className="text-gray-200">{updatedDisplay}</strong>
              </span>
            </div>
          )}
        </div>
      )}

      {status === "NO_CONFIDENT_MATCH" && (
        <div className="p-3.5 rounded-lg bg-gray-950/80 border border-amber-500/30 space-y-3">
          <div>
            <span className="text-xs font-semibold text-amber-300 block">No Confident Enrolled Match</span>
            <p className="text-[11px] text-gray-400 mt-0.5">
              Speaker does not exceed identity threshold with any enrolled profile.
            </p>
          </div>

          {bestCandidateName && (
            <div className="flex items-center justify-between p-2 rounded bg-gray-900 border border-gray-800 text-xs font-mono">
              <div>
                <span className="text-[10px] text-gray-500 block">Closest Candidate</span>
                <span className="text-gray-200 font-medium">{bestCandidateName}</span>
              </div>
              <div className="text-right">
                <span className="text-[10px] text-gray-500 block">Similarity</span>
                <span className="text-amber-400 font-semibold">{similarityVal !== null ? similarityVal.toFixed(4) : "—"}</span>
              </div>
            </div>
          )}

          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-gray-800/60 text-xs font-mono text-gray-400">
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">Profiles Checked</span>
              <span className="text-gray-200 font-semibold">{profilesChecked}</span>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">Speech Window</span>
              <span className="text-gray-300">{windowId}</span>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">Updated</span>
              <span className="text-gray-200">{updatedDisplay}</span>
            </div>
          </div>

          {isContinuityState && (
            <div className="flex items-center justify-between px-2.5 py-1.5 rounded bg-gray-900/90 border border-gray-800 text-[11px] font-mono mt-2">
              <span className="text-amber-400 flex items-center gap-1.5 font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                Monitoring...
              </span>
              <span className="text-gray-400">
                Last speech evidence: <strong className="text-gray-200">{updatedDisplay}</strong>
              </span>
            </div>
          )}
        </div>
      )}

      {status === "NO_ENROLLED_PROFILES" && (
        <div className="p-4 rounded-lg bg-gray-950/60 border border-gray-800 text-center space-y-1.5">
          <p className="text-sm font-medium text-gray-300">No Enrolled Speakers</p>
          <p className="text-xs text-gray-500">
            Zero speaker profiles registered. Enroll a voice in Speaker Enrollment to enable live identity recognition.
          </p>
          <a
            href="/enroll"
            className="inline-block mt-2 text-xs text-cyan-400 hover:text-cyan-300 underline font-mono"
          >
            → Go to Enrollment
          </a>
        </div>
      )}

      {status === "INSUFFICIENT_EVIDENCE" && (
        <div className="p-4 rounded-lg bg-gray-950/60 border border-gray-800 text-center space-y-1.5">
          <p className="text-sm font-medium text-gray-400">Insufficient Speech Evidence</p>
          <p className="text-xs text-gray-500 italic">
            Listening for continuous vocal speech (requires ≥ 1.0s clear speech above noise floor).
          </p>
          {latest?.reason && (
            <div className="inline-block mt-1 px-2.5 py-0.5 rounded bg-gray-900 border border-gray-800 text-[10px] font-mono text-gray-400">
              Trigger reason: <span className="text-cyan-400 font-semibold">{latest.reason}</span>
            </div>
          )}
        </div>
      )}

      {status === "UNAVAILABLE" && (
        <div className="p-4 rounded-lg bg-red-950/30 border border-red-500/20 text-center space-y-1">
          <p className="text-sm font-medium text-red-300">Speaker Identification Unavailable</p>
          <p className="text-xs text-red-400/80">
            ECAPA-TDNN model is offline or encountered an isolated inference error. Call audio and anti-spoofing continue uninterrupted.
          </p>
        </div>
      )}

      {/* Alternative Candidates mini-list if multiple profiles exist */}
      {latest?.candidates && latest.candidates.length > 1 && (
        <div className="pt-2 border-t border-gray-800/40 text-[11px]">
          <span className="text-gray-500 block mb-1">Ranked Candidate Profiles ({latest.candidates.length}):</span>
          <div className="space-y-1">
            {latest.candidates.slice(0, 3).map((c, idx) => (
              <div key={idx} className="flex justify-between items-center text-xs font-mono px-2 py-1 rounded bg-gray-950/40 border border-gray-800/50">
                <span className="text-gray-300">{c.speaker_name}</span>
                <span className="text-gray-400">
                  Similarity: <strong className={c.status === "MATCH" ? "text-emerald-400" : "text-gray-300"}>{c.similarity.toFixed(4)}</strong>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ────────────────────────────────────────────
   Main Monitor Page
   ──────────────────────────────────────────── */
export default function MonitorPage() {
  const {
    sessionId,
    isConnected,
    webrtcState,
    error,
    transcripts,
    detections,
    speakerEvents,
    vadActive,
    windowProgress,
    telemetry,
    browserDiagnostics,
    audioDevices,
    selectedDeviceId,
    setSelectedDeviceId,
    uiSpeechStatus,
    isStarting,
    startMonitor,
    stopMonitor,
  } = useMonitorSocket();

  const isActive = sessionId !== null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Real-Time Monitor
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            WebRTC Audio → Caption + Anti-Spoof pipelines
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Connection status */}
          <div className="flex items-center gap-2 text-xs">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected
                  ? "bg-emerald-400"
                  : isStarting
                    ? "bg-amber-400 animate-ping"
                    : "bg-gray-600"
              }`}
            />
            <span className="text-gray-500">
              {isConnected
                ? "Connected"
                : isStarting
                  ? "Connecting..."
                  : "Disconnected"}
            </span>
          </div>

          {/* Start/Stop */}
          {!isActive ? (
            <button
              onClick={startMonitor}
              disabled={isStarting}
              className={`px-4 py-2 rounded-lg text-white text-sm font-medium transition-all ${
                isStarting
                  ? "bg-gray-700 cursor-not-allowed opacity-75 animate-pulse"
                  : "bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 cursor-pointer shadow-lg shadow-cyan-900/30"
              }`}
            >
              {isStarting ? "Connecting WebRTC..." : "Start Monitor"}
            </button>
          ) : (
            <button
              onClick={stopMonitor}
              className="px-4 py-2 rounded-lg bg-gray-800 text-red-400 text-sm font-medium border border-red-500/20 hover:bg-red-500/10 cursor-pointer transition-all"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Session info */}
      {sessionId && (
        <p className="text-xs text-gray-600 font-mono">
          Session: {sessionId}
        </p>
      )}

      {/* 4 Signal Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <SpeechPanel
          telemetry={telemetry}
          webrtcState={webrtcState}
          browserDiagnostics={browserDiagnostics}
          uiSpeechStatus={uiSpeechStatus}
          audioDevices={audioDevices}
          selectedDeviceId={selectedDeviceId}
          setSelectedDeviceId={setSelectedDeviceId}
        />
        <TranscriptPanel transcripts={transcripts} telemetry={telemetry} />
        <DetectionPanel
          detections={detections}
          windowProgress={windowProgress}
        />
        <SpeakerPanel speakerEvents={speakerEvents} telemetry={telemetry} />
      </div>

      {/* LIVE AUDIO TRUTH & RUNTIME MODEL IDENTITY (Phase 2, 7, 8, 18) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Live Audio Truth Panel */}
        <div className="p-4 rounded-xl bg-gray-950 border border-cyan-500/20 font-mono text-xs text-gray-300">
          <div className="flex items-center justify-between pb-2 mb-3 border-b border-gray-800">
            <h4 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              LIVE AUDIO TRUTH
            </h4>
            <span className="text-[10px] text-gray-500">Zero-Trust Audio Lineage & Signal Chain</span>
          </div>

          {/* Phase 7: End-to-End Signal Flow Probe Chain */}
          <div className="mb-3 p-2 rounded bg-gray-900/60 border border-gray-800 text-[11px] grid grid-cols-4 gap-2 text-center">
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">1. Mic Captured</span>
              <span className="text-cyan-300 font-bold font-mono">
                {(browserDiagnostics?.samplesCaptured ?? 0).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">2. RTP Packets</span>
              <span className="text-cyan-300 font-bold font-mono">
                {(telemetry?.webrtc?.packets_received ?? 0).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">3. Decoded Frames</span>
              <span className="text-cyan-300 font-bold font-mono">
                {(telemetry?.webrtc?.frames_decoded ?? 0).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 block uppercase">4. Canonical (16k)</span>
              <span className="text-emerald-400 font-bold font-mono">
                {(telemetry?.webrtc?.total_samples ?? 0).toLocaleString()}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
            <div>
              <span className="text-gray-500">Session: </span>
              <span className="text-gray-200">{telemetry?.audio_truth?.session_id ? `${telemetry.audio_truth.session_id.substring(0, 12)}...` : (sessionId ? `${sessionId.substring(0, 12)}...` : "—")}</span>
            </div>
            <div>
              <span className="text-gray-500">Track: </span>
              <span className="text-cyan-300 font-bold">{telemetry?.audio_truth?.track_id || "—"}</span>
            </div>
            <div>
              <span className="text-gray-500">Peer: </span>
              <span className="text-gray-200">{telemetry?.audio_truth?.observed_peer_id || "local"}</span>
            </div>
            <div>
              <span className="text-gray-500">Direction: </span>
              <span className="text-gray-200">{telemetry?.audio_truth?.track_direction || "outbound"}</span>
            </div>
            <div className="col-span-2 pt-1 border-t border-gray-800/60">
              <span className="text-gray-500">Source: </span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30">
                {telemetry?.audio_truth?.track_source || "LOCAL_MIC"}
              </span>
            </div>

            <div className="pt-2">
              <span className="text-gray-500">Codec / Format: </span>
              <span className="text-gray-200">{telemetry?.webrtc?.codec || "UNKNOWN"} ({telemetry?.webrtc?.source_sample_rate ? `${telemetry.webrtc.source_sample_rate}Hz` : "NOT_REPORTED"})</span>
            </div>
            <div className="pt-2">
              <span className="text-gray-500">Audio duration: </span>
              <span className="text-gray-200">{(telemetry?.webrtc?.total_duration_s ?? 0).toFixed(2)}s</span>
            </div>

            {/* Phase 8: Raw Signal Statistics */}
            <div className="col-span-2 pt-2 border-t border-gray-800/60 grid grid-cols-2 gap-x-4 gap-y-1.5">
              <div>
                <span className="text-gray-500">Canonical RMS: </span>
                <span className={`font-bold ${
                  (telemetry?.webrtc?.current_rms ?? telemetry?.webrtc?.rms ?? 0) < 0.003 ? "text-amber-400" : "text-emerald-400"
                }`}>
                  {(telemetry?.webrtc?.current_rms ?? telemetry?.webrtc?.rms ?? 0).toFixed(5)}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Peak: </span>
                <span className="text-gray-200">{(telemetry?.webrtc?.current_peak ?? telemetry?.webrtc?.peak ?? 0).toFixed(5)}</span>
              </div>
              <div>
                <span className="text-gray-500">VAD probability: </span>
                <span className="text-gray-200">{(telemetry?.webrtc?.vad_probability ?? 0).toFixed(3)}</span>
              </div>
              <div>
                <span className="text-gray-500">Speech active: </span>
                <span className={vadActive ? "text-emerald-400 font-bold" : "text-gray-500"}>
                  {vadActive ? "TRUE (VAD ONSET)" : "FALSE (SILENCE)"}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Zero Crossing: </span>
                <span className="text-gray-300">{(telemetry?.webrtc?.zero_pct ?? 0).toFixed(1)}% zeros</span>
              </div>
              <div>
                <span className="text-gray-500">Resampler: </span>
                <span className="text-gray-300">{telemetry?.webrtc?.resampler_active ? "48k → 16k Active" : "16k Direct"}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Runtime Model Identity Panel */}
        <div className="p-4 rounded-xl bg-gray-950 border border-gray-800 font-mono text-xs text-gray-300">
          <div className="flex items-center justify-between pb-2 mb-3 border-b border-gray-800">
            <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              RUNTIME MODEL IDENTITY
            </h4>
            <span className="text-[10px] text-gray-500">Verified Process Models</span>
          </div>

          <div className="space-y-2 text-[11px]">
            <div className="flex justify-between items-center py-1 border-b border-gray-900">
              <span className="text-gray-500">ASR Provider:</span>
              <span className="text-gray-200 font-semibold">{telemetry?.models?.asr_provider || "UNAVAILABLE"}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-gray-900">
              <span className="text-gray-500">ASR Model:</span>
              <span className="text-gray-200 font-semibold">{telemetry?.models?.asr_model || "UNAVAILABLE"}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-gray-900">
              <span className="text-gray-500">ASR Backend:</span>
              <span className="text-gray-200 font-semibold">{telemetry?.models?.asr_backend || "UNAVAILABLE"}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-gray-900">
              <span className="text-gray-500">AASIST Model:</span>
              <span className="text-gray-200 font-semibold">{telemetry?.models?.aasist_model || "UNAVAILABLE"}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-gray-900">
              <span className="text-gray-500">AASIST Checkpoint:</span>
              <span className="text-emerald-400 font-semibold">{telemetry?.models?.aasist_loaded ? "Loaded (weights_aasist.pth)" : (telemetry?.models ? "Not loaded" : "UNAVAILABLE")}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-gray-900">
              <span className="text-gray-500">ECAPA Model:</span>
              <span className="text-gray-200 font-semibold">{telemetry?.models?.ecapa_model || "UNAVAILABLE"}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-gray-900">
              <span className="text-gray-500">VAD Model:</span>
              <span className="text-gray-200 font-semibold">{telemetry?.models?.vad_model || "UNAVAILABLE"}</span>
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="text-gray-500">Calibration:</span>
              <span className={`font-semibold ${
                telemetry?.models?.calibration_status?.startsWith("CALIBRATED") ? "text-emerald-400" : "text-amber-400"
              }`}>
                {telemetry?.models?.calibration_status || "UNCALIBRATED"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom telemetry bar */}
      {telemetry?.audio_bus && (
        <div className="rounded-xl border border-gray-800/40 bg-gray-900/30 p-4">
          <div className="flex items-center justify-between text-xs font-mono text-gray-500">
            <span>
              Audio: {telemetry.audio_bus.total_duration_s}s processed
            </span>
            <span>
              Chunks published: {telemetry.audio_bus.total_published.toLocaleString()}
            </span>
            {telemetry.audio_bus.consumers &&
              Object.entries(telemetry.audio_bus.consumers).map(
                ([name, c]) => (
                  <span key={name}>
                    {name}: depth={c.queue_depth} dropped={c.dropped}
                  </span>
                )
              )}
          </div>
        </div>
      )}
    </div>
  );
}
