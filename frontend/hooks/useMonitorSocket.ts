"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

// --- Types ---

export interface CaptionEvent {
  type: "caption";
  text: string;
  is_partial: boolean;
  is_final?: boolean;
  utterance_id?: number;
  sample_start: number;
  sample_end: number;
  latency_ms: number;
  analysis_lag_ms?: number;
  model_name: string;
  asr_provider?: string;
  asr_model?: string;
  asr_backend?: string;
  vad_speech_active?: boolean;
}

export interface DetectionEvent {
  type: "detection";
  window_index: number;
  sample_start: number;
  sample_end: number;
  latency_ms: number;
  raw_logit_spoof: number;
  raw_logit_bonafide: number;
  spoof_score: number;
  bonafide_score: number;
  model: string;
  calibrated_probability: number | null;
  decision: string;
  calibration_status: string;
  calibration_version: string;
  calibration_method: string;
  operating_threshold: number | null;
  target_domain?: string;
  window_valid: boolean;
  invalidation_reason: string;
}

export type LiveSpeakerStatus =
  | "MATCH"
  | "NO_CONFIDENT_MATCH"
  | "NO_ENROLLED_PROFILES"
  | "INSUFFICIENT_EVIDENCE"
  | "UNAVAILABLE";

export interface LiveSpeakerCandidate {
  speaker_id: string;
  speaker_name: string;
  similarity: number;
  status: string;
}

export interface SpeakerEvent {
  type: "speaker" | "speaker_identity";
  sample_start?: number;
  sample_end?: number;
  embedding_dim?: number;
  speaker_similarity?: number | null;
  enrolled_speaker_id?: string | null;
  speaker_name?: string | null;
  match?: boolean | null;
  calibration?: string;
  latency_ms?: number;
  status?: LiveSpeakerStatus;
  best_match?: {
    speaker_id: string;
    speaker_name: string;
    similarity: number;
  } | null;
  candidates?: LiveSpeakerCandidate[];
  profiles_checked?: number;
  speech_window_id?: string;
  updated_at?: string;
  reason?: string | null;
}

export interface VADEvent {
  type: "vad";
  vad_speech_active: boolean;
  sample_start: number;
  sample_end: number;
}

export interface WindowProgressEvent {
  type: "window_progress";
  samples_collected: number;
  samples_needed: number;
  progress_pct: number;
  next_window_index: number;
}

export interface ProtectionDecisionEvent {
  type: "protection_decision";
  session_id: string;
  track_id: string;
  decision: string;
  risk_level: string;
  reason: string;
  evidence_ids: string[];
  timestamp: number;
  quality_status: string;
  calibration_status: string;
  window_count: number;
  suspicious_count: number;
  spoof_score_avg: number;
}

export interface TelemetryData {
  session_id?: string;
  uptime_s?: number;
  audio_truth?: {
    session_id?: string;
    room_id?: string;
    observed_peer_id?: string;
    track_id?: string;
    track_source?: string;
    track_direction?: string;
    connection_generation?: number;
    frames_received?: number;
    packets_received?: number;
    canonical_samples?: number;
    audio_duration?: number;
    current_timestamp?: number;
    rms?: number;
    peak?: number;
    min_val?: number;
    max_val?: number;
    mean_val?: number;
    zero_crossing_rate?: number;
    vad_probability?: number;
    speech_active?: boolean;
    status?: string;
    codec?: string;
    sample_rate?: number;
    channels?: number;
    resampler_active?: boolean;
  };
  models?: {
    asr_provider?: string;
    asr_model?: string;
    asr_backend?: string;
    asr_loaded?: boolean;
    aasist_model?: string;
    aasist_loaded?: boolean;
    ecapa_model?: string;
    ecapa_loaded?: boolean;
    vad_model?: string;
    calibration_status?: string;
    calibration_version?: string;
    calibration_domain?: string;
  };
  audio_bus?: {
    total_published: number;
    total_samples: number;
    total_duration_s: number;
    consumers: Record<string, { queue_depth: number; total_received: number; dropped: number }>;
  };
  webrtc?: {
    status?: string;
    codec?: string;
    source_sample_rate?: number;
    canonical_sample_rate?: number;
    packets_received?: number;
    frames_received?: number;
    samples_received?: number;
    frames_decoded?: number;
    total_samples?: number;
    total_duration_s?: number;
    rms?: number;
    current_rms?: number;
    peak?: number;
    current_peak?: number;
    vad_probability?: number;
    speech_active?: boolean;
    zero_pct?: number;
    resampler_active?: boolean;
    last_sample_timestamp?: number;
  };
}

export type MonitorEvent = CaptionEvent | DetectionEvent | SpeakerEvent | VADEvent | WindowProgressEvent;

// --- Browser-side microphone forensics and diagnostics (Phase 1-4, 7) ---

export interface BrowserMicDiagnostics {
  deviceId: string;
  deviceLabel: string;
  trackLabel: string;
  channelCount: number;
  sampleRate: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
  trackId: string;
  trackKind: string;
  trackReadyState: string;
  trackEnabled: boolean;
  trackMuted: boolean;
  senderTrackId: string;
  senderTrackKind: string;
  senderTrackReadyState: string;
  senderTrackEnabled: boolean;
  senderIdentityVerified: boolean;
  currentRms: number;
  peakRms: number;
  speechActive: boolean;
  samplesCaptured: number;
  nearZeroInputWarning: boolean;
}

export type UISpeechStatus =
  | "NO_MIC_INPUT"
  | "CONNECTING_WEBRTC"
  | "AUDIO_RECEIVED"
  | "SPEECH_DETECTED"
  | "SILENCE"
  | "VAD_UNAVAILABLE";

// --- Transcript entry ---

export interface TranscriptEntry {
  text: string;
  is_partial: boolean;
  sample_start: number;
  sample_end: number;
  latency_ms: number;
  timestamp: number;
  utterance_id?: number;
}

// --- Hook ---

interface UseMonitorSocketReturn {
  // Connection
  sessionId: string | null;
  isConnected: boolean;
  isStarting: boolean;
  webrtcState: string;
  error: string | null;

  // Pipeline state
  transcripts: TranscriptEntry[];
  detections: DetectionEvent[];
  speakerEvents: SpeakerEvent[];
  vadActive: boolean;
  windowProgress: WindowProgressEvent | null;
  protectionDecision: ProtectionDecisionEvent | null;
  telemetry: TelemetryData | null;

  // Browser Microphone Forensics (Phase 1-4)
  browserDiagnostics: BrowserMicDiagnostics | null;
  audioDevices: MediaDeviceInfo[];
  selectedDeviceId: string;
  setSelectedDeviceId: (id: string) => void;
  refreshAudioDevices: () => Promise<void>;
  uiSpeechStatus: UISpeechStatus;

  // Actions
  startMonitor: () => Promise<void>;
  stopMonitor: () => void;
}

export function useMonitorSocket(): UseMonitorSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const meterIntervalRef = useRef<any>(null);
  const isStartingRef = useRef(false);
  const sessionIdRef = useRef<string | null>(null);
  const samplesCapturedRef = useRef<number>(0);
  const silentStartTimestampRef = useRef<number | null>(null);
  const latestFinalUtteranceIdRef = useRef<number>(0);
  const latestSpeechWindowIndexRef = useRef<number>(0);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [webrtcState, setWebrtcState] = useState("new");
  const [error, setError] = useState<string | null>(null);

  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [detections, setDetections] = useState<DetectionEvent[]>([]);
  const [speakerEvents, setSpeakerEvents] = useState<SpeakerEvent[]>([]);
  const [vadActive, setVadActive] = useState(false);
  const [windowProgress, setWindowProgress] = useState<WindowProgressEvent | null>(null);
  const [protectionDecision, setProtectionDecision] = useState<ProtectionDecisionEvent | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);

  // Browser-side forensics and device selection
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [browserDiagnostics, setBrowserDiagnostics] = useState<BrowserMicDiagnostics | null>(null);

  // Enumerate input devices
  const refreshAudioDevices = useCallback(async () => {
    try {
      if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
        return;
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter((d) => d.kind === "audioinput");
      setAudioDevices(audioInputs);
      if (!selectedDeviceId && audioInputs.length > 0) {
        setSelectedDeviceId(audioInputs[0].deviceId);
      }
    } catch (err) {
      console.warn("[VoiceShield] Could not enumerate audio devices:", err);
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    refreshAudioDevices();
    if (typeof navigator !== "undefined" && navigator.mediaDevices) {
      navigator.mediaDevices.ondevicechange = refreshAudioDevices;
      return () => {
        navigator.mediaDevices.ondevicechange = null;
      };
    }
  }, [refreshAudioDevices]);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      if (data.session_id && sessionIdRef.current && data.session_id !== sessionIdRef.current) {
        return; // Reject stale event from previous session
      }

      switch (data.type) {
        case "connected":
          setTelemetry(data.telemetry || null);
          break;

        case "caption":
          if (data.text) {
            setTranscripts((prev) => {
              const utteranceId = typeof data.utterance_id === "number" ? data.utterance_id : 0;
              const isPartial = Boolean(data.is_partial);

              // 1. Drop stale partials belonging to utterances already finalized
              if (isPartial && utteranceId > 0 && utteranceId <= latestFinalUtteranceIdRef.current) {
                return prev;
              }

              // 2. Reject duplicate final entries for the same utterance
              if (!isPartial && utteranceId > 0 && prev.some((p) => !p.is_partial && p.utterance_id === utteranceId)) {
                return prev;
              }

              if (!isPartial && utteranceId > 0) {
                latestFinalUtteranceIdRef.current = Math.max(latestFinalUtteranceIdRef.current, utteranceId);
              }

              const entry: TranscriptEntry = {
                text: data.text,
                is_partial: isPartial,
                sample_start: data.sample_start,
                sample_end: data.sample_end,
                latency_ms: data.latency_ms || 0,
                timestamp: Date.now(),
                utterance_id: utteranceId,
              };

              // If last entry was a partial:
              if (prev.length > 0 && prev[prev.length - 1].is_partial) {
                const last = prev[prev.length - 1];
                // If it belongs to the same utterance (or replacing in-flight partial with final)
                if (!last.utterance_id || !utteranceId || last.utterance_id === utteranceId) {
                  return [...prev.slice(0, -1), entry];
                }
              }

              return [...prev.slice(-100), entry];
            });
          }
          break;

        case "detection":
          setDetections((prev) => [...prev.slice(-50), data as DetectionEvent]);
          break;

        case "protection_decision":
          setProtectionDecision(data as ProtectionDecisionEvent);
          break;

        case "speaker":
        case "speaker_identity": {
          const spk = data as SpeakerEvent;
          // Stale-event protection (Part 25): Monotonic speech_window_id check
          if (spk.speech_window_id) {
            const match = spk.speech_window_id.match(/\d+/);
            if (match) {
              const winIdx = parseInt(match[0], 10);
              if (winIdx < latestSpeechWindowIndexRef.current) {
                return; // Stale event discarded
              }
              latestSpeechWindowIndexRef.current = Math.max(latestSpeechWindowIndexRef.current, winIdx);
            }
          }
          setSpeakerEvents((prev) => [...prev.slice(-20), spk]);
          break;
        }

        case "vad":
          setVadActive(data.vad_speech_active || false);
          break;

        case "window_progress":
          setWindowProgress(data as WindowProgressEvent);
          break;

        case "telemetry":
          setTelemetry(data as TelemetryData);
          break;
      }
    } catch {
      // Non-JSON message
    }
  }, []);

  const stopMonitor = useCallback(() => {
    isStartingRef.current = false;
    setIsStarting(false);

    if (meterIntervalRef.current) {
      clearInterval(meterIntervalRef.current);
      meterIntervalRef.current = null;
    }

    if (audioCtxRef.current) {
      try {
        audioCtxRef.current.close();
      } catch {}
      audioCtxRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Close WebRTC
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }

    // Stop media stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    // Cleanup session on server
    if (sessionIdRef.current) {
      fetch(`${API_URL}/api/monitor/session/${sessionIdRef.current}`, {
        method: "DELETE",
      }).catch(() => {});
    }

    setIsConnected(false);
    setWebrtcState("closed");
    setSessionId(null);
    sessionIdRef.current = null;
    setBrowserDiagnostics(null);
    samplesCapturedRef.current = 0;
    silentStartTimestampRef.current = null;
    latestFinalUtteranceIdRef.current = 0;
    latestSpeechWindowIndexRef.current = 0;
  }, []);

  const startMonitor = useCallback(async () => {
    if (isStartingRef.current || isConnected || wsRef.current || pcRef.current) {
      return;
    }
    isStartingRef.current = true;
    setIsStarting(true);
    setError(null);
    setTranscripts([]);
    setDetections([]);
    setSpeakerEvents([]);
    setWindowProgress(null);
    setProtectionDecision(null);
    samplesCapturedRef.current = 0;
    silentStartTimestampRef.current = null;
    latestFinalUtteranceIdRef.current = 0;

    try {
      // 1. Get microphone access with verified constraints & selected deviceId
      const audioConstraints: MediaTrackConstraints = {
        channelCount: 1,
        sampleRate: 48000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      };
      if (selectedDeviceId) {
        audioConstraints.deviceId = { exact: selectedDeviceId };
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: audioConstraints,
        video: false,
      });
      streamRef.current = stream;
      const audioTrack = stream.getAudioTracks()[0];

      if (!audioTrack) {
        throw new Error("No audio track returned by navigator.mediaDevices.getUserMedia");
      }

      // Re-enumerate to get device labels now that permission is granted
      refreshAudioDevices();

      const settings = audioTrack.getSettings();

      // 2. Setup browser-side diagnostic meter (AudioContext + AnalyserNode) BEFORE WebRTC
      let analyser: AnalyserNode | null = null;
      try {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        if (AudioCtx) {
          const ctx = new AudioCtx();
          audioCtxRef.current = ctx;
          const sourceNode = ctx.createMediaStreamSource(stream);
          analyser = ctx.createAnalyser();
          analyser.fftSize = 512;
          sourceNode.connect(analyser);
        }
      } catch (err) {
        console.warn("[VoiceShield] AudioContext meter initialization failed:", err);
      }

      // 3. Create monitor session with verified track identity
      const sessionRes = await fetch(`${API_URL}/api/monitor/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asr_model_size: "base",
          track_id: audioTrack.id,
          track_source: "LOCAL_MIC",
          track_direction: "outbound",
        }),
      });

      if (!sessionRes.ok) {
        throw new Error(`Session creation failed: ${sessionRes.status}`);
      }

      const { session_id } = await sessionRes.json();
      sessionIdRef.current = session_id;
      setSessionId(session_id);

      // 4. Create WebRTC peer connection
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });
      pcRef.current = pc;

      pc.oniceconnectionstatechange = () => {
        setWebrtcState(pc.iceConnectionState);
      };

      pc.onconnectionstatechange = () => {
        setWebrtcState(pc.connectionState);
      };

      // Add verified audio track to WebRTC sender
      pc.addTrack(audioTrack, stream);

      // Phase 4: Prove WebRTC sender actually uses the mic track
      const senders = pc.getSenders();
      const audioSender = senders.find((s) => s.track && s.track.kind === "audio");
      const senderTrack = audioSender?.track;
      const senderVerified = senderTrack ? senderTrack.id === audioTrack.id : false;

      // Start periodic browser-side diagnostic metering
      if (analyser) {
        const timeData = new Float32Array(analyser.fftSize);
        let peakRms = 0;
        meterIntervalRef.current = setInterval(() => {
          if (!analyser || !audioTrack) return;
          analyser.getFloatTimeDomainData(timeData);
          let sumSq = 0;
          for (let i = 0; i < timeData.length; i++) {
            sumSq += timeData[i] * timeData[i];
          }
          const rms = Math.sqrt(sumSq / timeData.length);
          if (rms > peakRms) peakRms = rms;
          samplesCapturedRef.current += timeData.length;

          // Detect near-zero mic input (< 0.0001 for > 3.0 seconds)
          const now = Date.now();
          if (rms < 0.0001) {
            if (silentStartTimestampRef.current === null) {
              silentStartTimestampRef.current = now;
            }
          } else {
            silentStartTimestampRef.current = null;
          }
          const nearZeroWarning =
            silentStartTimestampRef.current !== null &&
            now - silentStartTimestampRef.current > 3000;

          setBrowserDiagnostics({
            deviceId: settings.deviceId || audioTrack.getSettings().deviceId || selectedDeviceId || "default",
            deviceLabel: audioTrack.label || "Default Microphone",
            trackLabel: audioTrack.label || "Default Microphone",
            channelCount: settings.channelCount || 1,
            sampleRate: settings.sampleRate || 48000,
            echoCancellation: Boolean(settings.echoCancellation ?? true),
            noiseSuppression: Boolean(settings.noiseSuppression ?? true),
            autoGainControl: Boolean(settings.autoGainControl ?? true),
            trackId: audioTrack.id,
            trackKind: audioTrack.kind,
            trackReadyState: audioTrack.readyState,
            trackEnabled: audioTrack.enabled,
            trackMuted: audioTrack.muted,
            senderTrackId: senderTrack?.id || "",
            senderTrackKind: senderTrack?.kind || "",
            senderTrackReadyState: senderTrack?.readyState || "",
            senderTrackEnabled: Boolean(senderTrack?.enabled),
            senderIdentityVerified: senderVerified,
            currentRms: rms,
            peakRms,
            speechActive: rms > 0.006,
            samplesCaptured: samplesCapturedRef.current,
            nearZeroInputWarning: nearZeroWarning,
          });
        }, 100);
      }

      // Create offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE gathering
      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          pc.onicegatheringstatechange = () => {
            if (pc.iceGatheringState === "complete") resolve();
          };
          setTimeout(resolve, 3000);
        }
      });

      // 5. Send offer to backend
      const offerRes = await fetch(`${API_URL}/api/monitor/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id,
          sdp: pc.localDescription!.sdp,
          type: "offer",
        }),
      });

      if (!offerRes.ok) {
        throw new Error(`SDP exchange failed: ${offerRes.status}`);
      }

      const { sdp: answerSdp } = await offerRes.json();
      await pc.setRemoteDescription(new RTCSessionDescription({
        type: "answer",
        sdp: answerSdp,
      }));

      // 6. Connect WebSocket for events
      const ws = new WebSocket(`${WS_URL}/api/monitor/ws/${session_id}`);
      ws.onmessage = handleMessage;
      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => setIsConnected(false);
      ws.onerror = () => setError("WebSocket connection error");
      wsRef.current = ws;

    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      console.error("Monitor start failed:", err);
      stopMonitor();
    } finally {
      isStartingRef.current = false;
      setIsStarting(false);
    }
  }, [isConnected, selectedDeviceId, refreshAudioDevices, handleMessage, stopMonitor]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopMonitor();
    };
  }, [stopMonitor]);

  // Compute resolved UI Speech Status without false "SILENCE" on uncaptured mics
  let uiSpeechStatus: UISpeechStatus = "NO_MIC_INPUT";
  if (!isConnected && !isStarting) {
    uiSpeechStatus = "NO_MIC_INPUT";
  } else if (isStarting || (webrtcState !== "connected" && webrtcState !== "completed")) {
    uiSpeechStatus = "CONNECTING_WEBRTC";
  } else if (browserDiagnostics?.nearZeroInputWarning) {
    uiSpeechStatus = "NO_MIC_INPUT";
  } else if (vadActive) {
    uiSpeechStatus = "SPEECH_DETECTED";
  } else if (telemetry?.audio_truth?.frames_received && telemetry.audio_truth.frames_received > 0) {
    uiSpeechStatus = "SILENCE";
  } else {
    uiSpeechStatus = "AUDIO_RECEIVED";
  }

  return {
    sessionId,
    isConnected,
    isStarting,
    webrtcState,
    error,
    transcripts,
    detections,
    speakerEvents,
    vadActive,
    windowProgress,
    protectionDecision,
    telemetry,
    browserDiagnostics,
    audioDevices,
    selectedDeviceId,
    setSelectedDeviceId,
    refreshAudioDevices,
    uiSpeechStatus,
    startMonitor,
    stopMonitor,
  };
}
