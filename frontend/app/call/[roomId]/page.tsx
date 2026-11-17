"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const getApiUrl = () => {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== "undefined") return "";
  return "http://localhost:8000";
};

const getWsUrl = () => {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}`;
  }
  return "ws://localhost:8000";
};

interface CaptionItem {
  text: string;
  is_partial: boolean;
  sample_start: number;
  sample_end: number;
  latency_ms: number;
}

interface DetectionItem {
  window_index: number;
  sample_start: number;
  sample_end: number;
  raw_logit_spoof: number;
  raw_logit_bonafide: number;
  spoof_score: number;
  bonafide_score: number;
  calibrated_probability: number | null;
  decision: string;
  calibration_status: string;
  calibration_version: string;
  calibration_method: string;
  operating_threshold: number | null;
  window_valid: boolean;
  invalidation_reason: string;
  latency_ms: number;
}

interface SpeakerItem {
  embedding_dim: number;
  speaker_similarity: number | null;
  calibration: string;
  latency_ms: number;
}

export interface AudioTruthData {
  session_id: string;
  room_id: string;
  observed_peer_id: string;
  track_id: string;
  track_source: "LOCAL_MIC" | "REMOTE_PEER";
  track_direction: string;
  connection_generation: number;
  frames_received?: number;
  canonical_samples?: number;
  audio_duration?: number;
  current_timestamp?: number;
  rms?: number;
  peak?: number;
  vad_probability?: number;
  speech_active?: boolean;
}

export interface ProtectionDecisionData {
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

export interface ModelIdentityData {
  asr_provider: string;
  asr_model: string;
  asr_backend: string;
  asr_loaded: boolean;
  aasist_model: string;
  aasist_loaded: boolean;
  ecapa_model: string;
  ecapa_loaded: boolean;
  vad_model: string;
  calibration_status?: string;
  calibration_version?: string;
}

/**
 * Sequential operation queue to serialize WebRTC signaling operations
 * and prevent concurrent SDP / ICE state corruptions.
 */
class SignalingQueue {
  private chain: Promise<void> = Promise.resolve();

  enqueue<T>(task: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.chain = this.chain.then(async () => {
        try {
          const result = await task();
          resolve(result);
        } catch (err) {
          reject(err);
        }
      });
    });
  }
}

export default function CallRoomPage({
  params,
}: {
  params: Promise<{ roomId: string }>;
}) {
  const { roomId } = use(params);
  const router = useRouter();

  // P2P WebRTC Call State
  const [callState, setCallState] = useState<"connecting" | "waiting" | "connected" | "ended">("connecting");
  const [peerCount, setPeerCount] = useState(1);
  const peerCountRef = useRef(1);
  const [remoteTrackAvailable, setRemoteTrackAvailable] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [callDuration, setCallDuration] = useState(0);

  // Monitor Observer Tap State
  const [monitorActive, setMonitorActive] = useState(true);
  const [monitorTarget, setMonitorTarget] = useState<"remote" | "local">("remote");
  const [monitorConnected, setMonitorConnected] = useState(false);
  const [monitorError, setMonitorError] = useState<string | null>(null);

  // Live Audio Truth & Model Identity State (Developer Diagnostics)
  const [audioTruth, setAudioTruth] = useState<AudioTruthData | null>(null);
  const [protectionDecision, setProtectionDecision] = useState<ProtectionDecisionData | null>(null);
  const [modelIdentity, setModelIdentity] = useState<ModelIdentityData | null>(null);
  const [partialCaption, setPartialCaption] = useState<string>("");
  const [latestValidWindow, setLatestValidWindow] = useState<{ start: number; end: number } | null>(null);
  const [latestAsrSegment, setLatestAsrSegment] = useState<{ start: number; end: number } | null>(null);

  // ML Evidence State
  const [vadActive, setVadActive] = useState(false);
  const [captions, setCaptions] = useState<CaptionItem[]>([]);
  const [detections, setDetections] = useState<DetectionItem[]>([]);
  const [speakerInfo, setSpeakerInfo] = useState<SpeakerItem | null>(null);
  const [windowProgress, setWindowProgress] = useState<number>(0);

  // Refs for P2P WebRTC
  const localStreamRef = useRef<MediaStream | null>(null);
  const remoteAudioRef = useRef<HTMLAudioElement | null>(null);
  const remoteTrackRef = useRef<MediaStreamTrack | null>(null);
  const p2pPcRef = useRef<RTCPeerConnection | null>(null);
  const signalingWsRef = useRef<WebSocket | null>(null);

  // Perfect Negotiation State Refs
  const isPoliteRef = useRef<boolean>(false);
  const makingOfferRef = useRef<boolean>(false);
  const ignoreOfferRef = useRef<boolean>(false);
  const connectionGenerationRef = useRef<number>(1);
  const peerIdRef = useRef<string>(`peer_${Math.random().toString(36).substring(2, 9)}`);
  const latestFinalUtteranceRef = useRef<number>(0);

  // Refs for Observer Monitor
  const monitorPcRef = useRef<RTCPeerConnection | null>(null);
  const monitorWsRef = useRef<WebSocket | null>(null);
  const monitorSessionIdRef = useRef<string | null>(null);
  const lastEventSeqRef = useRef<number>(-1);

  // ────────────────────────────────────────────────────────────
  // 1. Call Duration Timer
  // ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (callState !== "connected") return;
    const interval = setInterval(() => setCallDuration((d) => d + 1), 1000);
    return () => clearInterval(interval);
  }, [callState]);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // ────────────────────────────────────────────────────────────
  // 2. Observer Monitor Tap Lifecycle (Isolated from Call)
  // ────────────────────────────────────────────────────────────
  const stopObserver = useCallback(() => {
    lastEventSeqRef.current = -1;
    if (monitorWsRef.current) {
      monitorWsRef.current.close();
      monitorWsRef.current = null;
    }
    if (monitorPcRef.current) {
      monitorPcRef.current.close();
      monitorPcRef.current = null;
    }
    if (monitorSessionIdRef.current) {
      const apiBase = getApiUrl();
      fetch(`${apiBase}/api/monitor/session/${monitorSessionIdRef.current}`, { method: "DELETE" }).catch(() => {});
      monitorSessionIdRef.current = null;
    }
    setMonitorConnected(false);
    setVadActive(false);
    setWindowProgress(0);
    latestFinalUtteranceRef.current = 0;
  }, []);

  const startObserver = useCallback(async () => {
    stopObserver();
    lastEventSeqRef.current = -1;
    setMonitorError(null);
    setCaptions([]);
    setDetections([]);
    setSpeakerInfo(null);
    setWindowProgress(0);

    let trackToAnalyze: MediaStreamTrack | null = null;
    if (monitorTarget === "remote") {
      trackToAnalyze = remoteTrackRef.current;
      if (!trackToAnalyze) {
        setMonitorError("Waiting for remote participant audio track...");
        return;
      }
    } else {
      trackToAnalyze = localStreamRef.current?.getAudioTracks()[0] || null;
      if (!trackToAnalyze) {
        setMonitorError("Local microphone not available");
        return;
      }
    }

    try {
      const apiBase = getApiUrl();
      // 1. Create monitor session with explicit track metadata & origin proof
      const sessionRes = await fetch(`${apiBase}/api/monitor/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asr_model_size: "base",
          room_id: roomId,
          observed_peer_id: monitorTarget === "remote" ? "remote_peer" : peerIdRef.current,
          track_id: trackToAnalyze.id,
          track_source: monitorTarget === "remote" ? "REMOTE_PEER" : "LOCAL_MIC",
          track_direction: monitorTarget === "remote" ? "inbound" : "outbound",
          connection_generation: connectionGenerationRef.current,
        }),
      });
      if (!sessionRes.ok) throw new Error("Monitor session creation failed");
      const { session_id } = await sessionRes.json();
      monitorSessionIdRef.current = session_id;

      // 2. Create observer RTCPeerConnection to backend aiortc
      const monitorPc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });
      monitorPcRef.current = monitorPc;

      // Add the selected participant's actual track
      monitorPc.addTrack(trackToAnalyze, new MediaStream([trackToAnalyze]));

      const offer = await monitorPc.createOffer();
      await monitorPc.setLocalDescription(offer);

      // Wait for ICE gathering
      await new Promise<void>((resolve) => {
        if (monitorPc.iceGatheringState === "complete") resolve();
        else {
          monitorPc.onicegatheringstatechange = () => {
            if (monitorPc.iceGatheringState === "complete") resolve();
          };
          setTimeout(resolve, 2000);
        }
      });

      // 3. Send offer to monitor backend
      const offerRes = await fetch(`${apiBase}/api/monitor/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id,
          sdp: monitorPc.localDescription!.sdp,
          type: "offer",
        }),
      });
      if (!offerRes.ok) throw new Error("Observer SDP negotiation failed");
      const { sdp: answerSdp } = await offerRes.json();
      await monitorPc.setRemoteDescription(new RTCSessionDescription({ type: "answer", sdp: answerSdp }));

      // 4. Open monitor event WebSocket
      const wsUrl = getWsUrl();
      const ws = new WebSocket(`${wsUrl}/api/monitor/ws/${session_id}`);
      ws.onopen = () => setMonitorConnected(true);
      ws.onclose = () => setMonitorConnected(false);
      ws.onerror = () => setMonitorError("Observer WebSocket disconnected (Two-Way Call Continues)");

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // Reject stale messages from a previous / mismatched monitor session
          if (data.session_id && data.session_id !== monitorSessionIdRef.current) {
            return;
          }
          // Enforce monotonic event sequence ordering
          if (typeof data.seq === "number") {
            if (data.seq <= lastEventSeqRef.current) {
              return; // Drop out-of-order or replayed frames
            }
            lastEventSeqRef.current = data.seq;
          }

          if (data.type === "telemetry" || data.type === "connected") {
            const t = data.telemetry || data;
            const at = t.audio_truth || {};
            const w = t.webrtc || {};
            setAudioTruth((prev) => ({
              ...prev,
              ...at,
              frames_received: at.frames_received ?? w.frames_decoded ?? w.frames_received ?? prev?.frames_received ?? 0,
              canonical_samples: at.canonical_samples ?? w.total_samples ?? w.samples_received ?? prev?.canonical_samples ?? 0,
              audio_duration: at.audio_duration ?? (w.total_samples ? w.total_samples / 16000 : prev?.audio_duration ?? 0),
              current_timestamp: at.current_timestamp ?? w.last_sample_timestamp ?? (Date.now() / 1000),
              rms: at.rms ?? w.rms ?? w.current_rms ?? prev?.rms ?? 0,
              peak: at.peak ?? w.peak ?? w.current_peak ?? prev?.peak ?? 0,
              vad_probability: at.vad_probability ?? w.vad_probability ?? prev?.vad_probability ?? 0,
              speech_active: at.speech_active ?? w.speech_active ?? t.caption_pipeline?.vad_speech_active ?? prev?.speech_active ?? false,
              track_source: at.track_source ?? w.track_source ?? prev?.track_source ?? (monitorTarget === "remote" ? "REMOTE_PEER" : "LOCAL_MIC"),
              track_direction: at.track_direction ?? w.track_direction ?? prev?.track_direction ?? (monitorTarget === "remote" ? "inbound" : "outbound"),
            }));
            if (t.models) {
              setModelIdentity(t.models);
            }
          } else if (data.type === "protection_decision") {
            setProtectionDecision(data);
          } else if (data.type === "vad") {
            setVadActive(Boolean(data.vad_speech_active));
          } else if (data.type === "caption") {
            const utteranceId = typeof data.utterance_id === "number" ? data.utterance_id : 0;
            if (data.is_partial) {
              if (utteranceId === 0 || utteranceId > latestFinalUtteranceRef.current) {
                setPartialCaption(data.text || "");
              }
            } else {
              if (utteranceId > 0) {
                latestFinalUtteranceRef.current = Math.max(latestFinalUtteranceRef.current, utteranceId);
              }
              setPartialCaption("");
              if (data.text) {
                setCaptions((prev) => {
                  if (utteranceId > 0 && prev.some((c: any) => c.utterance_id === utteranceId)) {
                    return prev;
                  }
                  return [...prev, data];
                });
                setLatestAsrSegment({ start: data.sample_start, end: data.sample_end });
              }
            }
          } else if (data.type === "detection") {
            setDetections((prev) => [...prev, data]);
            if (data.window_valid && data.decision !== "ABSTAIN") {
              setLatestValidWindow({ start: data.sample_start, end: data.sample_end });
            }
          } else if (data.type === "speaker") {
            setSpeakerInfo(data);
          } else if (data.type === "window_progress") {
            setWindowProgress(data.progress_pct || 0);
          }
        } catch {
          // Ignore parse errors
        }
      };
      monitorWsRef.current = ws;

    } catch (err) {
      const msg = err instanceof Error ? err.message : "Observer tap error";
      console.warn(`[VoiceShield Observer] ${msg} (Two-Way Call remains active)`);
      setMonitorError(`${msg} — Two-way call remains connected.`);
    }
  }, [monitorTarget, stopObserver]);

  // Re-run observer tap when target changes or when remote track becomes available
  useEffect(() => {
    if (monitorActive && ((monitorTarget === "remote" && remoteTrackAvailable) || monitorTarget === "local")) {
      startObserver();
    } else if (!monitorActive) {
      stopObserver();
    }
    return () => stopObserver();
  }, [monitorActive, monitorTarget, remoteTrackAvailable, startObserver, stopObserver]);

  // ────────────────────────────────────────────────────────────
  // 3. Two-Way P2P WebRTC State Machine with Perfect Negotiation & Dual Signaling
  // ────────────────────────────────────────────────────────────
  useEffect(() => {
    let isCancelled = false;
    let signalingWs: WebSocket | null = null;
    let p2pPc: RTCPeerConnection | null = null;
    let pollInterval: any = null;
    let lastSeenSeq = 0;
    let isWsConnected = false;
    const queue = new SignalingQueue();

    const sendSignal = (type: string, payload: any) => {
      const msg = {
        type,
        sender_id: peerIdRef.current,
        generation: connectionGenerationRef.current,
        payload,
        timestamp: Date.now() / 1000,
      };

      if (signalingWs && signalingWs.readyState === WebSocket.OPEN) {
        signalingWs.send(JSON.stringify(msg));
      } else {
        fetch("/api/rtc/signal", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "send",
            room_id: roomId,
            peer_id: peerIdRef.current,
            message: msg,
          }),
        }).catch(() => {});
      }
    };

    async function initP2PCall() {
      try {
        // 1. Get user media (microphone)
        const localStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 48000,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
          video: false,
        });

        if (isCancelled) {
          localStream.getTracks().forEach((t) => t.stop());
          return;
        }

        localStreamRef.current = localStream;

        // 2. Setup RTCPeerConnection
        p2pPc = new RTCPeerConnection({
          iceServers: [
            { urls: "stun:stun.l.google.com:19302" },
            { urls: "stun:global.stun.twilio.com:3478" },
          ],
        });
        p2pPcRef.current = p2pPc;

        // Add local mic audio track
        localStream.getAudioTracks().forEach((track) => {
          p2pPc!.addTrack(track, localStream);
        });

        // Remote track received from peer B
        p2pPc.ontrack = (event) => {
          console.log("[P2P WebRTC] Remote track received:", event.track.kind, event.track.id);
          remoteTrackRef.current = event.track;
          if (remoteAudioRef.current) {
            if (event.streams && event.streams[0]) {
              remoteAudioRef.current.srcObject = event.streams[0];
            } else {
              remoteAudioRef.current.srcObject = new MediaStream([event.track]);
            }
          }
          setRemoteTrackAvailable(true);
          setCallState("connected");
        };

        const handleConnUpdate = () => {
          const conn = p2pPc?.connectionState;
          const ice = p2pPc?.iceConnectionState;
          console.log("[P2P WebRTC] State update: conn=", conn, "ice=", ice);
          if (conn === "connected" || ice === "connected" || ice === "completed") {
            setCallState("connected");
          } else if (conn === "failed" || ice === "failed" || conn === "disconnected" || ice === "disconnected") {
            setCallState("waiting");
          }
        };
        p2pPc.onconnectionstatechange = handleConnUpdate;
        p2pPc.oniceconnectionstatechange = handleConnUpdate;

        p2pPc.onicecandidate = (event) => {
          if (event.candidate) {
            sendSignal("candidate", event.candidate);
          }
        };

        // W3C Perfect Negotiation: onnegotiationneeded handler
        p2pPc.onnegotiationneeded = async () => {
          if (peerCountRef.current < 2) {
            console.log("[P2P WebRTC] Skipping onnegotiationneeded — waiting for peer to enter room");
            return;
          }
          await queue.enqueue(async () => {
            if (!p2pPc || isCancelled) return;
            try {
              makingOfferRef.current = true;
              await p2pPc.setLocalDescription();
              sendSignal("offer", p2pPc.localDescription);
            } catch (err) {
              console.error("[P2P WebRTC] Error during onnegotiationneeded:", err);
            } finally {
              makingOfferRef.current = false;
            }
          });
        };

        const handleSignalData = async (data: any) => {
          try {
            const { type, payload, is_polite, generation, peer_count } = data;

            if (generation && generation < connectionGenerationRef.current) {
              console.warn(`[Signaling] Rejecting stale generation message (${generation} < ${connectionGenerationRef.current})`);
              return;
            }

            if (type === "joined") {
              isPoliteRef.current = Boolean(is_polite);
              if (generation) connectionGenerationRef.current = generation;
              const count = peer_count || 1;
              peerCountRef.current = count;
              setPeerCount(count);
              console.log(`[Signaling] Joined room. Role: ${isPoliteRef.current ? "polite" : "impolite"}, Gen: ${generation}, Peers: ${count}`);
              if (count >= 2 && !isPoliteRef.current) {
                await queue.enqueue(async () => {
                  if (!p2pPc || isCancelled) return;
                  try {
                    makingOfferRef.current = true;
                    await p2pPc.setLocalDescription();
                    sendSignal("offer", p2pPc.localDescription);
                  } catch (err) {
                    console.error("[P2P WebRTC] Error initiating offer on join:", err);
                  } finally {
                    makingOfferRef.current = false;
                  }
                });
              }
            } else if (type === "peer_joined") {
              const count = peer_count || 2;
              peerCountRef.current = count;
              setPeerCount(count);
              console.log("[Signaling] Remote peer joined room. Peers:", count);
              if (!isPoliteRef.current) {
                await queue.enqueue(async () => {
                  if (!p2pPc || isCancelled) return;
                  try {
                    makingOfferRef.current = true;
                    await p2pPc.setLocalDescription();
                    sendSignal("offer", p2pPc.localDescription);
                  } catch (err) {
                    console.error("[P2P WebRTC] Error initiating offer to joined peer:", err);
                  } finally {
                    makingOfferRef.current = false;
                  }
                });
              }
            } else if (type === "peer_left") {
              console.log("[Signaling] Remote peer left room.");
              peerCountRef.current = 1;
              setPeerCount(1);
              setCallState("waiting");
              remoteTrackRef.current = null;
              setRemoteTrackAvailable(false);
            } else if (type === "offer" && payload) {
              await queue.enqueue(async () => {
                if (!p2pPc || isCancelled) return;
                const offerCollision = makingOfferRef.current || p2pPc.signalingState !== "stable";
                ignoreOfferRef.current = !isPoliteRef.current && offerCollision;

                if (ignoreOfferRef.current) {
                  console.warn("[P2P WebRTC] Glare collision on impolite peer: safely ignoring remote offer");
                  return;
                }

                if (offerCollision) {
                  console.info("[P2P WebRTC] Glare collision on polite peer: rolling back local description");
                  await p2pPc.setLocalDescription({ type: "rollback" });
                }

                await p2pPc.setRemoteDescription(new RTCSessionDescription(payload));
                await p2pPc.setLocalDescription();
                sendSignal("answer", p2pPc.localDescription);
              });
            } else if (type === "answer" && payload) {
              await queue.enqueue(async () => {
                if (!p2pPc || isCancelled) return;
                if (p2pPc.signalingState !== "have-local-offer" && p2pPc.signalingState !== "have-remote-pranswer") {
                  console.warn(`[P2P WebRTC] Rejecting stale answer in signalingState=${p2pPc.signalingState}`);
                  return;
                }
                await p2pPc.setRemoteDescription(new RTCSessionDescription(payload));
              });
            } else if (type === "candidate" && payload) {
              await queue.enqueue(async () => {
                if (!p2pPc || isCancelled) return;
                try {
                  await p2pPc.addIceCandidate(new RTCIceCandidate(payload));
                } catch (err) {
                  if (!ignoreOfferRef.current) {
                    console.warn("[P2P WebRTC] addIceCandidate error:", err);
                  }
                }
              });
            }
          } catch (err) {
            console.error("[Signaling] Message parse/processing error:", err);
          }
        };

        const startHttpSignaling = async () => {
          if (isCancelled || isWsConnected) return;
          console.log("[Signaling] Using HTTP polling signaling for room", roomId);
          setCallState("waiting");

          try {
            const joinRes = await fetch("/api/rtc/signal", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ action: "join", room_id: roomId, peer_id: peerIdRef.current }),
            });
            if (joinRes.ok) {
              const joinData = await joinRes.json();
              lastSeenSeq = joinData.current_seq || 0;
              await handleSignalData(joinData);
            }
          } catch (e) {
            console.warn("HTTP join error", e);
          }

          if (pollInterval) clearInterval(pollInterval);
          pollInterval = setInterval(async () => {
            if (isCancelled || isWsConnected) return;
            try {
              const pollRes = await fetch("/api/rtc/signal", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "poll", room_id: roomId, peer_id: peerIdRef.current, last_seq: lastSeenSeq }),
              });
              if (pollRes.ok) {
                const pollData = await pollRes.json();
                if (typeof pollData.current_seq === "number") {
                  lastSeenSeq = pollData.current_seq;
                }
                if (typeof pollData.peer_count === "number" && pollData.peer_count !== peerCountRef.current) {
                  peerCountRef.current = pollData.peer_count;
                  setPeerCount(pollData.peer_count);
                }
                if (Array.isArray(pollData.messages)) {
                  for (const m of pollData.messages) {
                    await handleSignalData(m);
                  }
                }
              }
            } catch {
              // Ignore polling hiccups
            }
          }, 700);
        };

        // 3. Setup WebSocket Signaling with peer_id query param
        try {
          const wsUrl = getWsUrl();
          signalingWs = new WebSocket(`${wsUrl}/api/rtc/ws/${roomId}?peer_id=${peerIdRef.current}`);
          signalingWsRef.current = signalingWs;

          signalingWs.onopen = () => {
            console.log("[Signaling] WebSocket connected to room", roomId, "as", peerIdRef.current);
            isWsConnected = true;
            if (pollInterval) clearInterval(pollInterval);
            setCallState("waiting");
          };

          signalingWs.onmessage = async (msgEvent) => {
            try {
              const data = JSON.parse(msgEvent.data);
              await handleSignalData(data);
            } catch (err) {
              console.error("[Signaling] WS parse error:", err);
            }
          };

          signalingWs.onerror = () => {
            if (!isWsConnected) {
              startHttpSignaling();
            }
          };

          // Fallback to HTTP if WS doesn't open within 1.2s
          setTimeout(() => {
            if (!isWsConnected) {
              startHttpSignaling();
            }
          }, 1200);
        } catch {
          startHttpSignaling();
        }

      } catch (err) {
        if (!isCancelled) {
          console.error("Init P2P call failed:", err);
          setCallState("ended");
        }
      }
    }

    initP2PCall();

    return () => {
      isCancelled = true;
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
      fetch("/api/rtc/signal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "leave", room_id: roomId, peer_id: peerIdRef.current }),
        keepalive: true,
      }).catch(() => {});
      if (signalingWsRef.current) {
        signalingWsRef.current.close();
        signalingWsRef.current = null;
      }
      if (p2pPcRef.current) {
        p2pPcRef.current.close();
        p2pPcRef.current = null;
      }
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((t) => t.stop());
        localStreamRef.current = null;
      }
      stopObserver();
    };
  }, [roomId, stopObserver]);

  // ────────────────────────────────────────────────────────────
  // 4. Call Controls
  // ────────────────────────────────────────────────────────────
  const toggleMute = () => {
    if (localStreamRef.current) {
      const audioTrack = localStreamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsMuted(!audioTrack.enabled);
      }
    }
  };

  const endCall = () => {
    if (signalingWsRef.current) {
      signalingWsRef.current.send(JSON.stringify({
        type: "bye",
        sender_id: peerIdRef.current,
        generation: connectionGenerationRef.current,
      }));
      signalingWsRef.current.close();
    }
    if (p2pPcRef.current) p2pPcRef.current.close();
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((t) => t.stop());
    }
    stopObserver();
    setCallState("ended");
    router.push("/call");
  };

  const latestDetection = detections.length > 0 ? detections[detections.length - 1] : null;
  const validDetections = detections.filter((d) => d.window_valid && d.decision !== "ABSTAIN" && d.decision !== "INVALID");
  const suspiciousWindows = validDetections.filter((d) => d.decision === "SUSPICIOUS");
  const genuineWindows = validDetections.filter((d) => d.decision === "GENUINE");
  const uncertainWindows = validDetections.filter((d) => d.decision === "UNCERTAIN");
  const abstainCount = detections.filter((d) => d.decision === "ABSTAIN").length;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Hidden Audio Output for Remote Participant */}
      <audio ref={remoteAudioRef} autoPlay playsInline />

      {/* Header & Sticky Call Control Bar */}
      <div className="rounded-2xl border border-gray-800 bg-gray-900/90 backdrop-blur p-4 shadow-xl flex flex-wrap items-center justify-between gap-4 sticky top-4 z-50">
        <div className="flex items-center gap-3">
          <Link href="/call" className="text-gray-400 hover:text-white text-xs">
            ← Lobby
          </Link>
          <div className="h-4 w-px bg-gray-800" />
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-cyan-400 font-semibold">{roomId}</span>
              <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${
                callState === "connected"
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                  : peerCount >= 2
                  ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
                  : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
              }`}>
                {callState === "connected"
                  ? "Call Connected"
                  : peerCount >= 2
                  ? "Connecting / Negotiating..."
                  : "Waiting for Peer"}
              </span>
            </div>
            <span className="text-xs font-mono text-gray-400">
              Duration: {formatDuration(callDuration)} · Peers: {peerCount}/2
            </span>
          </div>
        </div>

        {/* Primary Call Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={toggleMute}
            className={`px-4 py-2 rounded-xl text-xs font-medium transition-all cursor-pointer border ${
              isMuted
                ? "bg-amber-500/20 text-amber-400 border-amber-500/30"
                : "bg-gray-800 hover:bg-gray-700 text-gray-200 border-gray-700"
            }`}
          >
            {isMuted ? "🔇 Unmute Mic" : "🎙 Mute Mic"}
          </button>

          <button
            onClick={endCall}
            className="px-4 py-2 rounded-xl bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-500/30 text-xs font-semibold transition-all cursor-pointer"
          >
            End Call
          </button>
        </div>
      </div>

      {/* Controlled Observer Monitor Control Panel (Non-Invasive) */}
      <div className="rounded-2xl border border-gray-800/80 bg-gray-950/60 p-5 shadow-lg">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4 pb-3 border-b border-gray-800/60">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-white tracking-wide uppercase">
              VoiceShield Observer Monitor (Non-Invasive)
            </h2>
            <span className={`text-xs px-2 py-0.5 rounded ${
              monitorConnected ? "bg-emerald-500/10 text-emerald-400" : "bg-gray-800 text-gray-400"
            }`}>
              {monitorConnected ? "● Observing" : "○ Disconnected"}
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Participant Track Selector */}
            <div className="flex items-center bg-gray-900 rounded-lg p-1 border border-gray-800 text-xs">
              <span className="text-gray-500 px-2">Analyze:</span>
              <button
                onClick={() => setMonitorTarget("remote")}
                className={`px-2.5 py-1 rounded-md transition-all ${
                  monitorTarget === "remote"
                    ? "bg-cyan-500 text-white font-medium"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                Remote Participant
              </button>
              <button
                onClick={() => setMonitorTarget("local")}
                className={`px-2.5 py-1 rounded-md transition-all ${
                  monitorTarget === "local"
                    ? "bg-cyan-500 text-white font-medium"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                Local Microphone
              </button>
            </div>

            <button
              onClick={() => setMonitorActive(!monitorActive)}
              className="px-3 py-1 bg-gray-900 border border-gray-800 text-xs text-gray-300 rounded-lg hover:bg-gray-800 transition"
            >
              {monitorActive ? "Pause Monitor" : "Resume Monitor"}
            </button>
          </div>
        </div>

        {monitorError && (
          <div className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 p-2.5 rounded-lg mb-4">
            ℹ {monitorError}
          </div>
        )}

        {/* 4 Multi-Signal Observer Panels */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Panel 1: Live Speech */}
          <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/60">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Target Speech Detection ({monitorTarget === "remote" ? "Remote Peer" : "Local Mic"})
            </h3>
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${
                !monitorConnected
                  ? "bg-gray-600"
                  : vadActive
                    ? "bg-emerald-400 animate-pulse shadow-lg shadow-emerald-400/50"
                    : (audioTruth?.frames_received ?? 0) === 0
                      ? "bg-amber-400 animate-ping"
                      : "bg-blue-400"
              }`} />
              <span className="text-sm font-medium text-gray-200">
                {!monitorConnected
                  ? "OBSERVER DISCONNECTED"
                  : vadActive
                    ? "🎙 SPEECH DETECTED"
                    : (audioTruth?.frames_received ?? 0) === 0
                      ? "AWAITING AUDIO INPUT"
                      : "— SILENCE (Verified VAD Silence)"}
              </span>
            </div>
          </div>

          {/* Panel 2: Live Captions (Voice-Assistant Style Streaming) */}
          <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/60">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Live Captions ({modelIdentity?.asr_model ? `${modelIdentity.asr_provider} (${modelIdentity.asr_model})` : "Streaming Whisper"})
              </h3>
              {partialCaption && (
                <span className="text-[10px] text-cyan-400 animate-pulse font-mono font-medium flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" /> STREAMING
                </span>
              )}
            </div>

            {/* Real-time partial caption hypothesis */}
            {partialCaption && (
              <div className="mb-2 p-2 rounded-lg bg-cyan-950/40 border border-cyan-500/30 text-xs font-mono text-cyan-200">
                <span className="text-cyan-500 text-[10px] uppercase font-bold mr-1.5">[streaming hypothesis]</span>
                {partialCaption}
              </div>
            )}

            <div className="space-y-1.5 max-h-36 overflow-y-auto text-xs font-mono">
              {captions.length === 0 && !partialCaption ? (
                <p className="text-gray-600 italic">Waiting for speech...</p>
              ) : (
                captions.map((c, i) => (
                  <div key={i} className="flex items-baseline justify-between gap-2 p-1.5 rounded bg-gray-950/40 border border-gray-800/40">
                    <span className="text-gray-200">{c.text}</span>
                    <span className="text-[10px] text-gray-500 shrink-0">
                      [{c.sample_start}..{c.sample_end}] ({c.latency_ms.toFixed(0)}ms)
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Panel 3: Anti-Spoof Evidence */}
          <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/60">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Anti-Spoof Evidence (Spectra-AASIST)
              </h3>
              <span className="text-[10px] text-gray-500 font-mono">
                Window Cadence: 4.04s
              </span>
            </div>

            {/* Backend Protection Decision Badge (Single Source of Truth) */}
            <div className={`mb-3 p-3 rounded-xl border ${
              protectionDecision?.decision === "BLOCK" || protectionDecision?.decision === "WARN" || latestDetection?.decision === "SUSPICIOUS"
                ? "bg-red-500/10 border-red-500/30 text-red-400"
                : protectionDecision?.decision === "MONITOR" || protectionDecision?.decision === "VERIFY" || suspiciousWindows.length > 0
                  ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                  : protectionDecision?.decision === "SAFE" || latestDetection?.decision === "GENUINE"
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                    : latestDetection?.decision === "ABSTAIN"
                      ? "bg-gray-900/80 border-gray-800 text-slate-400"
                      : "bg-gray-900/80 border-gray-800 text-slate-300"
            }`}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <span className="text-[10px] text-gray-400 uppercase tracking-wider block font-semibold">
                    Backend Protection Decision (Single Source of Truth)
                  </span>
                  <span className="text-xs font-mono font-bold">
                    {protectionDecision
                      ? `${protectionDecision.decision} (${protectionDecision.risk_level} RISK) — ${protectionDecision.reason}`
                      : latestDetection?.decision === "ABSTAIN"
                        ? "CURRENT: SUB-GATE ACOUSTIC ENERGY (Silence / Quality Gated)"
                        : "AWAITING SPEECH WINDOW (4.04s accumulation)"}
                  </span>
                </div>
                <div className="text-right text-[11px] font-mono text-gray-400">
                  <span>Valid: {protectionDecision?.window_count ?? validDetections.length} · Anomalies: {protectionDecision?.suspicious_count ?? suspiciousWindows.length} · Abstain: {abstainCount}</span>
                </div>
              </div>

              {/* Historical Anomaly Audit Trail */}
              {suspiciousWindows.length > 0 && (
                <div className="mt-2 pt-2 border-t border-amber-500/20 text-[11px] font-mono text-amber-300/90">
                  <span className="font-semibold text-amber-400">Historical Anomaly Audit Trail: </span>
                  {suspiciousWindows.map((w, idx) => (
                    <span key={idx} className="mr-2 inline-block">
                      Window #{w.window_index} (samples {w.sample_start}–{w.sample_end}, score {w.spoof_score.toFixed(4)})
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Window progress */}
            <div className="mb-3">
              <div className="flex justify-between text-[11px] text-gray-500 mb-1">
                <span>Next Window Accumulation</span>
                <span>{windowProgress.toFixed(0)}%</span>
              </div>
              <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-500 rounded-full transition-all duration-200"
                  style={{ width: `${Math.min(windowProgress, 100)}%` }}
                />
              </div>
            </div>

            {latestDetection ? (
              <div className="space-y-2 text-xs">
                <div className="grid grid-cols-2 gap-2 bg-gray-950 p-2.5 rounded-lg border border-gray-800/60 font-mono text-[11px]">
                  <div>
                    <span className="text-gray-500">Raw Spoof Logit: </span>
                    <span className="text-gray-300">{latestDetection.raw_logit_spoof.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Raw Bonafide Logit: </span>
                    <span className="text-gray-300">{latestDetection.raw_logit_bonafide.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Raw Softmax Spoof: </span>
                    <span className="text-gray-300">{latestDetection.spoof_score.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Raw Bonafide: </span>
                    <span className="text-gray-300">{latestDetection.bonafide_score.toFixed(4)}</span>
                  </div>
                </div>

                <div className="pt-2 border-t border-gray-800/60 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500 block text-[11px]">Calibrated Spoof Probability</span>
                    <div className="flex items-baseline gap-1.5">
                      <span className="font-mono text-sm font-semibold text-gray-200">
                        {latestDetection.calibrated_probability !== null && latestDetection.calibrated_probability !== undefined
                          ? `${(latestDetection.calibrated_probability * 100).toFixed(1)}%`
                          : "N/A"}
                      </span>
                      {latestDetection.calibrated_probability !== null && (
                        <span className="text-[9px] text-amber-400/80 font-mono uppercase">
                          (Experimental)
                        </span>
                      )}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-500 block text-[11px]">Latest Window Decision</span>
                    <span className={`font-mono text-sm font-bold ${
                      latestDetection.decision === "SUSPICIOUS"
                        ? "text-red-400"
                        : latestDetection.decision === "GENUINE"
                          ? "text-emerald-400"
                          : latestDetection.decision === "UNCERTAIN"
                            ? "text-amber-400"
                            : "text-slate-300"
                    }`}>
                      {latestDetection.decision || "UNCALIBRATED"}
                    </span>
                  </div>
                </div>

                <div className="pt-2 border-t border-gray-800/60 text-[11px] text-gray-400 space-y-1">
                  <div className="flex justify-between items-center">
                    <span>Evidence Breakdown: <strong className="text-gray-200">{suspiciousWindows.length} Suspicious · {genuineWindows.length} Genuine · {uncertainWindows.length} Uncertain</strong></span>
                    <span className="text-[10px] text-amber-400/90 font-mono">Calibration: Moderate (ECE 0.19)</span>
                  </div>
                  <p className="text-[10px] text-gray-500 italic">
                    * Single-window probabilities are provisional estimates undergoing multi-condition testing, not definitive voice clone percentages.
                  </p>
                </div>

                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-gray-500">
                    Status:{" "}
                    <span className={
                      latestDetection.calibration_status === "CALIBRATED"
                        ? "text-emerald-400"
                        : latestDetection.calibration_status === "NOT_APPLICABLE_SILENCE"
                          ? "text-slate-400"
                          : "text-amber-400"
                    }>
                      {latestDetection.calibration_status === "NOT_APPLICABLE_SILENCE"
                        ? "SILENCE (PAUSE)"
                        : latestDetection.calibration_status}
                    </span>
                  </span>
                  <span className="text-gray-500 font-mono">
                    Latency: {latestDetection.latency_ms.toFixed(0)}ms
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-gray-600 text-xs italic">Waiting for first 4.04s AASIST window...</p>
            )}
          </div>

          {/* Panel 4: Speaker Evidence (ECAPA) */}
          <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/60">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Speaker Evidence (ECAPA-TDNN)
            </h3>
            {speakerInfo ? (
              <div className="space-y-2 text-xs font-mono text-gray-400">
                <p>Embedding Dimension: {speakerInfo.embedding_dim}-dim</p>
                <p>Similarity: {speakerInfo.speaker_similarity !== null ? speakerInfo.speaker_similarity.toFixed(4) : "— (no enrolled profile)"}</p>
                <p className="text-gray-600">Latency: {speakerInfo.latency_ms.toFixed(0)}ms</p>
              </div>
            ) : (
              <p className="text-gray-600 text-xs italic">Waiting for validated speech segment...</p>
            )}
          </div>
        </div>

        {/* LIVE AUDIO TRUTH PANEL & RUNTIME MODEL IDENTITY (Phase 2 & Phase 18) */}
        <div className="mt-6 pt-5 border-t border-gray-800/80 grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Diagnostic Panel: Live Audio Truth */}
          <div className="p-4 rounded-xl bg-gray-950 border border-cyan-500/20 font-mono text-xs text-gray-300">
            <div className="flex items-center justify-between pb-2 mb-3 border-b border-gray-800">
              <h4 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400" />
                LIVE AUDIO TRUTH
              </h4>
              <span className="text-[10px] text-gray-500">Zero-Trust Audio Lineage</span>
            </div>

            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
              <div>
                <span className="text-gray-500">Session: </span>
                <span className="text-gray-200">{audioTruth?.session_id ? `${audioTruth.session_id.substring(0, 12)}...` : "—"}</span>
              </div>
              <div>
                <span className="text-gray-500">Track: </span>
                <span className="text-cyan-300 font-bold">{audioTruth?.track_id || "—"}</span>
              </div>
              <div>
                <span className="text-gray-500">Peer: </span>
                <span className="text-gray-200">{audioTruth?.observed_peer_id || "—"}</span>
              </div>
              <div>
                <span className="text-gray-500">Direction: </span>
                <span className="text-gray-200">{audioTruth?.track_direction || "—"}</span>
              </div>
              <div className="col-span-2 pt-1 border-t border-gray-800/60">
                <span className="text-gray-500">Source: </span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                  audioTruth?.track_source === "REMOTE_PEER"
                    ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                    : "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                }`}>
                  {audioTruth?.track_source || (monitorTarget === "remote" ? "REMOTE_PEER" : "LOCAL_MIC")}
                </span>
              </div>

              <div className="pt-2">
                <span className="text-gray-500">Frames received: </span>
                <span className="text-gray-200">{audioTruth?.frames_received ?? 0}</span>
              </div>
              <div className="pt-2">
                <span className="text-gray-500">Canonical samples: </span>
                <span className="text-gray-200">{audioTruth?.canonical_samples ?? 0}</span>
              </div>
              <div>
                <span className="text-gray-500">Audio duration: </span>
                <span className="text-gray-200">{(audioTruth?.audio_duration ?? 0).toFixed(2)}s</span>
              </div>
              <div>
                <span className="text-gray-500">Current timestamp: </span>
                <span className="text-gray-200">{(audioTruth?.current_timestamp ?? 0).toFixed(2)}s</span>
              </div>

              <div className="col-span-2 pt-2 border-t border-gray-800/60 grid grid-cols-2 gap-x-4 gap-y-1.5">
                <div>
                  <span className="text-gray-500">RMS: </span>
                  <span className={`font-bold ${
                    (audioTruth?.rms ?? 0) < 0.003 ? "text-amber-400" : "text-emerald-400"
                  }`}>
                    {(audioTruth?.rms ?? 0).toFixed(5)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Peak: </span>
                  <span className="text-gray-200">{(audioTruth?.peak ?? 0).toFixed(5)}</span>
                </div>
                <div>
                  <span className="text-gray-500">VAD probability: </span>
                  <span className="text-gray-200">{(audioTruth?.vad_probability ?? 0).toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-gray-500">Speech active: </span>
                  <span className={audioTruth?.speech_active ? "text-emerald-400 font-bold" : "text-gray-500"}>
                    {audioTruth?.speech_active ? "TRUE (VAD ONSET)" : "FALSE (SILENCE)"}
                  </span>
                </div>
              </div>

              <div className="col-span-2 pt-2 border-t border-gray-800/60">
                <span className="text-gray-500 block">AASIST latest valid window:</span>
                <span className="text-gray-300">
                  {latestValidWindow ? `start: ${latestValidWindow.start} · end: ${latestValidWindow.end} (${((latestValidWindow.end - latestValidWindow.start)/16000).toFixed(2)}s)` : "none (all windows abstained on silence/noise)"}
                </span>
              </div>

              <div className="col-span-2 pt-1">
                <span className="text-gray-500 block">ASR latest segment:</span>
                <span className="text-gray-300">
                  {latestAsrSegment ? `start: ${latestAsrSegment.start} · end: ${latestAsrSegment.end} (${((latestAsrSegment.end - latestAsrSegment.start)/16000).toFixed(2)}s)` : "none (awaiting speech endpointing)"}
                </span>
              </div>
            </div>
          </div>

          {/* Diagnostic Panel: Runtime Model Identity */}
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
                <span className="text-gray-200 font-semibold">{modelIdentity?.asr_provider || "UNAVAILABLE"}</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-gray-900">
                <span className="text-gray-500">ASR Model:</span>
                <span className="text-gray-200 font-semibold">{modelIdentity?.asr_model || "UNAVAILABLE"}</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-gray-900">
                <span className="text-gray-500">ASR Backend:</span>
                <span className="text-gray-200 font-semibold">{modelIdentity?.asr_backend || "UNAVAILABLE"}</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-gray-900">
                <span className="text-gray-500">AASIST Model:</span>
                <span className="text-gray-200 font-semibold">{modelIdentity?.aasist_model || "UNAVAILABLE"}</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-gray-900">
                <span className="text-gray-500">AASIST Checkpoint:</span>
                <span className="text-emerald-400 font-semibold">{modelIdentity?.aasist_loaded ? "Loaded (weights_aasist.pth)" : (modelIdentity ? "Not loaded" : "UNAVAILABLE")}</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-gray-900">
                <span className="text-gray-500">ECAPA Model:</span>
                <span className="text-gray-200 font-semibold">{modelIdentity?.ecapa_model || "UNAVAILABLE"}</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-gray-900">
                <span className="text-gray-500">VAD Model:</span>
                <span className="text-gray-200 font-semibold">{modelIdentity?.vad_model || "UNAVAILABLE"}</span>
              </div>
              <div className="flex justify-between items-center py-1">
                <span className="text-gray-500">Calibration:</span>
                <span className={`font-semibold ${
                  modelIdentity?.calibration_status?.startsWith("CALIBRATED") ? "text-emerald-400" : "text-amber-400"
                }`}>
                  {modelIdentity?.calibration_status || "UNCALIBRATED"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
