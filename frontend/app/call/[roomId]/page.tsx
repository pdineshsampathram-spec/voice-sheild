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
 * Sequential operation queue to serialize WebRTC signaling operations.
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

  // Mode Selection: "discord" (Clean 2-Way Voice Room) vs "shield" (Security Forensic View)
  const [uiMode, setUiMode] = useState<"discord" | "shield">("discord");
  const [showNoShieldModal, setShowNoShieldModal] = useState(false);
  const [isBackendAvailable, setIsBackendAvailable] = useState<boolean | null>(null);

  // P2P WebRTC Call State
  const [callState, setCallState] = useState<"connecting" | "waiting" | "connected" | "ended">("connecting");
  const [peerCount, setPeerCount] = useState(1);
  const [remotePeerId, setRemotePeerId] = useState<string | null>(null);
  const [remoteTrackAvailable, setRemoteTrackAvailable] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isDeafened, setIsDeafened] = useState(false);
  const [remoteVolume, setRemoteVolume] = useState(1.0);
  const [callDuration, setCallDuration] = useState(0);
  const [autoplayBlocked, setAutoplayBlocked] = useState(false);
  const [copiedLink, setCopiedLink] = useState(false);

  // Live Audio Meters (Discord-style Voice Activity Detection)
  const [isLocalSpeaking, setIsLocalSpeaking] = useState(false);
  const [isRemoteSpeaking, setIsRemoteSpeaking] = useState(false);
  const [localAudioLevel, setLocalAudioLevel] = useState(0); // 0 to 100
  const [remoteAudioLevel, setRemoteAudioLevel] = useState(0); // 0 to 100

  // Monitor Observer Tap State (for optional Shield view)
  const [monitorActive, setMonitorActive] = useState(false);
  const [monitorTarget, setMonitorTarget] = useState<"remote" | "local">("remote");
  const [monitorConnected, setMonitorConnected] = useState(false);
  const [monitorError, setMonitorError] = useState<string | null>(null);

  // ML Evidence State (for optional Shield view)
  const [audioTruth, setAudioTruth] = useState<AudioTruthData | null>(null);
  const [protectionDecision, setProtectionDecision] = useState<ProtectionDecisionData | null>(null);
  const [modelIdentity, setModelIdentity] = useState<ModelIdentityData | null>(null);
  const [partialCaption, setPartialCaption] = useState<string>("");
  const [latestValidWindow, setLatestValidWindow] = useState<{ start: number; end: number } | null>(null);
  const [latestAsrSegment, setLatestAsrSegment] = useState<{ start: number; end: number } | null>(null);
  const [vadActive, setVadActive] = useState(false);
  const [captions, setCaptions] = useState<CaptionItem[]>([]);
  const [detections, setDetections] = useState<DetectionItem[]>([]);
  const [speakerInfo, setSpeakerInfo] = useState<SpeakerItem | null>(null);
  const [windowProgress, setWindowProgress] = useState<number>(0);

  // WebRTC & Audio Processing Refs
  const localStreamRef = useRef<MediaStream | null>(null);
  const remoteStreamRef = useRef<MediaStream | null>(null);
  const remoteAudioRef = useRef<HTMLAudioElement | null>(null);
  const remoteTrackRef = useRef<MediaStreamTrack | null>(null);
  const p2pPcRef = useRef<RTCPeerConnection | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const remoteGainNodeRef = useRef<GainNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Signaling & State Refs
  const peerIdRef = useRef<string>(`peer_${Math.random().toString(36).substring(2, 9)}`);
  const isPoliteRef = useRef<boolean>(false);
  const makingOfferRef = useRef<boolean>(false);
  const ignoreOfferRef = useRef<boolean>(false);
  const remoteDescriptionSetRef = useRef<boolean>(false);
  const iceCandidateQueueRef = useRef<RTCIceCandidateInit[]>([]);
  const signalingWsRef = useRef<WebSocket | null>(null);
  const ntfyWsRef = useRef<WebSocket | null>(null);
  const processedMsgIdsRef = useRef<Set<string>>(new Set());

  // Monitor Refs
  const monitorPcRef = useRef<RTCPeerConnection | null>(null);
  const monitorWsRef = useRef<WebSocket | null>(null);
  const monitorSessionIdRef = useRef<string | null>(null);
  const lastEventSeqRef = useRef<number>(-1);
  const latestFinalUtteranceRef = useRef<number>(0);

  // ────────────────────────────────────────────────────────────
  // 1. Backend Probe Check on Page Mount
  // ────────────────────────────────────────────────────────────
  useEffect(() => {
    async function checkBackend() {
      try {
        const res = await fetch("/api/monitor/probe", {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
        if (res.ok) {
          const data = await res.json();
          if (data.available) {
            setIsBackendAvailable(true);
            // If backend is active, user can start in shield or discord
            return;
          }
        }
      } catch {
        // Fallback
      }
      // Backend validation not found or not deployed
      setIsBackendAvailable(false);
      setShowNoShieldModal(true);
      setUiMode("discord");
    }
    checkBackend();
  }, []);

  // ────────────────────────────────────────────────────────────
  // 2. Call Duration Timer
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
  // 3. Web Audio Setup & Live Volume Monitoring
  // ────────────────────────────────────────────────────────────
  const getOrCreateAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      audioContextRef.current = new AudioCtx();
    }
    if (audioContextRef.current.state === "suspended") {
      audioContextRef.current.resume().catch(() => {});
    }
    return audioContextRef.current;
  }, []);

  const setupAudioProcessing = useCallback((localStream: MediaStream, remoteStream?: MediaStream) => {
    try {
      const audioCtx = getOrCreateAudioContext();

      // Local microphone analyzer
      const localSource = audioCtx.createMediaStreamSource(localStream);
      const localAnalyser = audioCtx.createAnalyser();
      localAnalyser.fftSize = 256;
      localSource.connect(localAnalyser);

      // Remote incoming audio analyzer + gain routing
      let remoteAnalyser: AnalyserNode | null = null;
      if (remoteStream) {
        const remoteSource = audioCtx.createMediaStreamSource(remoteStream);
        remoteAnalyser = audioCtx.createAnalyser();
        remoteAnalyser.fftSize = 256;

        const gainNode = audioCtx.createGain();
        gainNode.gain.value = isDeafened ? 0 : remoteVolume;
        remoteGainNodeRef.current = gainNode;

        remoteSource.connect(remoteAnalyser);
        remoteAnalyser.connect(gainNode);
        gainNode.connect(audioCtx.destination);
      }

      const localDataArray = new Uint8Array(localAnalyser.frequencyBinCount);
      const remoteDataArray = remoteAnalyser ? new Uint8Array(remoteAnalyser.frequencyBinCount) : null;

      const updateVolumeLoop = () => {
        // Measure local volume
        localAnalyser.getByteFrequencyData(localDataArray);
        let localSum = 0;
        for (let i = 0; i < localDataArray.length; i++) {
          localSum += localDataArray[i];
        }
        const localAvg = localSum / localDataArray.length;
        const localPct = Math.min(100, Math.round((localAvg / 128) * 100));
        setLocalAudioLevel(localPct);
        setIsLocalSpeaking(localPct > 12 && !isMuted);

        // Measure remote volume
        if (remoteAnalyser && remoteDataArray) {
          remoteAnalyser.getByteFrequencyData(remoteDataArray);
          let remoteSum = 0;
          for (let i = 0; i < remoteDataArray.length; i++) {
            remoteSum += remoteDataArray[i];
          }
          const remoteAvg = remoteSum / remoteDataArray.length;
          const remotePct = Math.min(100, Math.round((remoteAvg / 128) * 100));
          setRemoteAudioLevel(remotePct);
          setIsRemoteSpeaking(remotePct > 12);
        }

        animationFrameRef.current = requestAnimationFrame(updateVolumeLoop);
      };

      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = requestAnimationFrame(updateVolumeLoop);
    } catch (e) {
      console.warn("[WebAudio] Setup warning:", e);
    }
  }, [getOrCreateAudioContext, isDeafened, isMuted, remoteVolume]);

  // Audio Test Chime for speakers
  const playTestChime = () => {
    try {
      const audioCtx = getOrCreateAudioContext();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
      osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.15); // A5
      gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.4);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.4);
    } catch (e) {
      console.warn("Chime error:", e);
    }
  };

  // ────────────────────────────────────────────────────────────
  // 4. Two-Way P2P WebRTC Core Engine with Guaranteed Audio
  // ────────────────────────────────────────────────────────────
  useEffect(() => {
    let isCancelled = false;
    const queue = new SignalingQueue();
    let pollInterval: any = null;
    let heartbeatInterval: any = null;
    let lastSeenSeq = 0;

    const sanitizedTopic = `voiceshield_rtc_${roomId.toLowerCase().replace(/[^a-z0-9]/g, "_")}`;

    // Universal Signal Sender (ntfy.sh + /api/rtc/signal fallback)
    const sendSignal = async (type: string, payload: any) => {
      const msgId = `${peerIdRef.current}_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
      processedMsgIdsRef.current.add(msgId);

      const envelope = {
        msg_id: msgId,
        type,
        sender_id: peerIdRef.current,
        payload,
        timestamp: Date.now() / 1000,
      };

      // 1. Broadcast via public high-speed pub/sub broker
      fetch(`https://ntfy.sh/${sanitizedTopic}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(envelope),
      }).catch(() => {});

      // 2. Also send to local Next.js signaling route as secondary fallback
      fetch("/api/rtc/signal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "send",
          room_id: roomId,
          peer_id: peerIdRef.current,
          message: envelope,
        }),
      }).catch(() => {});
    };

    async function drainIceCandidateQueue(pc: RTCPeerConnection) {
      remoteDescriptionSetRef.current = true;
      while (iceCandidateQueueRef.current.length > 0) {
        const candidate = iceCandidateQueueRef.current.shift();
        if (candidate) {
          try {
            await pc.addIceCandidate(new RTCIceCandidate(candidate));
          } catch (err) {
            console.warn("[P2P WebRTC] Queued ICE candidate add error:", err);
          }
        }
      }
    }

    async function handleIncomingSignal(data: any, pc: RTCPeerConnection) {
      if (isCancelled) return;
      try {
        const { msg_id, type, sender_id, payload } = data;

        // Skip our own messages
        if (sender_id === peerIdRef.current) return;

        // Deduplicate messages received across both signaling pathways
        if (msg_id && processedMsgIdsRef.current.has(msg_id)) return;
        if (msg_id) processedMsgIdsRef.current.add(msg_id);

        if (sender_id) {
          setRemotePeerId(sender_id);
          setPeerCount(2);
          // W3C Polite Peer rule: deterministic tie-breaker
          isPoliteRef.current = peerIdRef.current > sender_id;
        }

        if (type === "join") {
          console.log("[P2P WebRTC] Peer joined room:", sender_id);
          sendSignal("peer_online", { peer_id: peerIdRef.current });
          setCallState("waiting");

          // Impolite peer initiates offer
          if (!isPoliteRef.current && pc.signalingState === "stable") {
            await queue.enqueue(async () => {
              if (isCancelled || !pc) return;
              try {
                makingOfferRef.current = true;
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                sendSignal("offer", pc.localDescription);
              } catch (err) {
                console.error("[P2P WebRTC] Error creating offer on join:", err);
              } finally {
                makingOfferRef.current = false;
              }
            });
          }
        } else if (type === "peer_online") {
          console.log("[P2P WebRTC] Peer confirmed online:", sender_id);
          if (!isPoliteRef.current && pc.signalingState === "stable") {
            await queue.enqueue(async () => {
              if (isCancelled || !pc) return;
              try {
                makingOfferRef.current = true;
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                sendSignal("offer", pc.localDescription);
              } catch (err) {
                console.error("[P2P WebRTC] Error creating offer on peer_online:", err);
              } finally {
                makingOfferRef.current = false;
              }
            });
          }
        } else if (type === "offer" && payload) {
          await queue.enqueue(async () => {
            if (isCancelled || !pc) return;
            const offerCollision = makingOfferRef.current || pc.signalingState !== "stable";
            ignoreOfferRef.current = !isPoliteRef.current && offerCollision;

            if (ignoreOfferRef.current) {
              console.warn("[P2P WebRTC] Glare collision on impolite peer: ignoring remote offer");
              return;
            }

            if (offerCollision) {
              console.info("[P2P WebRTC] Glare collision on polite peer: rolling back local description");
              await pc.setLocalDescription({ type: "rollback" });
            }

            await pc.setRemoteDescription(new RTCSessionDescription(payload));
            await drainIceCandidateQueue(pc);

            const answer = await pc.createAnswer();
            await pc.setLocalDescription(answer);
            sendSignal("answer", pc.localDescription);
          });
        } else if (type === "answer" && payload) {
          await queue.enqueue(async () => {
            if (isCancelled || !pc) return;
            if (pc.signalingState === "have-local-offer" || pc.signalingState === "have-remote-pranswer") {
              await pc.setRemoteDescription(new RTCSessionDescription(payload));
              await drainIceCandidateQueue(pc);
            }
          });
        } else if (type === "candidate" && payload) {
          await queue.enqueue(async () => {
            if (isCancelled || !pc) return;
            if (pc.remoteDescription && remoteDescriptionSetRef.current) {
              try {
                await pc.addIceCandidate(new RTCIceCandidate(payload));
              } catch (err) {
                if (!ignoreOfferRef.current) {
                  console.warn("[P2P WebRTC] addIceCandidate error:", err);
                }
              }
            } else {
              iceCandidateQueueRef.current.push(payload);
            }
          });
        } else if (type === "bye") {
          console.log("[P2P WebRTC] Peer left call");
          setPeerCount(1);
          setRemotePeerId(null);
          setCallState("waiting");
          setRemoteTrackAvailable(false);
          remoteTrackRef.current = null;
        }
      } catch (err) {
        console.error("[P2P WebRTC] Signal handle error:", err);
      }
    }

    async function initCall() {
      try {
        // 1. Acquire Local Microphone
        const stream = await navigator.mediaDevices.getUserMedia({
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
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        localStreamRef.current = stream;
        setupAudioProcessing(stream);

        // 2. Setup RTCPeerConnection with STUN
        const pc = new RTCPeerConnection({
          iceServers: [
            { urls: "stun:stun.l.google.com:19302" },
            { urls: "stun:stun1.l.google.com:19302" },
            { urls: "stun:stun2.l.google.com:19302" },
            { urls: "stun:stun.cloudflare.com:3478" },
            { urls: "stun:global.stun.twilio.com:3478" },
          ],
        });
        p2pPcRef.current = pc;

        // Add local mic track
        stream.getAudioTracks().forEach((track) => {
          track.enabled = true;
          pc.addTrack(track, stream);
        });

        // Send ICE candidates
        pc.onicecandidate = (event) => {
          if (event.candidate) {
            sendSignal("candidate", event.candidate);
          }
        };

        // Remote track received
        pc.ontrack = (event) => {
          console.log("[P2P WebRTC] Remote audio track received:", event.track.id);
          remoteTrackRef.current = event.track;
          const remoteStream = event.streams && event.streams[0] ? event.streams[0] : new MediaStream([event.track]);
          remoteStreamRef.current = remoteStream;

          // Connect to HTML audio tag
          if (remoteAudioRef.current) {
            remoteAudioRef.current.srcObject = remoteStream;
            remoteAudioRef.current.muted = false;
            remoteAudioRef.current.volume = 1.0;
            const playPromise = remoteAudioRef.current.play();
            if (playPromise !== undefined) {
              playPromise.catch((err) => {
                console.warn("[P2P WebRTC] Autoplay pending user interaction:", err);
                setAutoplayBlocked(true);
              });
            }
          }

          // Connect to Web Audio Analyser and destination
          setupAudioProcessing(stream, remoteStream);
          setRemoteTrackAvailable(true);
          setCallState("connected");
        };

        const handleConnChange = () => {
          const conn = pc.connectionState;
          const ice = pc.iceConnectionState;
          console.log(`[P2P WebRTC] State changed: conn=${conn}, ice=${ice}`);
          if (conn === "connected" || ice === "connected" || ice === "completed") {
            setCallState("connected");
          } else if (conn === "failed" || ice === "failed") {
            setCallState("waiting");
          }
        };
        pc.onconnectionstatechange = handleConnChange;
        pc.oniceconnectionstatechange = handleConnChange;

        // 3. Connect to High-Speed ntfy.sh Pub/Sub WebSocket
        try {
          const ntfyWs = new WebSocket(`wss://ntfy.sh/${sanitizedTopic}/ws`);
          ntfyWsRef.current = ntfyWs;

          ntfyWs.onopen = () => {
            console.log("[Signaling] ntfy.sh WebSocket opened for room", roomId);
            // Broadcast join
            sendSignal("join", { peer_id: peerIdRef.current });
          };

          ntfyWs.onmessage = async (event) => {
            try {
              const raw = JSON.parse(event.data);
              if (raw.event === "message" && raw.message) {
                const payload = typeof raw.message === "string" ? JSON.parse(raw.message) : raw.message;
                await handleIncomingSignal(payload, pc);
              }
            } catch {
              // Ignore ping/keepalive parse errors
            }
          };

          ntfyWs.onerror = () => {
            console.warn("[Signaling] ntfy.sh WebSocket error; relying on HTTP fallback");
          };
        } catch (e) {
          console.warn("[Signaling] ntfy.sh init error:", e);
        }

        // 4. Secondary HTTP Polling Fallback (ensures reliability)
        pollInterval = setInterval(async () => {
          if (isCancelled) return;
          try {
            const pollRes = await fetch("/api/rtc/signal", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                action: "poll",
                room_id: roomId,
                peer_id: peerIdRef.current,
                last_seq: lastSeenSeq,
              }),
            });
            if (pollRes.ok) {
              const data = await pollRes.json();
              if (typeof data.current_seq === "number") {
                lastSeenSeq = data.current_seq;
              }
              if (Array.isArray(data.messages)) {
                for (const m of data.messages) {
                  await handleIncomingSignal(m, pc);
                }
              }
            }
          } catch {
            // Polling hiccup
          }
        }, 800);

        // 5. Periodic Heartbeat / Room Ping until connected
        heartbeatInterval = setInterval(() => {
          if (isCancelled || callState === "connected") return;
          sendSignal("ping", { peer_id: peerIdRef.current });
        }, 2500);

        // Initial announce
        sendSignal("join", { peer_id: peerIdRef.current });
      } catch (err) {
        if (!isCancelled) {
          console.error("Init P2P call failed:", err);
          setCallState("ended");
        }
      }
    }

    initCall();

    return () => {
      isCancelled = true;
      if (pollInterval) clearInterval(pollInterval);
      if (heartbeatInterval) clearInterval(heartbeatInterval);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

      sendSignal("bye", { peer_id: peerIdRef.current });

      if (ntfyWsRef.current) {
        ntfyWsRef.current.close();
        ntfyWsRef.current = null;
      }
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
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
        audioContextRef.current = null;
      }
    };
  }, [roomId, setupAudioProcessing]);

  // ────────────────────────────────────────────────────────────
  // 5. Observer Monitor Tap (Shield View Only)
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
      const sessionRes = await fetch(`${apiBase}/api/monitor/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asr_model_size: "base",
          room_id: roomId,
          observed_peer_id: monitorTarget === "remote" ? (remotePeerId || "remote_peer") : peerIdRef.current,
          track_id: trackToAnalyze.id,
          track_source: monitorTarget === "remote" ? "REMOTE_PEER" : "LOCAL_MIC",
          track_direction: monitorTarget === "remote" ? "inbound" : "outbound",
          connection_generation: 1,
        }),
      });
      if (!sessionRes.ok) throw new Error("Monitor session creation failed");
      const { session_id } = await sessionRes.json();
      monitorSessionIdRef.current = session_id;

      const monitorPc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });
      monitorPcRef.current = monitorPc;
      monitorPc.addTrack(trackToAnalyze, new MediaStream([trackToAnalyze]));

      const offer = await monitorPc.createOffer();
      await monitorPc.setLocalDescription(offer);

      await new Promise<void>((resolve) => {
        if (monitorPc.iceGatheringState === "complete") resolve();
        else {
          monitorPc.onicegatheringstatechange = () => {
            if (monitorPc.iceGatheringState === "complete") resolve();
          };
          setTimeout(resolve, 2000);
        }
      });

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

      const wsUrl = getWsUrl();
      const ws = new WebSocket(`${wsUrl}/api/monitor/ws/${session_id}`);
      ws.onopen = () => setMonitorConnected(true);
      ws.onclose = () => setMonitorConnected(false);
      ws.onerror = () => setMonitorError("Observer WebSocket disconnected");

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.session_id && data.session_id !== monitorSessionIdRef.current) return;
          if (typeof data.seq === "number") {
            if (data.seq <= lastEventSeqRef.current) return;
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
            if (t.models) setModelIdentity(t.models);
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
              if (utteranceId > 0) latestFinalUtteranceRef.current = Math.max(latestFinalUtteranceRef.current, utteranceId);
              setPartialCaption("");
              if (data.text) {
                setCaptions((prev) => (utteranceId > 0 && prev.some((c: any) => c.utterance_id === utteranceId) ? prev : [...prev, data]));
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
      console.warn(`[VoiceShield Observer] ${msg}`);
      setMonitorError(`${msg} — Two-way voice call remains connected.`);
    }
  }, [monitorTarget, remotePeerId, roomId, stopObserver]);

  useEffect(() => {
    if (uiMode === "shield" && monitorActive && ((monitorTarget === "remote" && remoteTrackAvailable) || monitorTarget === "local")) {
      startObserver();
    } else {
      stopObserver();
    }
    return () => stopObserver();
  }, [uiMode, monitorActive, monitorTarget, remoteTrackAvailable, startObserver, stopObserver]);

  // ────────────────────────────────────────────────────────────
  // 6. User Call Actions (Mute, Deafen, Volume, Hangup)
  // ────────────────────────────────────────────────────────────
  const toggleMute = () => {
    if (localStreamRef.current) {
      const track = localStreamRef.current.getAudioTracks()[0];
      if (track) {
        track.enabled = !track.enabled;
        setIsMuted(!track.enabled);
      }
    }
  };

  const toggleDeafen = () => {
    const nextDeaf = !isDeafened;
    setIsDeafened(nextDeaf);
    if (remoteAudioRef.current) {
      remoteAudioRef.current.muted = nextDeaf;
    }
    if (remoteGainNodeRef.current) {
      remoteGainNodeRef.current.gain.value = nextDeaf ? 0 : remoteVolume;
    }
  };

  const handleVolumeChange = (newVol: number) => {
    setRemoteVolume(newVol);
    if (remoteAudioRef.current && !isDeafened) {
      remoteAudioRef.current.volume = Math.min(1.0, newVol);
    }
    if (remoteGainNodeRef.current && !isDeafened) {
      remoteGainNodeRef.current.gain.value = newVol;
    }
  };

  const unlockAudio = () => {
    const audioCtx = getOrCreateAudioContext();
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    if (remoteAudioRef.current) {
      remoteAudioRef.current.play().catch(() => {});
    }
    setAutoplayBlocked(false);
  };

  const handleProceedWithoutShield = () => {
    setShowNoShieldModal(false);
    setUiMode("discord");
    unlockAudio();
  };

  const handleEndCall = () => {
    if (p2pPcRef.current) p2pPcRef.current.close();
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((t) => t.stop());
    }
    stopObserver();
    setCallState("ended");
    router.push("/call");
  };

  const handleCopyLink = () => {
    const url = typeof window !== "undefined" ? window.location.href : "";
    navigator.clipboard.writeText(url);
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2500);
  };

  const latestDetection = detections.length > 0 ? detections[detections.length - 1] : null;
  const validDetections = detections.filter((d) => d.window_valid && d.decision !== "ABSTAIN" && d.decision !== "INVALID");
  const suspiciousWindows = validDetections.filter((d) => d.decision === "SUSPICIOUS");
  const genuineWindows = validDetections.filter((d) => d.decision === "GENUINE");
  const uncertainWindows = validDetections.filter((d) => d.decision === "UNCERTAIN");
  const abstainCount = detections.filter((d) => d.decision === "ABSTAIN").length;

  return (
    <div className="min-h-screen bg-[#1e1f22] text-gray-200 -mx-4 -my-8 p-4 md:p-8 flex flex-col justify-between">
      {/* Hidden Audio Output Element */}
      <audio ref={remoteAudioRef} autoPlay playsInline />

      {/* ────────────────────────────────────────────────────────────
          POPUP MODAL: VoiceShield Validation Not Found
          ──────────────────────────────────────────────────────────── */}
      {showNoShieldModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-[#2b2d31] border border-amber-500/30 rounded-2xl p-6 sm:p-8 max-w-lg w-full shadow-2xl text-center transform transition-all animate-in fade-in zoom-in-95 duration-200">
            {/* Warning Shield Icon */}
            <div className="w-16 h-16 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 mx-auto mb-4 flex items-center justify-center text-3xl">
              🛡️
            </div>

            <h3 className="text-xl font-bold text-white mb-2">
              VoiceShield Validation Not Found
            </h3>

            <p className="text-sm font-medium text-amber-300 mb-3">
              Entering the room without VoiceShield because the validation was not found.
            </p>

            <p className="text-xs text-gray-400 mb-6 leading-relaxed">
              The AI deepfake protection backend is offline or not deployed. You can still proceed with a high-definition, two-way online voice call with your peer like Discord.
            </p>

            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <button
                onClick={handleProceedWithoutShield}
                className="px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold text-sm shadow-lg shadow-emerald-900/30 transition-all cursor-pointer flex items-center justify-center gap-2"
              >
                <span>Proceed to Call</span>
                <span>→</span>
              </button>
              <button
                onClick={handleEndCall}
                className="px-6 py-3 rounded-xl bg-[#383a40] hover:bg-[#474b53] text-gray-300 hover:text-white font-medium text-sm transition-all cursor-pointer border border-gray-700"
              >
                End Call
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Autoplay blocked banner */}
      {autoplayBlocked && (
        <div
          onClick={unlockAudio}
          className="fixed top-4 left-1/2 -translate-x-1/2 z-40 bg-emerald-600 hover:bg-emerald-500 text-white px-5 py-2.5 rounded-full text-xs font-semibold shadow-xl cursor-pointer flex items-center gap-2 animate-bounce"
        >
          <span>🔊</span>
          <span>Click here to enable audio playback (Browser Autoplay Blocked)</span>
        </div>
      )}

      {/* ────────────────────────────────────────────────────────────
          HEADER: Discord-style Channel Header
          ──────────────────────────────────────────────────────────── */}
      <header className="rounded-2xl bg-[#2b2d31] border border-[#1e1f22] p-4 shadow-md flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link href="/call" className="text-gray-400 hover:text-white text-xs px-2 py-1 rounded bg-[#1e1f22]">
            ← Lobby
          </Link>
          <div className="h-4 w-px bg-gray-700" />
          <div className="flex items-center gap-2 font-mono">
            <span className="text-gray-400 font-bold">#</span>
            <span className="text-white font-bold text-sm tracking-wide">{roomId}</span>
          </div>
          <span
            className={`text-[10px] font-semibold px-2.5 py-0.5 rounded-full flex items-center gap-1.5 ${
              callState === "connected"
                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                : peerCount >= 2
                ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
                : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                callState === "connected"
                  ? "bg-emerald-400 animate-pulse"
                  : peerCount >= 2
                  ? "bg-sky-400"
                  : "bg-amber-400"
              }`}
            />
            {callState === "connected"
              ? "Voice Connected"
              : peerCount >= 2
              ? "Connecting..."
              : "Waiting for Peer"}
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs">
          <span className="font-mono text-gray-400">
            {formatDuration(callDuration)}
          </span>
          <span className="text-gray-400 font-mono hidden sm:inline">
            Peers: {peerCount}/2
          </span>

          {/* Toggle between Clean Discord UI and Forensic Shield View (if backend available) */}
          {isBackendAvailable && (
            <div className="flex items-center bg-[#1e1f22] p-1 rounded-lg border border-gray-800 text-[11px]">
              <button
                onClick={() => setUiMode("discord")}
                className={`px-3 py-1 rounded-md transition-all ${
                  uiMode === "discord"
                    ? "bg-[#5865f2] text-white font-semibold"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                Voice Room
              </button>
              <button
                onClick={() => setUiMode("shield")}
                className={`px-3 py-1 rounded-md transition-all ${
                  uiMode === "shield"
                    ? "bg-cyan-600 text-white font-semibold"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                Shield Diagnostics
              </button>
            </div>
          )}
        </div>
      </header>

      {/* ────────────────────────────────────────────────────────────
          MAIN CONTENT AREA
          ──────────────────────────────────────────────────────────── */}
      <main className="my-6 flex-1 flex flex-col justify-center">
        {uiMode === "discord" ? (
          /* ════════════════════════════════════════════════════════════
             DISCORD-STYLE VOICE ROOM UI (Non-VoiceShield)
             ════════════════════════════════════════════════════════════ */
          <div className="max-w-4xl w-full mx-auto space-y-6">
            {/* 2-Caller Voice Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Local User Card (You) */}
              <div className="bg-[#2b2d31] rounded-2xl p-8 border border-[#1e1f22] shadow-xl flex flex-col items-center justify-center relative min-h-[280px]">
                {/* Speaking Ring Avatar */}
                <div className="relative mb-4">
                  <div
                    className={`w-28 h-28 rounded-full bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center text-3xl font-bold text-white shadow-2xl transition-all duration-150 ${
                      isLocalSpeaking
                        ? "ring-4 ring-[#23a55a] shadow-[0_0_25px_rgba(35,165,90,0.8)] scale-105"
                        : "ring-2 ring-gray-700/50"
                    }`}
                  >
                    YOU
                  </div>
                  {/* Mic mute indicator badge on avatar */}
                  <div
                    className={`absolute bottom-0 right-0 w-8 h-8 rounded-full flex items-center justify-center text-xs shadow-md border-2 border-[#2b2d31] ${
                      isMuted ? "bg-[#da373c] text-white" : "bg-[#23a55a] text-white"
                    }`}
                  >
                    {isMuted ? "🔇" : "🎙"}
                  </div>
                </div>

                <div className="text-center">
                  <h3 className="text-base font-bold text-white flex items-center justify-center gap-2">
                    You <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#1e1f22] text-indigo-400">Caller A</span>
                  </h3>
                  <p className="text-xs text-gray-400 mt-1">
                    {isMuted ? "Microphone Muted" : isLocalSpeaking ? "Speaking..." : "Active"}
                  </p>
                </div>

                {/* Animated Audio Waveform Bars */}
                <div className="flex items-center gap-1.5 mt-5 h-6">
                  {[0.4, 0.8, 1, 0.7, 0.3].map((factor, i) => (
                    <div
                      key={i}
                      className="w-1.5 bg-[#23a55a] rounded-full transition-all duration-75"
                      style={{
                        height: isLocalSpeaking ? `${Math.max(4, localAudioLevel * factor * 0.25)}px` : "4px",
                        opacity: isLocalSpeaking ? 1 : 0.3,
                      }}
                    />
                  ))}
                </div>
              </div>

              {/* Remote Peer Card */}
              <div className="bg-[#2b2d31] rounded-2xl p-8 border border-[#1e1f22] shadow-xl flex flex-col items-center justify-center relative min-h-[280px]">
                {callState === "connected" && remoteTrackAvailable ? (
                  <>
                    {/* Speaking Ring Avatar */}
                    <div className="relative mb-4">
                      <div
                        className={`w-28 h-28 rounded-full bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center text-3xl font-bold text-white shadow-2xl transition-all duration-150 ${
                          isRemoteSpeaking
                            ? "ring-4 ring-[#23a55a] shadow-[0_0_25px_rgba(35,165,90,0.8)] scale-105"
                            : "ring-2 ring-gray-700/50"
                        }`}
                      >
                        PEER
                      </div>
                      <div className="absolute bottom-0 right-0 w-8 h-8 rounded-full bg-[#23a55a] text-white flex items-center justify-center text-xs shadow-md border-2 border-[#2b2d31]">
                        🔊
                      </div>
                    </div>

                    <div className="text-center">
                      <h3 className="text-base font-bold text-white flex items-center justify-center gap-2">
                        Remote Caller <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#1e1f22] text-cyan-400">Caller B</span>
                      </h3>
                      <p className="text-xs text-gray-400 mt-1 font-mono">
                        {isRemoteSpeaking ? "Speaking..." : "Connected · Listening"}
                      </p>
                    </div>

                    {/* Animated Audio Waveform Bars */}
                    <div className="flex items-center gap-1.5 mt-5 h-6">
                      {[0.5, 0.9, 1, 0.8, 0.4].map((factor, i) => (
                        <div
                          key={i}
                          className="w-1.5 bg-[#23a55a] rounded-full transition-all duration-75"
                          style={{
                            height: isRemoteSpeaking ? `${Math.max(4, remoteAudioLevel * factor * 0.25)}px` : "4px",
                            opacity: isRemoteSpeaking ? 1 : 0.3,
                          }}
                        />
                      ))}
                    </div>

                    {/* Remote Volume Slider */}
                    <div className="mt-4 flex items-center gap-2 text-xs text-gray-400 w-48">
                      <span className="text-[10px]">Vol:</span>
                      <input
                        type="range"
                        min="0"
                        max="1.5"
                        step="0.05"
                        value={remoteVolume}
                        onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
                        className="w-full accent-[#5865f2] h-1.5 bg-[#1e1f22] rounded-lg cursor-pointer"
                      />
                      <span className="text-[10px] font-mono">{Math.round(remoteVolume * 100)}%</span>
                    </div>
                  </>
                ) : (
                  /* Waiting for Peer State */
                  <div className="text-center space-y-4">
                    <div className="w-24 h-24 rounded-full bg-[#1e1f22] border-2 border-dashed border-gray-700 mx-auto flex items-center justify-center text-gray-500 text-3xl animate-pulse">
                      👤
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-gray-300">Waiting for Peer</h3>
                      <p className="text-xs text-gray-500 mt-1 max-w-xs mx-auto">
                        Share this room link with your peer to establish two-way voice communication.
                      </p>
                    </div>
                    <button
                      onClick={handleCopyLink}
                      className="px-4 py-2 rounded-xl bg-[#5865f2] hover:bg-[#4752c4] text-white text-xs font-semibold shadow transition-all cursor-pointer inline-flex items-center gap-1.5"
                    >
                      {copiedLink ? "✓ Copied Invite Link" : "📋 Copy Room Invite Link"}
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Direct Voice Mode Status Pill */}
            <div className="text-center">
              <span className="text-[11px] font-mono text-gray-500 bg-[#2b2d31] px-4 py-1.5 rounded-full border border-gray-800">
                Direct P2P Voice Stream · High-Definition Opus 48kHz
              </span>
            </div>
          </div>
        ) : (
          /* ════════════════════════════════════════════════════════════
             SECURITY DIAGNOSTICS VIEW (Only when backend is active & enabled)
             ════════════════════════════════════════════════════════════ */
          <div className="space-y-6 max-w-7xl mx-auto w-full">
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
                  <div className="flex items-center bg-gray-900 rounded-lg p-1 border border-gray-800 text-xs">
                    <span className="text-gray-500 px-2">Analyze:</span>
                    <button
                      onClick={() => setMonitorTarget("remote")}
                      className={`px-2.5 py-1 rounded-md transition-all ${
                        monitorTarget === "remote" ? "bg-cyan-500 text-white font-medium" : "text-gray-400 hover:text-white"
                      }`}
                    >
                      Remote Participant
                    </button>
                    <button
                      onClick={() => setMonitorTarget("local")}
                      className={`px-2.5 py-1 rounded-md transition-all ${
                        monitorTarget === "local" ? "bg-cyan-500 text-white font-medium" : "text-gray-400 hover:text-white"
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

              {/* Multi-Signal Observer Panels */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Speech Detection */}
                <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/60">
                  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                    Target Speech Detection ({monitorTarget === "remote" ? "Remote Peer" : "Local Mic"})
                  </h3>
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${
                      !monitorConnected ? "bg-gray-600" : vadActive ? "bg-emerald-400 animate-pulse shadow-lg shadow-emerald-400/50" : "bg-blue-400"
                    }`} />
                    <span className="text-sm font-medium text-gray-200">
                      {!monitorConnected ? "OBSERVER DISCONNECTED" : vadActive ? "🎙 SPEECH DETECTED" : "— SILENCE (Verified VAD Silence)"}
                    </span>
                  </div>
                </div>

                {/* Live Captions */}
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

                  {partialCaption && (
                    <div className="mb-2 p-2 rounded-lg bg-cyan-950/40 border border-cyan-500/30 text-xs font-mono text-cyan-200">
                      <span className="text-cyan-500 text-[10px] uppercase font-bold mr-1.5">[streaming]</span>
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

                {/* Anti-Spoof Evidence */}
                <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/60">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      Anti-Spoof Evidence (Spectra-AASIST)
                    </h3>
                    <span className="text-[10px] text-gray-500 font-mono">
                      Window Cadence: 4.04s
                    </span>
                  </div>

                  <div className={`mb-3 p-3 rounded-xl border ${
                    protectionDecision?.decision === "BLOCK" || protectionDecision?.decision === "WARN" || latestDetection?.decision === "SUSPICIOUS"
                      ? "bg-red-500/10 border-red-500/30 text-red-400"
                      : protectionDecision?.decision === "SAFE" || latestDetection?.decision === "GENUINE"
                      ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                      : "bg-gray-900/80 border-gray-800 text-slate-300"
                  }`}>
                    <span className="text-[10px] text-gray-400 uppercase tracking-wider block font-semibold">
                      Backend Protection Decision
                    </span>
                    <span className="text-xs font-mono font-bold">
                      {protectionDecision ? `${protectionDecision.decision} (${protectionDecision.risk_level} RISK) — ${protectionDecision.reason}` : "AWAITING SPEECH WINDOW (4.04s)"}
                    </span>
                  </div>

                  {latestDetection && (
                    <div className="grid grid-cols-2 gap-2 bg-gray-950 p-2.5 rounded-lg border border-gray-800/60 font-mono text-[11px]">
                      <div>
                        <span className="text-gray-500">Spoof Score: </span>
                        <span className="text-gray-300">{latestDetection.spoof_score.toFixed(4)}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Decision: </span>
                        <span className={latestDetection.decision === "SUSPICIOUS" ? "text-red-400 font-bold" : "text-emerald-400 font-bold"}>
                          {latestDetection.decision}
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Speaker Biometrics */}
                <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/60">
                  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                    Speaker Evidence (ECAPA-TDNN)
                  </h3>
                  {speakerInfo ? (
                    <div className="space-y-2 text-xs font-mono text-gray-400">
                      <p>Embedding Dimension: {speakerInfo.embedding_dim}-dim</p>
                      <p>Similarity: {speakerInfo.speaker_similarity !== null ? speakerInfo.speaker_similarity.toFixed(4) : "—"}</p>
                    </div>
                  ) : (
                    <p className="text-gray-600 text-xs italic">Waiting for validated speech segment...</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ────────────────────────────────────────────────────────────
          FOOTER: Discord-Style Floating Voice Control Deck
          ──────────────────────────────────────────────────────────── */}
      <footer className="sticky bottom-4 z-30">
        <div className="bg-[#2b2d31]/95 backdrop-blur-md rounded-2xl p-3 border border-[#1e1f22] shadow-2xl flex items-center justify-center gap-3 sm:gap-4 max-w-lg mx-auto">
          {/* Mute Microphone */}
          <button
            onClick={toggleMute}
            title={isMuted ? "Unmute Microphone" : "Mute Microphone"}
            className={`w-12 h-12 rounded-full flex items-center justify-center text-lg transition-all cursor-pointer shadow-md ${
              isMuted
                ? "bg-[#da373c] text-white hover:bg-[#a1282c]"
                : "bg-[#383a40] text-gray-200 hover:bg-[#474b53] hover:text-white"
            }`}
          >
            {isMuted ? "🔇" : "🎙"}
          </button>

          {/* Deafen (Mute Incoming Audio) */}
          <button
            onClick={toggleDeafen}
            title={isDeafened ? "Undeafen Audio" : "Deafen Audio"}
            className={`w-12 h-12 rounded-full flex items-center justify-center text-lg transition-all cursor-pointer shadow-md ${
              isDeafened
                ? "bg-[#da373c] text-white hover:bg-[#a1282c]"
                : "bg-[#383a40] text-gray-200 hover:bg-[#474b53] hover:text-white"
            }`}
          >
            {isDeafened ? "🔕" : "🎧"}
          </button>

          {/* Speaker Sound Test */}
          <button
            onClick={playTestChime}
            title="Test Speakers / Audio Output"
            className="w-12 h-12 rounded-full bg-[#383a40] text-gray-200 hover:bg-[#474b53] hover:text-white flex items-center justify-center text-lg transition-all cursor-pointer shadow-md"
          >
            🔔
          </button>

          {/* Copy Invite Link */}
          <button
            onClick={handleCopyLink}
            title="Copy Invite Link"
            className="w-12 h-12 rounded-full bg-[#383a40] text-gray-200 hover:bg-[#474b53] hover:text-white flex items-center justify-center text-lg transition-all cursor-pointer shadow-md"
          >
            {copiedLink ? "✓" : "📋"}
          </button>

          {/* Disconnect / End Call Button */}
          <button
            onClick={handleEndCall}
            title="Disconnect Call"
            className="w-12 h-12 rounded-full bg-[#da373c] hover:bg-[#a1282c] text-white flex items-center justify-center text-lg transition-all cursor-pointer shadow-lg shadow-red-900/40"
          >
            📞
          </button>
        </div>
      </footer>
    </div>
  );
}
