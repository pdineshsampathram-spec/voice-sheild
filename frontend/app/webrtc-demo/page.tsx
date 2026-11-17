'use client';

import { useState, useRef, useEffect, useCallback } from 'react';

/**
 * WebRTC Demo Caller — Browser-side peer for VoiceShield testing.
 *
 * This page:
 * 1. Captures local microphone via getUserMedia
 * 2. Connects to FastAPI WebSocket signaling server
 * 3. Creates RTCPeerConnection
 * 4. Exchanges SDP offer/answer + ICE candidates
 * 5. Sends local audio track to VoiceShield Android
 *
 * The VoiceShield Android app receives this remote audio and analyzes it.
 */
export default function WebRtcDemoPage() {
  const [roomId, setRoomId] = useState('voiceshield-test');
  const [backendUrl, setBackendUrl] = useState('ws://localhost:8000/api/rtc/ws');
  const [connectionState, setConnectionState] = useState('Disconnected');
  const [iceState, setIceState] = useState('new');
  const [signalingState, setSignalingState] = useState('Disconnected');
  const [audioStats, setAudioStats] = useState({
    bytesSent: 0,
    packetsSent: 0,
    sampleRate: 0,
  });
  const [error, setError] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [localStreamActive, setLocalStreamActive] = useState(false);

  const pcRef = useRef<RTCPeerConnection | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const statsIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const cleanup = useCallback(() => {
    if (statsIntervalRef.current) {
      clearInterval(statsIntervalRef.current);
      statsIntervalRef.current = null;
    }
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((t: MediaStreamTrack) => t.stop());
      localStreamRef.current = null;
    }
    setLocalStreamActive(false);
    setIsConnected(false);
    setConnectionState('Disconnected');
    setIceState('new');
    setSignalingState('Disconnected');
  }, []);

  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  const handleSignalingMessage = (msg: { type: string; payload: Record<string, unknown> }) => {
    const pc = pcRef.current;
    if (!pc) return;

    if (msg.type === 'answer') {
      const payload = msg.payload as { sdp?: string };
      if (!payload.sdp) return;
      pc.setRemoteDescription(
        new RTCSessionDescription({
          type: 'answer',
          sdp: payload.sdp,
        })
      ).then(() => {
        setConnectionState('Connected');
      }).catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Answer failed: ${message}`);
      });
    } else if (msg.type === 'candidate') {
      const payload = msg.payload as { candidate?: string; sdpMid?: string | null; sdpMLineIndex?: number | null };
      if (payload.candidate) {
        pc.addIceCandidate(
          new RTCIceCandidate({
            sdpMid: payload.sdpMid ?? undefined,
            sdpMLineIndex: payload.sdpMLineIndex ?? undefined,
            candidate: payload.candidate,
          })
        ).catch((err: unknown) => console.warn('ICE candidate error:', err));
      }
    }
  };

  const startStatsPolling = (pc: RTCPeerConnection) => {
    statsIntervalRef.current = setInterval(async () => {
      try {
        const stats = await pc.getStats();
        stats.forEach((report) => {
          if (report.type === 'outbound-rtp' && report.kind === 'audio') {
            setAudioStats({
              bytesSent: (report.bytesSent as number) || 0,
              packetsSent: (report.packetsSent as number) || 0,
              sampleRate: (report.sampleRate as number) || 0,
            });
          }
        });
      } catch {
        // Stats not available yet
      }
    }, 1000);
  };

  const createPeerConnection = (ws: WebSocket, stream: MediaStream) => {
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    });
    pcRef.current = pc;

    // Add local audio track
    stream.getAudioTracks().forEach((track: MediaStreamTrack) => {
      pc.addTrack(track, stream);
    });

    pc.onicecandidate = (event: RTCPeerConnectionIceEvent) => {
      if (event.candidate) {
        ws.send(
          JSON.stringify({
            type: 'candidate',
            payload: {
              sdpMid: event.candidate.sdpMid,
              sdpMLineIndex: event.candidate.sdpMLineIndex,
              candidate: event.candidate.candidate,
            },
          })
        );
      }
    };

    pc.oniceconnectionstatechange = () => {
      setIceState(pc.iceConnectionState);
      if (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed') {
        setIsConnected(true);
        startStatsPolling(pc);
      }
    };

    pc.onconnectionstatechange = () => {
      setConnectionState(pc.connectionState);
    };

    // Create and send offer
    pc.createOffer()
      .then((offer) => pc.setLocalDescription(offer))
      .then(() => {
        if (pc.localDescription) {
          ws.send(
            JSON.stringify({
              type: 'offer',
              payload: {
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
              },
            })
          );
        }
        setConnectionState('Offering...');
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Offer failed: ${message}`);
      });
  };

  const startCall = async () => {
    setError('');
    try {
      // 1. Get microphone
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });
      localStreamRef.current = stream;
      setLocalStreamActive(true);

      // 2. Connect to signaling
      const wsUrl = `${backendUrl}/${roomId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setSignalingState('Connected');
        createPeerConnection(ws, stream);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          handleSignalingMessage(msg);
        } catch (e) {
          console.error('Error parsing signaling message', e);
        }
      };

      ws.onerror = () => {
        setError('WebSocket error — is the backend running?');
        setSignalingState('Error');
      };

      ws.onclose = () => {
        setSignalingState('Disconnected');
      };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Failed: ${message}`);
    }
  };

  const endCall = () => {
    cleanup();
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#0A0E14', 
      color: '#E0E6ED', 
      fontFamily: 'system-ui, -apple-system, sans-serif',
      padding: '2rem'
    }}>
      <div style={{ maxWidth: '600px', margin: '0 auto' }}>
        <h1 style={{ color: '#00E5FF', fontSize: '1.8rem', marginBottom: '0.5rem' }}>
          VoiceShield WebRTC Demo
        </h1>
        <p style={{ color: '#8B95A5', marginBottom: '2rem' }}>
          Browser-side caller for testing live voice analysis
        </p>

        {/* Connection Config */}
        <div style={{ 
          backgroundColor: '#111822', 
          borderRadius: '12px', 
          padding: '1.25rem',
          marginBottom: '1rem'
        }}>
          <label style={{ display: 'block', color: '#8B95A5', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
            Backend WebSocket URL
          </label>
          <input
            type="text"
            value={backendUrl}
            onChange={(e) => setBackendUrl(e.target.value)}
            disabled={isConnected}
            style={{
              width: '100%', padding: '0.5rem', borderRadius: '6px',
              backgroundColor: '#1A2332', color: '#E0E6ED', border: '1px solid #2A3342',
              marginBottom: '0.75rem', boxSizing: 'border-box'
            }}
          />
          <label style={{ display: 'block', color: '#8B95A5', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
            Room ID
          </label>
          <input
            type="text"
            value={roomId}
            onChange={(e) => setRoomId(e.target.value)}
            disabled={isConnected}
            style={{
              width: '100%', padding: '0.5rem', borderRadius: '6px',
              backgroundColor: '#1A2332', color: '#E0E6ED', border: '1px solid #2A3342',
              boxSizing: 'border-box'
            }}
          />
        </div>

        {/* Call Button */}
        <div style={{ textAlign: 'center', margin: '1.5rem 0' }}>
          {!isConnected ? (
            <button
              onClick={startCall}
              style={{
                backgroundColor: '#00E676', color: '#000', border: 'none',
                padding: '0.75rem 2rem', borderRadius: '30px', fontSize: '1rem',
                fontWeight: 'bold', cursor: 'pointer'
              }}
            >
              📞 Start Call
            </button>
          ) : (
            <button
              onClick={endCall}
              style={{
                backgroundColor: '#FF1744', color: '#FFF', border: 'none',
                padding: '0.75rem 2rem', borderRadius: '30px', fontSize: '1rem',
                fontWeight: 'bold', cursor: 'pointer'
              }}
            >
              📵 End Call
            </button>
          )}
        </div>

        {/* Error */}
        {error && (
          <div style={{
            backgroundColor: '#FF174420', border: '1px solid #FF1744',
            borderRadius: '8px', padding: '0.75rem', marginBottom: '1rem',
            color: '#FF1744', fontSize: '0.9rem'
          }}>
            {error}
          </div>
        )}

        {/* Status Dashboard */}
        <div style={{ 
          backgroundColor: '#111822', 
          borderRadius: '12px', 
          padding: '1.25rem',
          marginBottom: '1rem'
        }}>
          <h2 style={{ color: '#00E5FF', fontSize: '1rem', marginBottom: '1rem' }}>
            Connection Status
          </h2>
          <StatusRow label="Signaling" value={signalingState} 
            color={signalingState === 'Connected' ? '#00E676' : '#8B95A5'} />
          <StatusRow label="Connection" value={connectionState}
            color={connectionState === 'Connected' ? '#00E676' : '#8B95A5'} />
          <StatusRow label="ICE State" value={iceState}
            color={['connected', 'completed'].includes(iceState) ? '#00E676' : '#8B95A5'} />
          <StatusRow label="Local Mic" value={localStreamActive ? 'ACTIVE' : 'INACTIVE'}
            color={localStreamActive ? '#00E676' : '#8B95A5'} />
        </div>

        {/* Audio Stats */}
        {isConnected && (
          <div style={{ 
            backgroundColor: '#111822', 
            borderRadius: '12px', 
            padding: '1.25rem'
          }}>
            <h2 style={{ color: '#00E5FF', fontSize: '1rem', marginBottom: '1rem' }}>
              Audio Transmission
            </h2>
            <StatusRow label="Bytes Sent" value={formatBytes(audioStats.bytesSent)} color="#E0E6ED" />
            <StatusRow label="Packets Sent" value={audioStats.packetsSent.toString()} color="#E0E6ED" />
          </div>
        )}

        <p style={{ color: '#8B95A5', fontSize: '0.75rem', marginTop: '2rem', textAlign: 'center' }}>
          VoiceShield PS-104 — WebRTC Audio Test Tool
          <br />
          Your microphone audio is sent directly to the VoiceShield Android app via WebRTC.
          <br />
          No audio is uploaded to any server.
        </p>
      </div>
    </div>
  );
}

function StatusRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '0.35rem 0'
    }}>
      <span style={{ color: '#8B95A5', fontSize: '0.85rem' }}>{label}</span>
      <span style={{ 
        color: color, fontSize: '0.85rem', fontWeight: 'bold',
        backgroundColor: `${color}20`, padding: '2px 8px', borderRadius: '4px'
      }}>
        {value}
      </span>
    </div>
  );
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(2)} MB`;
}
