"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { WebSocketMessage, WindowResult, AlertPayload } from "@/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_BASE_DELAY_MS = 1000;

interface UseWebSocketReturn {
  isConnected: boolean;
  sessionId: string | null;
  lastResult: WindowResult | null;
  lastAlert: AlertPayload | null;
  results: WindowResult[];
  alerts: AlertPayload[];
  challenge: { phrase: string; challenge_id: string } | null;
  start: (speakerId?: string) => void;
  stop: () => void;
  dismiss: () => void;
  block: () => void;
  requestChallenge: () => void;
  sendAudio: (data: ArrayBuffer) => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<WindowResult | null>(null);
  const [lastAlert, setLastAlert] = useState<AlertPayload | null>(null);
  const [results, setResults] = useState<WindowResult[]>([]);
  const [alerts, setAlerts] = useState<AlertPayload[]>([]);
  const [challenge, setChallenge] = useState<{
    phrase: string;
    challenge_id: string;
  } | null>(null);

  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intentionalCloseRef = useRef(false);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    intentionalCloseRef.current = false;
    const ws = new WebSocket(`${WS_URL}/ws/voice-analysis`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttemptRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const msg: WebSocketMessage = JSON.parse(event.data);

        switch (msg.type) {
          case "connected":
            setSessionId(msg.session_id || null);
            break;

          case "session_started":
            break;

          case "analysis_result":
            if (msg.data) {
              setLastResult(msg.data);
              setResults((prev) => [...prev.slice(-99), msg.data!]);
            }
            if (msg.alert) {
              setLastAlert(msg.alert);
              setAlerts((prev) => [...prev.slice(-49), msg.alert!]);
            }
            break;

          case "challenge":
            if (msg.data) {
              setChallenge({
                phrase: (msg.data as unknown as { phrase: string }).phrase,
                challenge_id: (
                  msg.data as unknown as { challenge_id: string }
                ).challenge_id,
              });
            }
            break;

          case "challenge_result":
            setChallenge(null);
            break;

          case "session_stopped":
          case "session_blocked":
            break;

          case "alert_dismissed":
            setLastAlert(null);
            break;
        }
      } catch {
        // Non-JSON message, ignore
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      setSessionId(null);

      // Auto-reconnect with exponential backoff (unless intentionally closed)
      if (
        !intentionalCloseRef.current &&
        reconnectAttemptRef.current < MAX_RECONNECT_ATTEMPTS
      ) {
        const delay =
          RECONNECT_BASE_DELAY_MS *
          Math.pow(2, reconnectAttemptRef.current);
        reconnectAttemptRef.current += 1;
        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, delay);
      }
    };

    ws.onerror = () => {
      setIsConnected(false);
    };

    wsRef.current = ws;
  }, []);

  const sendCommand = useCallback(
    (action: string, data?: Record<string, unknown>) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action, ...data }));
      }
    },
    []
  );

  const start = useCallback(
    (speakerId?: string) => {
      connect();
      // Wait for connection then start
      const checkAndStart = () => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          sendCommand(
            "start",
            speakerId ? { speaker_id: speakerId } : undefined
          );
          setResults([]);
          setAlerts([]);
          setLastResult(null);
          setLastAlert(null);
          setChallenge(null);
        } else {
          setTimeout(checkAndStart, 100);
        }
      };
      checkAndStart();
    },
    [connect, sendCommand]
  );

  const stop = useCallback(() => {
    intentionalCloseRef.current = true;
    sendCommand("stop");
  }, [sendCommand]);

  const dismiss = useCallback(() => {
    sendCommand("dismiss");
    setLastAlert(null);
  }, [sendCommand]);

  const block = useCallback(() => {
    intentionalCloseRef.current = true;
    sendCommand("block");
  }, [sendCommand]);

  const requestChallenge = useCallback(() => {
    sendCommand("verify_challenge");
  }, [sendCommand]);

  const sendAudio = useCallback((data: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      intentionalCloseRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      wsRef.current?.close();
    };
  }, []);

  return {
    isConnected,
    sessionId,
    lastResult,
    lastAlert,
    results,
    alerts,
    challenge,
    start,
    stop,
    dismiss,
    block,
    requestChallenge,
    sendAudio,
  };
}
