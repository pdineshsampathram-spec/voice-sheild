"use client";

import { useCallback, useRef, useState } from "react";

interface UseAudioCaptureReturn {
  isRecording: boolean;
  startCapture: (onData: (data: ArrayBuffer) => void) => Promise<void>;
  stopCapture: () => void;
  error: string | null;
  /** Current waveform amplitude samples (0-1) for visualization */
  waveformData: Float32Array | null;
}

export function useAudioCapture(): UseAudioCaptureReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [waveformData, setWaveformData] = useState<Float32Array | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);
  const cleanupRef = useRef<(() => void) | null>(null);

  const startCapture = useCallback(
    async (onData: (data: ArrayBuffer) => void) => {
      try {
        setError(null);

        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
          },
        });

        mediaStreamRef.current = stream;

        const context = new AudioContext({ sampleRate: 16000 });
        contextRef.current = context;

        const source = context.createMediaStreamSource(stream);

        // Analyser node for waveform visualization
        const analyser = context.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;
        source.connect(analyser);

        // Try AudioWorklet first, fall back to ScriptProcessor
        let cleanup: () => void;

        try {
          await context.audioWorklet.addModule("/pcm-processor.js");
          const workletNode = new AudioWorkletNode(context, "pcm-processor");

          workletNode.port.onmessage = (e: MessageEvent) => {
            onData(e.data);
          };

          source.connect(workletNode);
          workletNode.connect(context.destination);

          cleanup = () => {
            workletNode.disconnect();
            source.disconnect();
          };
        } catch {
          // Fallback: ScriptProcessorNode (deprecated but widely supported)
          const processor = context.createScriptProcessor(4096, 1, 1);

          processor.onaudioprocess = (e: AudioProcessingEvent) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const pcm = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
              const s = Math.max(-1, Math.min(1, inputData[i]));
              pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }
            onData(pcm.buffer);
          };

          source.connect(processor);
          processor.connect(context.destination);

          cleanup = () => {
            processor.disconnect();
            source.disconnect();
          };
        }

        cleanupRef.current = cleanup;

        // Waveform animation loop
        const waveformBuffer = new Float32Array(analyser.frequencyBinCount);
        const updateWaveform = () => {
          analyser.getFloatTimeDomainData(waveformBuffer);
          setWaveformData(new Float32Array(waveformBuffer));
          animFrameRef.current = requestAnimationFrame(updateWaveform);
        };
        updateWaveform();

        setIsRecording(true);
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to access microphone";
        setError(msg);
        console.error("Audio capture error:", err);
      }
    },
    []
  );

  const stopCapture = useCallback(() => {
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (contextRef.current) {
      contextRef.current.close();
      contextRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }
    analyserRef.current = null;
    setIsRecording(false);
    setWaveformData(null);
  }, []);

  return { isRecording, startCapture, stopCapture, error, waveformData };
}
