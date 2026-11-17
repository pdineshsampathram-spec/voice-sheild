"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

export type PlayerState =
  | "AVAILABLE"
  | "LOADING"
  | "PLAYING"
  | "PAUSED"
  | "ENDED"
  | "UNAVAILABLE"
  | "ERROR";

interface AudioPlayerProps {
  playbackUrl?: string | null;
  playbackAvailable?: boolean;
  title?: string;
  originalFilename?: string;
  mimeType?: string;
  sha256?: string;
  durationSeconds?: number;
  compact?: boolean;
  unavailableMessage?: string;
  className?: string;
}

function formatTime(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export function AudioPlayer({
  playbackUrl,
  playbackAvailable = true,
  title = "Playback",
  originalFilename,
  mimeType,
  sha256,
  durationSeconds,
  compact = false,
  unavailableMessage = "RECORDING NO LONGER AVAILABLE",
  className = "",
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [state, setState] = useState<PlayerState>(
    playbackAvailable && playbackUrl ? "AVAILABLE" : "UNAVAILABLE"
  );
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(durationSeconds || 0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fullUrl = playbackUrl
    ? playbackUrl.startsWith("http")
      ? playbackUrl
      : `${api.baseUrl}${playbackUrl.startsWith("/") ? "" : "/"}${playbackUrl}`
    : null;

  useEffect(() => {
    if (!playbackAvailable || !playbackUrl) {
      setState("UNAVAILABLE");
      return;
    }
    setState("AVAILABLE");
    setErrorMessage(null);
    setCurrentTime(0);
    if (durationSeconds) setDuration(durationSeconds);
  }, [playbackUrl, playbackAvailable, durationSeconds]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      if (audio.duration && !isNaN(audio.duration) && isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
      if (state === "LOADING") setState("AVAILABLE");
    };

    const handleCanPlay = () => {
      if (state === "LOADING") setState("AVAILABLE");
    };

    const handlePlay = () => setState("PLAYING");
    const handlePause = () => {
      if (state !== "ENDED" && state !== "ERROR") {
        setState("PAUSED");
      }
    };
    const handleEnded = () => {
      setState("ENDED");
      setCurrentTime(audio.duration || 0);
    };
    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
      if (audio.duration && !isNaN(audio.duration) && isFinite(audio.duration) && duration === 0) {
        setDuration(audio.duration);
      }
    };
    const handleWaiting = () => setState("LOADING");
    const handleStalled = () => {
      if (state === "PLAYING") setState("LOADING");
    };
    const handleError = () => {
      setState("ERROR");
      setErrorMessage("Audio playback failed or artifact is missing (404)");
    };

    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("canplay", handleCanPlay);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("waiting", handleWaiting);
    audio.addEventListener("stalled", handleStalled);
    audio.addEventListener("error", handleError);

    return () => {
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("canplay", handleCanPlay);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("waiting", handleWaiting);
      audio.removeEventListener("stalled", handleStalled);
      audio.removeEventListener("error", handleError);
    };
  }, [state, duration]);

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (state === "PLAYING") {
      audio.pause();
    } else {
      try {
        setState("LOADING");
        await audio.play();
      } catch (err) {
        setState("ERROR");
        setErrorMessage(err instanceof Error ? err.message : "Playback blocked or failed");
      }
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const targetTime = parseFloat(e.target.value);
    setCurrentTime(targetTime);
    if (audioRef.current) {
      audioRef.current.currentTime = targetTime;
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    setVolume(v);
    setIsMuted(v === 0);
    if (audioRef.current) {
      audioRef.current.volume = v;
      audioRef.current.muted = v === 0;
    }
  };

  const toggleMute = () => {
    const nextMute = !isMuted;
    setIsMuted(nextMute);
    if (audioRef.current) {
      audioRef.current.muted = nextMute;
    }
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackRate(speed);
    if (audioRef.current) {
      audioRef.current.playbackRate = speed;
    }
  };

  if (state === "UNAVAILABLE" || !fullUrl) {
    return (
      <div
        className={`glass-card p-4 border border-red-500/20 bg-red-500/5 rounded-lg flex items-center justify-between flex-wrap gap-2 ${className}`}
      >
        <div className="flex items-center gap-3">
          <span className="w-8 h-8 rounded-full bg-red-500/20 text-red-400 flex items-center justify-center font-bold text-sm">
            ✕
          </span>
          <div>
            <div className="text-xs font-bold text-red-400 uppercase tracking-wider">
              {unavailableMessage}
            </div>
            {originalFilename && (
              <div className="text-[11px] text-gray-500 font-mono break-all">
                {originalFilename}
              </div>
            )}
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-gray-800 text-gray-400 border border-gray-700">
          UNAVAILABLE
        </span>
      </div>
    );
  }

  // Compact layout (used in enrollment sample rows)
  if (compact) {
    return (
      <div
        className={`flex items-center justify-between p-2.5 rounded-lg bg-gray-900/60 border border-gray-800/80 flex-wrap gap-2 ${className}`}
      >
        <audio ref={audioRef} src={fullUrl} preload="metadata" />

        <div className="flex items-center gap-3 min-w-0 flex-1">
          <button
            onClick={togglePlay}
            disabled={state === "ERROR"}
            className="w-8 h-8 rounded-full bg-cyan-600 hover:bg-cyan-500 text-white flex items-center justify-center text-xs font-bold transition-colors flex-shrink-0 disabled:opacity-50"
            title={state === "PLAYING" ? "Pause" : "Play"}
          >
            {state === "LOADING" ? (
              <span className="animate-spin text-xs">⟳</span>
            ) : state === "PLAYING" ? (
              "⏸"
            ) : (
              "▶"
            )}
          </button>

          <div className="min-w-0 flex-1">
            <div className="text-xs text-gray-200 font-medium truncate break-all">
              {originalFilename || title}
            </div>
            <div className="flex items-center gap-2 text-[10px] text-gray-500 font-mono mt-0.5">
              <span>{formatTime(currentTime)} / {formatTime(duration)}</span>
              {durationSeconds !== undefined && <span>({durationSeconds.toFixed(1)}s)</span>}
              {mimeType && <span className="uppercase text-gray-600">{mimeType.split("/")[1] || mimeType}</span>}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Seek slider */}
          <input
            type="range"
            min="0"
            max={duration || 100}
            step="0.05"
            value={currentTime}
            onChange={handleSeek}
            className="w-24 h-1 bg-gray-800 accent-cyan-500 rounded cursor-pointer"
          />

          <span
            className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded ${
              state === "PLAYING"
                ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                : state === "ERROR"
                ? "bg-red-500/20 text-red-400"
                : "bg-gray-800 text-gray-400"
            }`}
          >
            {state}
          </span>
        </div>
      </div>
    );
  }

  // Full Featured Layout (used on Analyze page)
  return (
    <div className={`glass-card p-5 border border-cyan-500/30 bg-gradient-to-r from-cyan-950/20 via-gray-900/60 to-gray-900/80 rounded-xl space-y-4 ${className}`}>
      <audio ref={audioRef} src={fullUrl} preload="metadata" />

      {/* Header with Title & State Badge */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-base">▶</span>
          <span className="text-xs text-cyan-400 font-bold uppercase tracking-wider">
            {title}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded ${
              state === "PLAYING"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 animate-pulse"
                : state === "ERROR"
                ? "bg-red-500/20 text-red-400 border border-red-500/40"
                : state === "LOADING"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                : "bg-gray-800 text-gray-300 border border-gray-700"
            }`}
          >
            {state}
          </span>
        </div>
      </div>

      {/* Audio Artifact Metadata Details */}
      {(originalFilename || sha256 || mimeType) && (
        <div className="p-3 rounded-lg bg-gray-950/70 border border-gray-800/80 grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
          {originalFilename && (
            <div className="min-w-0">
              <span className="text-[10px] text-gray-500 uppercase tracking-wider block">Recording</span>
              <span className="text-gray-200 font-mono font-medium truncate block" title={originalFilename}>
                {originalFilename}
              </span>
            </div>
          )}
          {mimeType && (
            <div>
              <span className="text-[10px] text-gray-500 uppercase tracking-wider block">Format</span>
              <span className="text-cyan-400 font-mono font-semibold">
                {mimeType}
              </span>
            </div>
          )}
          {sha256 && (
            <div className="min-w-0">
              <span className="text-[10px] text-gray-500 uppercase tracking-wider block">SHA-256 Audit Hash</span>
              <span className="text-gray-400 font-mono text-[11px] truncate block" title={sha256}>
                {sha256.slice(0, 16)}...{sha256.slice(-8)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Timeline scrubber bar */}
      <div className="space-y-1.5">
        <input
          type="range"
          min="0"
          max={duration || 100}
          step="0.05"
          value={currentTime}
          onChange={handleSeek}
          className="w-full h-2 bg-gray-800 accent-cyan-500 rounded-lg cursor-pointer transition-all"
        />
        <div className="flex justify-between text-xs text-gray-400 font-mono">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Controls Bar: Play/Pause, Volume, Speed */}
      <div className="flex items-center justify-between flex-wrap gap-4 pt-1">
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            disabled={state === "ERROR"}
            className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-bold text-xs uppercase tracking-wider flex items-center gap-2 transition-colors shadow-lg shadow-cyan-900/30"
          >
            {state === "LOADING" ? (
              <>
                <span className="animate-spin">⟳</span> Loading...
              </>
            ) : state === "PLAYING" ? (
              <>
                <span>⏸</span> Pause
              </>
            ) : (
              <>
                <span>▶</span> Play Recording
              </>
            )}
          </button>

          {/* Volume Control */}
          <div className="flex items-center gap-2 bg-gray-900/80 px-2.5 py-1.5 rounded-lg border border-gray-800">
            <button
              onClick={toggleMute}
              className="text-xs text-gray-400 hover:text-white transition-colors"
              title={isMuted ? "Unmute" : "Mute"}
            >
              {isMuted || volume === 0 ? "🔇" : volume < 0.5 ? "🔉" : "🔊"}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              className="w-16 h-1 bg-gray-800 accent-cyan-500 rounded cursor-pointer"
            />
          </div>
        </div>

        {/* Playback Rate Selector */}
        <div className="flex items-center gap-1.5 bg-gray-900/80 p-1 rounded-lg border border-gray-800">
          <span className="text-[10px] text-gray-500 uppercase px-1.5 font-semibold">Speed</span>
          {[0.75, 1, 1.25, 1.5].map((speed) => (
            <button
              key={speed}
              onClick={() => handleSpeedChange(speed)}
              className={`text-xs px-2 py-0.5 rounded font-mono font-medium transition-colors ${
                playbackRate === speed
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>

      {errorMessage && (
        <div className="p-2 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
          ⚠ {errorMessage}
        </div>
      )}
    </div>
  );
}
