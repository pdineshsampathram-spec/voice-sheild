"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

function generateRoomId(): string {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let result = "";
  for (let i = 0; i < 6; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return `VS-${result}`;
}

export default function CallLobbyPage() {
  const router = useRouter();
  const [customRoomId, setCustomRoomId] = useState("");
  const [createdRoom, setCreatedRoom] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleCreateCall = () => {
    const newRoom = generateRoomId();
    setCreatedRoom(newRoom);
  };

  const handleCopyLink = () => {
    if (!createdRoom) return;
    const url = `${window.location.origin}/call/${createdRoom}`;
    navigator.clipboard.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handleJoinCustom = (e: React.FormEvent) => {
    e.preventDefault();
    if (customRoomId.trim()) {
      router.push(`/call/${customRoomId.trim().toUpperCase()}`);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-4">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-500 bg-clip-text text-transparent">
          Two-Way WebRTC Call Lab
        </h1>
        <p className="text-sm text-gray-400 mt-2">
          Real-time peer-to-peer audio calls with independent VoiceShield observer monitoring.
        </p>
      </div>

      {/* Main Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Create Call Card */}
        <div className="rounded-2xl border border-cyan-500/20 bg-gradient-to-b from-gray-900/80 to-gray-950/80 p-6 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-2xl pointer-events-none" />
          <h2 className="text-xl font-semibold text-white mb-2">Create a Call Room</h2>
          <p className="text-xs text-gray-400 mb-6">
            Generate an isolated WebRTC room and invite a second participant to establish bidirectional voice audio.
          </p>

          {!createdRoom ? (
            <button
              onClick={handleCreateCall}
              className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium shadow-lg shadow-cyan-900/30 transition-all cursor-pointer"
            >
              Generate Call Room
            </button>
          ) : (
            <div className="space-y-4">
              <div className="p-3 bg-gray-950/80 rounded-xl border border-gray-800">
                <span className="text-xs text-gray-500 block mb-1">Room ID</span>
                <span className="text-lg font-mono text-cyan-400 font-bold">{createdRoom}</span>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleCopyLink}
                  className="flex-1 py-2.5 px-3 rounded-lg bg-gray-800 hover:bg-gray-700 text-xs font-medium text-gray-200 border border-gray-700 transition-all cursor-pointer flex items-center justify-center gap-1.5"
                >
                  {copied ? "✓ Copied Link" : "Copy Invite Link"}
                </button>
                <Link
                  href={`/call/${createdRoom}`}
                  className="flex-1 py-2.5 px-3 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 text-white text-xs font-medium text-center shadow transition-all cursor-pointer flex items-center justify-center"
                >
                  Enter Room →
                </Link>
              </div>
              <p className="text-xs text-gray-500 text-center">
                Share this link with Participant B before or after entering.
              </p>
            </div>
          )}
        </div>

        {/* Join Call Card */}
        <div className="rounded-2xl border border-gray-800/80 bg-gray-900/40 p-6 shadow-xl flex flex-col justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white mb-2">Join an Existing Room</h2>
            <p className="text-xs text-gray-400 mb-6">
              Enter a room ID provided by the host to connect directly into the ongoing WebRTC call.
            </p>

            <form onSubmit={handleJoinCustom} className="space-y-3">
              <input
                type="text"
                placeholder="e.g. VS-K7M9P2"
                value={customRoomId}
                onChange={(e) => setCustomRoomId(e.target.value)}
                className="w-full px-4 py-2.5 bg-gray-950 border border-gray-800 rounded-xl text-white placeholder-gray-600 text-sm font-mono focus:outline-none focus:border-cyan-500 transition-all"
              />
              <button
                type="submit"
                disabled={!customRoomId.trim()}
                className="w-full py-3 px-4 rounded-xl bg-gray-800 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-all cursor-pointer border border-gray-700"
              >
                Join Room
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* Controlled Experiment Protocols */}
      <div className="rounded-2xl border border-gray-800/60 bg-gray-900/30 p-6">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
          Controlled Experiment Protocols (Rule 13)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-gray-400">
          <div className="p-4 rounded-xl bg-gray-950/40 border border-gray-800/50">
            <span className="text-cyan-400 font-semibold block mb-1">MODE A: Human ↔ Human</span>
            <p>Peer A speaks to Peer B. Establishes the real-world baseline for WebRTC conversational acoustics.</p>
          </div>
          <div className="p-4 rounded-xl bg-gray-950/40 border border-gray-800/50">
            <span className="text-indigo-400 font-semibold block mb-1">MODE B: Human ↔ Synthetic</span>
            <p>One participant plays back synthetic/cloned audio. Verifies detection on real over-the-wire Opus streams.</p>
          </div>
          <div className="p-4 rounded-xl bg-gray-950/40 border border-gray-800/50">
            <span className="text-amber-400 font-semibold block mb-1">MODE C: Codec Degraded</span>
            <p>Participants test with background noise, packet loss, or varying bandwidth to test calibration robustness.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
