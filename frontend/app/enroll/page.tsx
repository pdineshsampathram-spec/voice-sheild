"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { SpeakerProfile } from "@/types";
import { AudioPlayer } from "@/components/AudioPlayer";

interface SampleDeleteTarget {
  sampleId: string;
  speakerName: string;
  filename?: string;
}

export default function EnrollPage() {
  const [profiles, setProfiles] = useState<SpeakerProfile[]>([]);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [sampleToDelete, setSampleToDelete] = useState<SampleDeleteTarget | null>(null);
  const [profileToDelete, setProfileToDelete] = useState<SpeakerProfile | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const refreshProfiles = useCallback(async () => {
    try {
      const data = await api.getProfiles();
      setProfiles(data);
    } catch {
      // Ignored
    }
  }, []);

  useEffect(() => {
    refreshProfiles();
  }, [refreshProfiles]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const f = new File([blob], "enrollment.webm", { type: "audio/webm" });
        setFile(f);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch {
      setError("Failed to access microphone");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  const handleEnroll = async () => {
    if (!file || !name.trim()) {
      setError("Please provide a name and voice sample");
      return;
    }

    setIsEnrolling(true);
    setError(null);
    setMessage(null);

    try {
      const result = await api.enroll(file, name.trim());
      setMessage(`✓ ${result.message} (ID: ${result.speaker_id})`);
      setName("");
      setFile(null);
      await refreshProfiles();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Enrollment failed");
    } finally {
      setIsEnrolling(false);
    }
  };

  const handleDeleteSample = async () => {
    if (!sampleToDelete) return;
    setIsDeleting(true);
    setError(null);
    setMessage(null);

    try {
      await api.deleteSample(sampleToDelete.sampleId);
      setMessage(`✓ Successfully deleted recording from '${sampleToDelete.speakerName}'. Profile centroid was recomputed.`);
      setSampleToDelete(null);
      await refreshProfiles();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete sample");
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteProfile = async () => {
    if (!profileToDelete) return;
    setIsDeleting(true);
    setError(null);
    setMessage(null);

    try {
      await api.deleteProfile(profileToDelete.speaker_id);
      setMessage(`✓ Successfully deleted speaker profile '${profileToDelete.speaker_name}'.`);
      setProfileToDelete(null);
      await refreshProfiles();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete profile");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Speaker Enrollment</h1>
        <p className="text-gray-500 mt-1">Create a voice profile for speaker verification</p>
      </div>

      {/* Enrollment Form */}
      <div className="glass-card p-6 space-y-4">
        <div>
          <label className="text-xs text-gray-500 uppercase tracking-wider">Speaker Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter speaker name"
            className="w-full mt-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none"
          />
        </div>

        <div>
          <label className="text-xs text-gray-500 uppercase tracking-wider">Voice Sample</label>
          <div className="flex gap-3 mt-2">
            {/* Record */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                isRecording
                  ? "bg-red-600 hover:bg-red-500 text-white"
                  : "bg-gray-800 hover:bg-gray-700 text-gray-300"
              }`}
            >
              {isRecording ? "⏹ Stop Recording" : "🎤 Record Voice"}
            </button>

            {/* Upload */}
            <label className="flex-1 py-3 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 font-medium text-center cursor-pointer transition-colors">
              📁 Upload File
              <input
                type="file"
                accept=".wav,.mp3,.m4a,.webm"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </label>
          </div>
          {file && (
            <p className="text-sm text-cyan-400 mt-2">
              ✓ {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </p>
          )}
        </div>

        <button
          onClick={handleEnroll}
          disabled={isEnrolling || !file || !name.trim()}
          className="w-full py-3 rounded-lg bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold transition-all"
        >
          {isEnrolling ? "Enrolling..." : "Enroll Speaker"}
        </button>

        {message && <p className="text-emerald-400 text-sm font-mono">{message}</p>}
        {error && <p className="text-red-400 text-sm font-mono">⚠ {error}</p>}
      </div>

      {/* Existing Profiles */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-white">Enrolled Speakers</h2>
          <span className="text-xs text-gray-400 font-mono">
            Profiles: {profiles.length}
          </span>
        </div>

        {profiles.length > 0 ? (
          <div className="space-y-4">
            {profiles.map((p) => (
              <div key={p.speaker_id} className="p-4 rounded-xl bg-gray-900/40 border border-gray-800/60 space-y-3">
                <div className="flex items-start justify-between flex-wrap gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-base text-white font-bold break-words">{p.speaker_name}</span>
                      <span className={`text-[10px] px-2 py-0.5 rounded font-mono border ${
                        p.quality_status === "EMPTY"
                          ? "bg-amber-500/10 text-amber-300 border-amber-500/20"
                          : "bg-purple-500/10 text-purple-300 border-purple-500/20"
                      }`}>
                        {p.quality_status || "VALID"}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 font-mono truncate mt-0.5" title={p.speaker_id}>
                      ID: {p.speaker_id}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 flex-wrap text-right">
                    <span className="text-[10px] px-2 py-0.5 rounded bg-gray-800 text-cyan-400 font-mono border border-gray-700">
                      Samples: {p.sample_count || (p.samples ? p.samples.length : 0)}
                    </span>
                    <span className="text-xs text-gray-500 font-mono">
                      {new Date(p.created_at).toLocaleDateString()}
                    </span>
                    <button
                      type="button"
                      onClick={() => setProfileToDelete(p)}
                      className="px-2 py-1 rounded bg-red-950/30 hover:bg-red-900/50 border border-red-500/30 text-red-400 hover:text-red-300 text-xs font-mono transition-colors flex items-center gap-1"
                      title="Delete entire speaker profile"
                    >
                      <span>🗑</span> Delete Profile
                    </button>
                  </div>
                </div>

                {/* Enrollment Audio Samples Playback & Individual Deletion */}
                <div className="pt-2 border-t border-gray-800/40 space-y-2">
                  <span className="text-[11px] text-gray-400 font-semibold uppercase tracking-wider block">
                    Stored Voice Samples ({p.samples ? p.samples.length : 0})
                  </span>
                  {p.samples && p.samples.length > 0 ? (
                    <div className="space-y-2">
                      {p.samples.map((sample, idx) => (
                        <div key={sample.sample_id} className="flex items-center gap-2">
                          <div className="flex-1 min-w-0">
                            <AudioPlayer
                              title={`Sample ${idx + 1}`}
                              playbackUrl={sample.playback_url}
                              playbackAvailable={sample.playback_available}
                              originalFilename={sample.original_filename}
                              mimeType={sample.mime_type}
                              durationSeconds={sample.duration_seconds}
                              sha256={sample.sha256}
                              compact={true}
                              unavailableMessage="Enrollment sample unavailable"
                            />
                          </div>
                          <button
                            type="button"
                            onClick={() => setSampleToDelete({
                              sampleId: sample.sample_id,
                              speakerName: p.speaker_name,
                              filename: sample.original_filename,
                            })}
                            className="px-2.5 py-1.5 rounded-lg border border-red-500/30 bg-red-950/20 hover:bg-red-900/40 text-red-400 hover:text-red-300 text-xs font-mono transition-colors flex items-center gap-1 shrink-0"
                            title={`Delete sample ${sample.sample_id}`}
                          >
                            <span>🗑</span> Delete
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="p-3 rounded-lg bg-gray-950/60 border border-gray-800 text-xs font-mono space-y-1">
                      <span className="text-gray-300 font-semibold block">Legacy Profile</span>
                      <p className="text-gray-500">No enrollment recordings available for individual sample deletion.</p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-sm">No speakers enrolled yet.</p>
        )}
      </div>

      {/* Confirmation Modal: Delete Sample */}
      {sampleToDelete && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-card max-w-md w-full p-6 space-y-4 border border-gray-700 shadow-2xl">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="text-red-400">⚠</span> Delete enrollment sample?
            </h3>
            <p className="text-sm text-gray-300">
              This removes the stored voice recording (<span className="font-mono text-cyan-400">{sampleToDelete.filename || sampleToDelete.sampleId}</span>) from profile <strong className="text-white">{sampleToDelete.speakerName}</strong>.
            </p>
            <p className="text-xs text-gray-400">
              The speaker profile will be automatically recomputed from the remaining valid samples. If this was the last sample, the profile will be marked empty.
            </p>
            <div className="flex justify-end gap-3 pt-2 border-t border-gray-800">
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => setSampleToDelete(null)}
                className="px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={handleDeleteSample}
                className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white text-sm font-bold transition-colors"
              >
                {isDeleting ? "Deleting..." : "Delete Sample"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal: Delete Profile */}
      {profileToDelete && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-card max-w-md w-full p-6 space-y-4 border border-red-500/40 shadow-2xl">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="text-red-400">⚠</span> Delete speaker profile?
            </h3>
            <p className="text-sm text-gray-300">
              This permanently removes the enrolled speaker identity for <strong className="text-white">{profileToDelete.speaker_name}</strong> and all associated voice recordings.
            </p>
            <p className="text-xs text-red-400/80">
              Live Monitor and Analyze will no longer recognize or verify this speaker.
            </p>
            <div className="flex justify-end gap-3 pt-2 border-t border-gray-800">
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => setProfileToDelete(null)}
                className="px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={handleDeleteProfile}
                className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white text-sm font-bold transition-colors"
              >
                {isDeleting ? "Deleting..." : "Delete Profile"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
