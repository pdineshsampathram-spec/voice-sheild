const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export const api = {
  baseUrl: API_BASE,
  wsUrl: WS_BASE,

  async health() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  },

  async analyzeUpload(file: File, speakerId?: string) {
    const formData = new FormData();
    formData.append("file", file);
    if (speakerId) formData.append("speaker_id", speakerId);

    const res = await fetch(`${API_BASE}/api/analyze/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis failed");
    }
    return res.json();
  },

  async enroll(file: File, speakerName: string) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("speaker_name", speakerName);

    const res = await fetch(`${API_BASE}/api/enrollment`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Enrollment failed");
    }
    return res.json();
  },

  async getProfiles() {
    const res = await fetch(`${API_BASE}/api/enrollment/profiles`);
    return res.json();
  },

  async deleteSample(sampleId: string) {
    const res = await fetch(`${API_BASE}/api/enrollment/samples/${encodeURIComponent(sampleId)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to delete enrollment sample");
    }
    return res.json();
  },

  async deleteProfile(speakerId: string) {
    const res = await fetch(`${API_BASE}/api/enrollment/${encodeURIComponent(speakerId)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to delete profile");
    }
    return res.json();
  },

  async getSessions(limit = 50) {
    const res = await fetch(`${API_BASE}/api/sessions?limit=${limit}`);
    return res.json();
  },

  async getRecentAlerts(limit = 20) {
    const res = await fetch(`${API_BASE}/api/alerts/recent?limit=${limit}`);
    return res.json();
  },
};
