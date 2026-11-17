// Use configured API URL if explicitly provided, otherwise fallback to same-origin relative URLs in browser
const getApiBase = () => {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== "undefined") return "";
  return "http://localhost:8000";
};

const getWsBase = () => {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}`;
  }
  return "ws://localhost:8000";
};

export const api = {
  get baseUrl() {
    return getApiBase();
  },
  get wsUrl() {
    return getWsBase();
  },

  async health() {
    const base = getApiBase();
    const res = await fetch(`${base}/api/health`);
    if (!res.ok) throw new Error("Health check failed");
    return res.json();
  },

  async analyzeUpload(file: File, speakerId?: string) {
    const formData = new FormData();
    formData.append("file", file);
    if (speakerId) formData.append("speaker_id", speakerId);

    const base = getApiBase();
    const res = await fetch(`${base}/api/analyze/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Analysis failed");
    }
    return res.json();
  },

  async enroll(file: File, speakerName: string) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("speaker_name", speakerName);

    const base = getApiBase();
    const res = await fetch(`${base}/api/enrollment`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Enrollment failed");
    }
    return res.json();
  },

  async getProfiles() {
    const base = getApiBase();
    const res = await fetch(`${base}/api/enrollment/profiles`);
    if (!res.ok) throw new Error("Failed to fetch profiles");
    return res.json();
  },

  async deleteProfile(profileId: string) {
    const base = getApiBase();
    const res = await fetch(`${base}/api/enrollment/${encodeURIComponent(profileId)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to delete profile");
    }
    return res.json();
  },

  async deleteSample(sampleId: string) {
    const base = getApiBase();
    const res = await fetch(`${base}/api/enrollment/samples/${encodeURIComponent(sampleId)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to delete enrollment sample");
    }
    return res.json();
  },

  async getRecentAlerts(limit = 10) {
    const base = getApiBase();
    const res = await fetch(`${base}/api/alerts/recent?limit=${limit}`);
    if (!res.ok) return { alerts: [] };
    return res.json();
  },

  async getSessions(limit = 20) {
    const base = getApiBase();
    const res = await fetch(`${base}/api/sessions?limit=${limit}`);
    if (!res.ok) return [];
    return res.json();
  },
};
