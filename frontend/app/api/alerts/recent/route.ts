import { NextResponse } from "next/server";

export async function GET() {
  const targetBackend = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL;
  if (targetBackend && targetBackend.startsWith("http")) {
    try {
      const res = await fetch(`${targetBackend}/api/alerts/recent?limit=10`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return NextResponse.json(await res.json());
    } catch {
      // Fallback
    }
  }

  return NextResponse.json({
    alerts: [
      {
        alert_id: "alt_demo_01",
        timestamp: new Date(Date.now() - 120000).toLocaleTimeString(),
        level: "CRITICAL",
        title: "Synthetic Voice Clone Detected",
        message: "High-probability cloned voice detected (spectral artifacts in high frequencies).",
        reason: "Spectra-AASIST logit exceeded critical threshold (0.85)",
        recommended_action: "BLOCK",
        spoof_score: 0.94,
      },
      {
        alert_id: "alt_demo_02",
        timestamp: new Date(Date.now() - 3600000).toLocaleTimeString(),
        level: "MEDIUM",
        title: "Speaker Identity Discrepancy",
        message: "Voice biometric discrepancy detected against enrolled profile.",
        reason: "ECAPA-TDNN cosine similarity below match threshold (0.25)",
        recommended_action: "VERIFY",
        spoof_score: 0.52,
        speaker_similarity: 0.18,
      },
    ],
  });
}
