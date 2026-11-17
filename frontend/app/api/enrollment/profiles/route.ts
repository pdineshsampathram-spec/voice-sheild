import { NextResponse } from "next/server";

export async function GET() {
  const targetBackend = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL;
  if (targetBackend && targetBackend.startsWith("http")) {
    try {
      const res = await fetch(`${targetBackend}/api/enrollment/profiles`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return NextResponse.json(await res.json());
    } catch {
      // Fallback
    }
  }

  return NextResponse.json({
    profiles: [
      {
        speaker_id: "spk_trusted_exec_01",
        speaker_name: "CEO Executive Voice Profile",
        created_at: new Date(Date.now() - 86400000 * 3).toISOString(),
        embedding_path: "profiles/spk_trusted_exec_01.npy",
        sample_count: 3,
        quality_status: "OPTIMAL",
      },
      {
        speaker_id: "spk_finance_officer",
        speaker_name: "Chief Financial Officer",
        created_at: new Date(Date.now() - 86400000 * 7).toISOString(),
        embedding_path: "profiles/spk_finance_officer.npy",
        sample_count: 2,
        quality_status: "OPTIMAL",
      },
    ],
  });
}
