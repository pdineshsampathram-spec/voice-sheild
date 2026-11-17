import { NextResponse } from "next/server";

export async function GET() {
  const targetBackend = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL;
  if (!targetBackend || !targetBackend.startsWith("http")) {
    return NextResponse.json({
      available: false,
      reason: "No backend API URL configured (Standalone Mode)",
    });
  }

  try {
    const res = await fetch(`${targetBackend}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json({
        available: true,
        data,
      });
    }
    return NextResponse.json({
      available: false,
      reason: `Backend returned HTTP ${res.status}`,
    });
  } catch {
    return NextResponse.json({
      available: false,
      reason: "Backend connection timed out or unreachable",
    });
  }
}
