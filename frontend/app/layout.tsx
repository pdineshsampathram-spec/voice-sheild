import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ActiveNavLink } from "@/components/nav";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "VoiceShield — Voice Cloning Detection",
  description:
    "AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-950 text-gray-100 min-h-screen`}>
        <nav className="border-b border-gray-800/50 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                  VoiceShield
                </span>
              </div>
              <div className="flex items-center gap-1">
                <ActiveNavLink href="/" label="Dashboard" />
                <ActiveNavLink href="/live" label="Live Protection" />
                <ActiveNavLink href="/monitor" label="Monitor" />
                <ActiveNavLink href="/call" label="Call Room" />
                <ActiveNavLink href="/analyze" label="Analyze" />
                <ActiveNavLink href="/enroll" label="Enroll" />
                <ActiveNavLink href="/history" label="History" />
              </div>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
