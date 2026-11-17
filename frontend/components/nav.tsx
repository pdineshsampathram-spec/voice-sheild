"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";

export function ActiveNavLink({ href, label }: { href: string; label: string }) {
  const pathname = usePathname();
  const isActive = href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <Link
      href={href}
      className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
        isActive
          ? "text-white bg-gray-800/70 border border-gray-700/50"
          : "text-gray-400 hover:text-white hover:bg-gray-800/50"
      }`}
    >
      {label}
    </Link>
  );
}
