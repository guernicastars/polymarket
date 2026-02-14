import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import {
  LayoutDashboard,
  TrendingUp,
  BarChart3,
  Zap,
} from "lucide-react";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Polymarket Signals",
  description: "Real-time Polymarket prediction market analytics dashboard",
};

const navItems = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/#markets", label: "Markets", icon: BarChart3 },
  { href: "/#trending", label: "Trending", icon: TrendingUp },
  { href: "/#movers", label: "Signals", icon: Zap },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <div className="min-h-screen flex">
          {/* Sidebar */}
          <aside className="hidden md:flex w-56 flex-col fixed inset-y-0 z-50 bg-[#0d0d14] border-r border-[#1e1e2e]">
            <div className="flex h-14 items-center px-6 border-b border-[#1e1e2e]">
              <Link href="/" className="flex items-center gap-2">
                <div className="h-6 w-6 rounded bg-[#00d4aa] flex items-center justify-center">
                  <Zap className="h-3.5 w-3.5 text-[#0a0a0f]" />
                </div>
                <span className="font-semibold text-sm tracking-tight">
                  Polymarket Signals
                </span>
              </Link>
            </div>
            <nav className="flex-1 px-3 py-4 space-y-1">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center gap-3 px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-[#1e1e2e] rounded-lg transition-colors"
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              ))}
            </nav>
            <div className="px-6 py-4 border-t border-[#1e1e2e]">
              <p className="text-xs text-muted-foreground">
                Data refreshes every 10s
              </p>
            </div>
          </aside>

          {/* Main content */}
          <div className="flex-1 md:pl-56">
            {/* Mobile header */}
            <header className="sticky top-0 z-40 md:hidden flex h-14 items-center px-4 border-b border-[#1e1e2e] bg-[#0a0a0f]/95 backdrop-blur">
              <Link href="/" className="flex items-center gap-2">
                <div className="h-6 w-6 rounded bg-[#00d4aa] flex items-center justify-center">
                  <Zap className="h-3.5 w-3.5 text-[#0a0a0f]" />
                </div>
                <span className="font-semibold text-sm">Polymarket Signals</span>
              </Link>
            </header>

            <main className="p-4 md:p-6 lg:p-8">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
