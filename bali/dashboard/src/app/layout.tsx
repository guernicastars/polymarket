import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bali Risk Intelligence",
  description: "Real estate risk assessment platform for Bali investors",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <nav className="sidebar">
          <h1>Bali Risk Intel</h1>
          <Link href="/">Overview</Link>
          <Link href="/districts">Districts</Link>
          <Link href="/seismic">Seismic</Link>
          <Link href="/legal">Legal Guide</Link>
          <Link href="/market">Market Data</Link>
        </nav>
        <main className="main">{children}</main>
      </body>
    </html>
  );
}
