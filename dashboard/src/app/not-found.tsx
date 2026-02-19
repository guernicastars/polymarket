import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
      <div className="text-6xl font-bold text-[#1e1e2e]">404</div>
      <h2 className="text-lg font-semibold">Page not found</h2>
      <p className="text-sm text-muted-foreground">
        The page you&apos;re looking for doesn&apos;t exist.
      </p>
      <Link
        href="/"
        className="mt-2 px-4 py-2 text-sm rounded-lg bg-[#00d4aa] text-[#0a0a0f] font-medium hover:bg-[#00d4aa]/90 transition-colors"
      >
        Back to dashboard
      </Link>
    </div>
  );
}
