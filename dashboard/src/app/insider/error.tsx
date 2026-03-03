"use client";

import { useEffect } from "react";

export default function InsiderError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Insider page error:", error);
  }, [error]);

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
      <div className="h-12 w-12 rounded-lg bg-[#ff4466]/10 flex items-center justify-center">
        <span className="text-[#ff4466] text-xl">!</span>
      </div>
      <h2 className="text-lg font-semibold">
        Insider Detection Unavailable
      </h2>
      <p className="text-sm text-muted-foreground max-w-md text-center">
        Could not load insider trading data. The ClickHouse database may be
        temporarily unavailable or under heavy load.
      </p>
      <button
        onClick={reset}
        className="mt-2 px-4 py-2 text-sm rounded-lg bg-[#00d4aa] text-[#0a0a0f] font-medium hover:bg-[#00d4aa]/90 transition-colors"
      >
        Try again
      </button>
    </div>
  );
}
