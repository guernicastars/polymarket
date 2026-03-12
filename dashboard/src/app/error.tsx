"use client";

import { useEffect, useState } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    console.error("Dashboard error:", error);
  }, [error]);

  const errorMessage = error.message || "Unknown error";
  const isConnectionError =
    errorMessage.includes("ECONNREFUSED") ||
    errorMessage.includes("ETIMEDOUT") ||
    errorMessage.includes("fetch failed") ||
    errorMessage.includes("connect");
  const isAuthError =
    errorMessage.includes("Authentication") ||
    errorMessage.includes("401") ||
    errorMessage.includes("403");

  const hint = isConnectionError
    ? "The ClickHouse database appears to be unreachable. Check that CLICKHOUSE_URL is correct and the server is running."
    : isAuthError
      ? "Authentication failed. Check CLICKHOUSE_USER and CLICKHOUSE_PASSWORD in your environment variables."
      : "This usually means the ClickHouse database is unavailable or waking up. Check /api/debug for diagnostics.";

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
      <div className="h-12 w-12 rounded-lg bg-[#ff4466]/10 flex items-center justify-center">
        <span className="text-[#ff4466] text-xl">!</span>
      </div>
      <h2 className="text-lg font-semibold">Something went wrong</h2>
      <p className="text-sm text-muted-foreground max-w-md text-center">
        {hint}
      </p>
      <div className="flex gap-3 mt-2">
        <button
          onClick={reset}
          className="px-4 py-2 text-sm rounded-lg bg-[#00d4aa] text-[#0a0a0f] font-medium hover:bg-[#00d4aa]/90 transition-colors"
        >
          Try again
        </button>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="px-4 py-2 text-sm rounded-lg border border-[#1e1e2e] text-muted-foreground hover:text-white transition-colors"
        >
          {showDetails ? "Hide details" : "Show details"}
        </button>
      </div>
      {showDetails && (
        <pre className="mt-2 p-4 bg-[#111118] border border-[#1e1e2e] rounded-lg text-xs text-muted-foreground max-w-lg overflow-auto max-h-40 w-full">
          {errorMessage}
          {error.digest && `\nDigest: ${error.digest}`}
        </pre>
      )}
    </div>
  );
}
