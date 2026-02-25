import { NextResponse } from "next/server";
import { query } from "@/lib/clickhouse";

export const dynamic = "force-dynamic";

export async function GET() {
  const results: Record<string, unknown> = {};

  // Check env vars exist (don't expose values)
  results.env = {
    CLICKHOUSE_URL: !!process.env.CLICKHOUSE_URL,
    CLICKHOUSE_USER: !!process.env.CLICKHOUSE_USER,
    CLICKHOUSE_PASSWORD: !!process.env.CLICKHOUSE_PASSWORD,
    CLICKHOUSE_DB: !!process.env.CLICKHOUSE_DB,
    CLICKHOUSE_URL_preview: process.env.CLICKHOUSE_URL?.substring(0, 30) + "...",
  };

  // Test basic query
  try {
    const start = Date.now();
    const rows = await query<{ cnt: number }>("SELECT count() AS cnt FROM markets FINAL");
    results.basicQuery = {
      ok: true,
      rows: rows[0]?.cnt,
      ms: Date.now() - start,
    };
  } catch (e: unknown) {
    const err = e as Error;
    results.basicQuery = {
      ok: false,
      error: err.message,
      stack: err.stack?.substring(0, 500),
    };
  }

  // Test signals overview query
  try {
    const start = Date.now();
    const rows = await query<Record<string, number>>(
      `SELECT countIf(active = 1 AND closed = 0) AS active_markets FROM markets FINAL`
    );
    results.signalsQuery = {
      ok: true,
      rows: rows[0],
      ms: Date.now() - start,
    };
  } catch (e: unknown) {
    const err = e as Error;
    results.signalsQuery = {
      ok: false,
      error: err.message,
    };
  }

  return NextResponse.json(results);
}
