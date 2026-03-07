import { createClient } from "@clickhouse/client";

const clickhouseUrl = process.env.CLICKHOUSE_URL ?? "http://localhost:8123";

const client = createClient({
  url: clickhouseUrl,
  username: process.env.CLICKHOUSE_USER ?? "default",
  password: process.env.CLICKHOUSE_PASSWORD ?? "",
  database: process.env.CLICKHOUSE_DB ?? "polymarket",
  request_timeout: 30_000,
  clickhouse_settings: {
    output_format_json_quote_64bit_integers: 0,
  },
});

if (!process.env.CLICKHOUSE_URL) {
  console.warn(
    "[ClickHouse] CLICKHOUSE_URL not set, defaulting to http://localhost:8123. Set CLICKHOUSE_URL in .env.local for production."
  );
}

export async function query<T>(
  sql: string,
  params?: Record<string, unknown>
): Promise<T[]> {
  try {
    const result = await client.query({
      query: sql,
      query_params: params,
      format: "JSONEachRow",
    });
    return result.json() as Promise<T[]>;
  } catch (error) {
    const queryPreview = sql.trim().substring(0, 80).replace(/\s+/g, " ");
    console.error(
      `[ClickHouse] Query failed (${clickhouseUrl}): "${queryPreview}..."`,
      error instanceof Error ? error.message : error
    );
    throw error;
  }
}

export default client;
