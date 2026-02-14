import { createClient } from "@clickhouse/client";

const client = createClient({
  url: process.env.CLICKHOUSE_URL ?? "http://localhost:8123",
  username: process.env.CLICKHOUSE_USER ?? "default",
  password: process.env.CLICKHOUSE_PASSWORD ?? "",
  database: process.env.CLICKHOUSE_DB ?? "polymarket",
  request_timeout: 30_000,
  clickhouse_settings: {
    output_format_json_quote_64bit_integers: 0,
  },
});

export async function query<T>(
  sql: string,
  params?: Record<string, unknown>
): Promise<T[]> {
  const result = await client.query({
    query: sql,
    query_params: params,
    format: "JSONEachRow",
  });
  return result.json() as Promise<T[]>;
}

export default client;
