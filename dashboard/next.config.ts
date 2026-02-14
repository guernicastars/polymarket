import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["@clickhouse/client"],
};

export default nextConfig;
