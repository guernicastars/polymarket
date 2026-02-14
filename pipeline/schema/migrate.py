"""Run ClickHouse schema migrations against ClickHouse Cloud.

Usage:
    python migrate.py

Environment variables:
    CLICKHOUSE_HOST      - ClickHouse Cloud hostname (required)
    CLICKHOUSE_PASSWORD  - ClickHouse Cloud password (required)
    CLICKHOUSE_PORT      - Port (default: 8443)
    CLICKHOUSE_USER      - Username (default: 'default')
"""

import os
import sys
from pathlib import Path

import clickhouse_connect


def get_client():
    """Create a ClickHouse Cloud client from environment variables."""
    host = os.environ.get("CLICKHOUSE_HOST")
    password = os.environ.get("CLICKHOUSE_PASSWORD")

    if not host or not password:
        print("ERROR: CLICKHOUSE_HOST and CLICKHOUSE_PASSWORD must be set")
        sys.exit(1)

    return clickhouse_connect.get_client(
        host=host,
        port=int(os.environ.get("CLICKHOUSE_PORT", "8443")),
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=password,
        secure=True,
        connect_timeout=30,
        send_receive_timeout=300,
    )


def run_migration(sql_path: Path):
    """Read a SQL file, split on semicolons, and execute each statement."""
    client = get_client()
    sql = sql_path.read_text()

    # Split on semicolons and filter out empty/whitespace-only statements
    statements = [s.strip() for s in sql.split(";") if s.strip()]

    total = len(statements)
    print(f"Running migration: {sql_path.name} ({total} statements)")
    print("-" * 60)

    for i, stmt in enumerate(statements, 1):
        # Extract first meaningful line for progress display
        first_line = ""
        for line in stmt.splitlines():
            line = line.strip()
            if line and not line.startswith("--"):
                first_line = line[:80]
                break

        print(f"[{i}/{total}] {first_line}...")

        try:
            client.command(stmt)
            print(f"         OK")
        except Exception as e:
            err_msg = str(e)
            if "already exists" in err_msg.lower():
                print(f"         SKIP (already exists)")
            else:
                print(f"         FAIL: {err_msg}")
                sys.exit(1)

    print("-" * 60)
    print(f"Migration complete: {total} statements executed.")


def main():
    sql_path = Path(__file__).parent / "001_init.sql"
    if not sql_path.exists():
        print(f"ERROR: Migration file not found: {sql_path}")
        sys.exit(1)

    run_migration(sql_path)


if __name__ == "__main__":
    main()
