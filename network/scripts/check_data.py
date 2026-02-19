"""Check what ClickHouse data exists for Ukraine GNN target markets."""
import json
import os
import pathlib
import clickhouse_connect

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    client = clickhouse_connect.get_client(
        host=os.environ.get("CLICKHOUSE_HOST", "ch.bloomsburytech.com"),
        port=int(os.environ.get("CLICKHOUSE_PORT", "443")),
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        database=os.environ.get("CLICKHOUSE_DATABASE", "polymarket"),
        secure=True,
    )
    print("Connected to ClickHouse\n")

    with open(DATA_DIR / "polymarket_mapping.json") as f:
        mapping = json.load(f)

    print(f"{'Settlement':<18} {'Table':<22} {'Rows':>8} {'First':>22} {'Last':>22}")
    print("=" * 95)

    tables = [
        ("market_prices", "condition_id", "timestamp"),
        ("market_trades", "condition_id", "timestamp"),
        ("ohlcv_1h", "condition_id", "bar_time"),
        ("ohlcv_1m", "condition_id", "bar_time"),
        ("orderbook_snapshots", "condition_id", "snapshot_time"),
    ]

    for settlement, info in mapping.items():
        cid = info.get("condition_id", "")
        if not cid:
            print(f"{settlement:<18} NO CONDITION ID")
            continue

        for table, cid_col, time_col in tables:
            try:
                rows = client.query(f"""
                    SELECT count(), min({time_col}), max({time_col})
                    FROM {table}
                    WHERE {cid_col} = {{cid:String}}
                """, parameters={"cid": cid}).result_rows
                cnt, first, last = rows[0]
                if cnt > 0:
                    print(f"{settlement:<18} {table:<22} {cnt:>8,} {str(first):>22} {str(last):>22}")
                else:
                    print(f"{settlement:<18} {table:<22}        0")
            except Exception as e:
                print(f"{settlement:<18} {table:<22} ERROR: {e}")
        print()

    # Also check token_id based queries
    print("\n" + "=" * 95)
    print("TOKEN-LEVEL: market_prices by token_id")
    print("=" * 95)
    for settlement, info in mapping.items():
        tid = info.get("token_id", "")
        if not tid:
            continue
        try:
            rows = client.query("""
                SELECT count(), min(timestamp), max(timestamp)
                FROM market_prices
                WHERE token_id = {tid:String}
            """, parameters={"tid": tid}).result_rows
            cnt = rows[0][0]
            if cnt > 0:
                print(f"  {settlement}: {cnt:,} rows ({rows[0][1]} → {rows[0][2]})")
            else:
                print(f"  {settlement}: 0 rows")
        except Exception as e:
            print(f"  {settlement}: ERROR: {e}")


if __name__ == "__main__":
    main()
