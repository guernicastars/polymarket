#!/usr/bin/env python3
"""Find Ukraine/Russia capture markets in ClickHouse and update polymarket_mapping.json.

Usage:
    cd ~/Coding/maynard/polymarket
    export $(grep -v '^#' pipeline/.env | xargs)
    python -m network.scripts.find_ukraine_markets
"""

import json
import os
import sys
import pathlib

import clickhouse_connect

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

def main():
    client = clickhouse_connect.get_client(
        host=os.environ.get("CLICKHOUSE_HOST", "ch.bloomsburytech.com"),
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        database=os.environ.get("CLICKHOUSE_DATABASE", "polymarket"),
        secure=True,
    )

    print("Connected to ClickHouse\n")

    # 1. Search for all Ukraine/Russia capture markets
    print("=" * 80)
    print("SEARCHING FOR UKRAINE/RUSSIA TERRITORY MARKETS")
    print("=" * 80)

    rows = client.query("""
        SELECT
            condition_id,
            question,
            market_slug,
            outcome_prices,
            volume_24h,
            volume_total,
            active,
            closed,
            end_date,
            token_ids
        FROM markets FINAL
        WHERE (
            lower(question) LIKE '%russia%capture%'
            OR lower(question) LIKE '%russia%control%'
            OR lower(question) LIKE '%ukraine%territory%'
            OR lower(question) LIKE '%ukraine%counterattack%'
            OR lower(question) LIKE '%ceasefire%'
            OR lower(question) LIKE '%donbas%'
            OR lower(question) LIKE '%pokrovsk%'
            OR lower(question) LIKE '%chasiv%yar%'
            OR lower(question) LIKE '%toretsk%'
            OR lower(question) LIKE '%kupiansk%'
            OR lower(question) LIKE '%zaporizhzhia%capture%'
            OR lower(question) LIKE '%orikhiv%'
            OR lower(question) LIKE '%hryshyne%'
            OR lower(question) LIKE '%kostyantynivka%'
            OR lower(question) LIKE '%myrnohrad%'
            OR lower(question) LIKE '%ivanopillya%'
            OR lower(question) LIKE '%rivnopillia%'
            OR lower(question) LIKE '%borova%'
            OR lower(question) LIKE '%sumy%capture%'
        )
        ORDER BY volume_total DESC
    """).result_rows

    print(f"\nFound {len(rows)} markets:\n")

    active_markets = []
    for r in rows:
        cid, question, slug, prices, vol24h, vol_total, active, closed, end_date, token_ids = r
        status = "ACTIVE" if active and not closed else ("CLOSED" if closed else "INACTIVE")
        print(f"  [{status}] {question}")
        print(f"    slug: {slug}")
        print(f"    condition_id: {cid}")
        print(f"    token_ids: {token_ids}")
        print(f"    prices: {prices}")
        print(f"    volume_24h: ${vol24h:,.0f}" if vol24h else "    volume_24h: N/A")
        print(f"    volume_total: ${vol_total:,.0f}" if vol_total else "    volume_total: N/A")
        print(f"    end_date: {end_date}")
        print()

        if active and not closed:
            active_markets.append({
                "condition_id": cid,
                "question": question,
                "slug": slug,
                "token_ids": token_ids,
                "prices": prices,
                "volume": vol_total,
                "end_date": str(end_date) if end_date else None,
            })

    # 2. Also search more broadly for ukraine-tagged events
    print("=" * 80)
    print("BROADER SEARCH: Ukraine-tagged markets with highest volume")
    print("=" * 80)

    rows2 = client.query("""
        SELECT
            condition_id,
            question,
            market_slug,
            outcome_prices,
            volume_24h,
            volume_total,
            active,
            closed,
            end_date,
            token_ids,
            category
        FROM markets FINAL
        WHERE (
            lower(category) LIKE '%ukraine%'
            OR has(tags, 'Ukraine')
            OR has(tags, 'ukraine')
            OR has(tags, 'Russia')
            OR has(tags, 'russia')
            OR has(tags, 'Ukraine-Russia')
        )
          AND active = 1
          AND closed = 0
        ORDER BY volume_total DESC
        LIMIT 30
    """).result_rows

    print(f"\nFound {len(rows2)} active Ukraine/Russia-tagged markets:\n")
    for r in rows2:
        cid, question, slug, prices, vol24h, vol_total, active, closed, end_date, token_ids, category = r
        print(f"  {question}")
        print(f"    slug: {slug}")
        print(f"    condition_id: {cid}")
        print(f"    token_ids: {token_ids}")
        print(f"    prices: {prices}  |  category: {category}")
        print(f"    volume: ${vol_total:,.0f}" if vol_total else "    volume: N/A")
        print()

    # 3. Check which settlement names from our mapping have actual markets
    print("=" * 80)
    print("MAPPING STATUS")
    print("=" * 80)

    mapping_path = DATA_DIR / "polymarket_mapping.json"
    with open(mapping_path) as f:
        current_mapping = json.load(f)

    print(f"\nCurrent mapping has {len(current_mapping)} settlements:")
    for sid, data in current_mapping.items():
        slug = data.get("slug", "")
        cid = data.get("condition_id")
        # Try to find a match
        matched = [m for m in active_markets if slug in (m.get("slug") or "")]
        if matched:
            print(f"  ✅ {sid}: FOUND — {matched[0]['question']}")
            print(f"       condition_id: {matched[0]['condition_id']}")
        elif cid:
            print(f"  ✓  {sid}: Has condition_id={cid}")
        else:
            print(f"  ❌ {sid}: NO MATCH (slug={slug})")

    # 4. Check price data availability for matched markets
    if active_markets:
        print(f"\n{'=' * 80}")
        print("PRICE DATA AVAILABILITY")
        print("=" * 80)

        for m in active_markets[:10]:
            cid = m["condition_id"]
            count_rows = client.query(f"""
                SELECT count() as cnt,
                       min(timestamp) as first_ts,
                       max(timestamp) as last_ts
                FROM market_prices
                WHERE condition_id = '{cid}'
            """).result_rows
            if count_rows and count_rows[0][0] > 0:
                cnt, first, last = count_rows[0]
                print(f"  {m['question'][:60]}...")
                print(f"    prices: {cnt:,} rows, {first} → {last}")
            else:
                print(f"  {m['question'][:60]}...")
                print(f"    prices: NO DATA")

    print("\n\nDone. Copy the condition_ids above into polymarket_mapping.json")
    print("or re-run with --update to auto-update the file.")

    # If --update flag, auto-generate updated mapping
    if "--update" in sys.argv:
        _auto_update_mapping(current_mapping, active_markets, rows)
        print("\n✅ Updated polymarket_mapping.json")


def _auto_update_mapping(current_mapping, active_markets, all_rows):
    """Try to match settlements to markets and update the mapping file."""
    # Build lookup by slug fragments
    all_market_data = {}
    for r in all_rows:
        cid, question, slug, prices, vol24h, vol_total, active, closed, end_date, token_ids = r
        if active and not closed and cid:
            all_market_data[slug] = {
                "condition_id": cid,
                "token_ids": token_ids,
                "question": question,
                "volume": vol_total,
            }

    settlement_keywords = {
        "pokrovsk": ["pokrovsk"],
        "chasiv_yar": ["chasiv-yar", "chasiv_yar", "chasiv"],
        "toretsk": ["toretsk", "toretske"],
        "kupiansk": ["kupiansk", "kupyansk"],
        "zaporizhzhia": ["zaporizhzhia"],
        "orikhiv": ["orikhiv"],
        "hryshyne": ["hryshyne"],
        "kostyantynivka": ["kostyantynivka"],
        "myrnohrad": ["myrnohrad"],
        "ivanopillya": ["ivanopillya"],
        "rivnopillia": ["rivnopillia"],
    }

    updated = {}
    for sid, data in current_mapping.items():
        keywords = settlement_keywords.get(sid, [sid])
        best_match = None
        best_vol = -1

        for slug, mdata in all_market_data.items():
            for kw in keywords:
                if kw in slug.lower():
                    vol = mdata.get("volume", 0) or 0
                    if vol > best_vol:
                        best_match = dict(mdata)  # copy to avoid mutation
                        best_match["slug"] = slug
                        best_vol = vol

        if best_match:
            # token_ids is already an array from ClickHouse
            token_ids_raw = best_match.get("token_ids", [])
            token_id = None
            if token_ids_raw:
                if isinstance(token_ids_raw, list) and len(token_ids_raw) > 0:
                    token_id = token_ids_raw[0]
                elif isinstance(token_ids_raw, str):
                    try:
                        ids = json.loads(token_ids_raw)
                        token_id = ids[0] if ids else None
                    except (json.JSONDecodeError, IndexError):
                        token_id = token_ids_raw

            updated[sid] = {
                "slug": best_match["slug"],
                "description": data.get("description", best_match.get("question", "")),
                "condition_id": best_match["condition_id"],
                "token_id": token_id,
            }
            print(f"  ✅ {sid} → {best_match['slug']} (vol=${best_vol:,.0f})")
        else:
            updated[sid] = data
            print(f"  ⚠️  {sid} — no match found, keeping original")

    # Write updated mapping
    mapping_path = DATA_DIR / "polymarket_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(updated, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
