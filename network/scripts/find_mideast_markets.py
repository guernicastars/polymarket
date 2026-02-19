#!/usr/bin/env python3
"""Find Middle East conflict markets in ClickHouse and generate polymarket_mapping_mideast.json.

Usage:
    cd ~/Coding/maynard/polymarket
    export $(grep -v '^#' pipeline/.env | xargs)
    python -m network.scripts.find_mideast_markets          # discover only
    python -m network.scripts.find_mideast_markets --update  # write mapping file
"""

import json
import os
import sys
import pathlib

import clickhouse_connect

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# Settlement keywords → market search terms
# Each settlement maps to slug/question substrings that would indicate a matching market
SETTLEMENT_KEYWORDS = {
    # Gaza
    "gaza_city": ["gaza"],
    "khan_younis": ["khan-younis", "khan_younis"],
    "rafah": ["rafah"],
    "jabalia": ["jabalia"],
    # Israel
    "tel_aviv": ["tel-aviv", "tel_aviv"],
    "jerusalem": ["jerusalem"],
    "haifa": ["haifa"],
    # Lebanon
    "beirut": ["beirut", "lebanon"],
    "south_lebanon": ["south-lebanon", "litani"],
    # Syria
    "damascus": ["damascus", "syria"],
    # Iran
    "tehran": ["tehran", "iran"],
    "isfahan": ["isfahan"],
    "natanz": ["natanz", "nuclear"],
    # Yemen / Maritime
    "sanaa": ["sanaa", "houthi", "yemen"],
    "hodeidah": ["hodeidah"],
    "bab_el_mandeb": ["bab-el-mandeb", "red-sea", "shipping"],
    "strait_of_hormuz": ["hormuz"],
}


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

    # 1. Search for Middle East conflict markets
    print("=" * 90)
    print("SEARCHING FOR MIDDLE EAST CONFLICT MARKETS")
    print("=" * 90)

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
            lower(question) LIKE '%israel%'
            OR lower(question) LIKE '%hamas%'
            OR lower(question) LIKE '%gaza%'
            OR lower(question) LIKE '%hezbollah%'
            OR lower(question) LIKE '%lebanon%war%'
            OR lower(question) LIKE '%ceasefire%israel%'
            OR lower(question) LIKE '%ceasefire%hamas%'
            OR lower(question) LIKE '%ceasefire%gaza%'
            OR lower(question) LIKE '%iran%strike%'
            OR lower(question) LIKE '%iran%attack%'
            OR lower(question) LIKE '%iran%nuclear%'
            OR lower(question) LIKE '%iran%bomb%'
            OR lower(question) LIKE '%iran%ceasefire%'
            OR lower(question) LIKE '%iran%war%'
            OR lower(question) LIKE '%houthi%'
            OR lower(question) LIKE '%red sea%'
            OR lower(question) LIKE '%netanyahu%'
            OR lower(question) LIKE '%rafah%'
            OR lower(question) LIKE '%hostage%'
            OR lower(question) LIKE '%west bank%'
            OR lower(question) LIKE '%golan%'
            OR lower(question) LIKE '%hormuz%'
            OR lower(question) LIKE '%middle east%'
            OR lower(question) LIKE '%syria%'
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
        print(f"    prices: {prices}  |  vol_total: ${vol_total:,.0f}" if vol_total else f"    prices: {prices}")
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

    # 2. Tagged search
    print("=" * 90)
    print("TAG-BASED SEARCH: Israel/Hamas/Iran/Middle East tagged markets")
    print("=" * 90)

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
            has(tags, 'Israel')
            OR has(tags, 'israel')
            OR has(tags, 'Hamas')
            OR has(tags, 'hamas')
            OR has(tags, 'Iran')
            OR has(tags, 'iran')
            OR has(tags, 'Middle East')
            OR has(tags, 'Gaza')
            OR has(tags, 'Hezbollah')
            OR has(tags, 'Lebanon')
            OR has(tags, 'Houthi')
            OR has(tags, 'Yemen')
            OR lower(category) LIKE '%israel%'
            OR lower(category) LIKE '%middle east%'
        )
          AND active = 1
          AND closed = 0
        ORDER BY volume_total DESC
        LIMIT 40
    """).result_rows

    print(f"\nFound {len(rows2)} active tagged markets:\n")
    for r in rows2:
        cid, question, slug, prices, vol24h, vol_total, active, closed, end_date, token_ids, category = r
        # Skip if already in active_markets
        if any(m["condition_id"] == cid for m in active_markets):
            continue
        print(f"  {question}")
        print(f"    slug: {slug}  |  category: {category}")
        print(f"    condition_id: {cid}")
        print(f"    volume: ${vol_total:,.0f}" if vol_total else "    volume: N/A")
        print()

    # 3. Price data availability
    print("=" * 90)
    print("PRICE DATA AVAILABILITY (top active markets)")
    print("=" * 90)

    for m in active_markets[:15]:
        cid = m["condition_id"]
        for table, col in [("market_prices", "timestamp"), ("market_trades", "timestamp")]:
            try:
                count_rows = client.query(f"""
                    SELECT count() as cnt,
                           min({col}) as first_ts,
                           max({col}) as last_ts
                    FROM {table}
                    WHERE condition_id = '{{cid}}'
                """.replace("{cid}", cid)).result_rows
                if count_rows and count_rows[0][0] > 0:
                    cnt, first, last = count_rows[0]
                    print(f"  {m['question'][:65]}")
                    print(f"    {table}: {cnt:,} rows, {first} -> {last}")
            except Exception as e:
                print(f"  {table} ERROR: {e}")

    # 4. Settlement match summary
    print(f"\n{'=' * 90}")
    print("SETTLEMENT MATCH SUMMARY")
    print("=" * 90)

    slug_lookup = {}
    for m in active_markets:
        slug_lookup[m["slug"]] = m

    match_count = 0
    for sid, keywords in SETTLEMENT_KEYWORDS.items():
        matched = None
        best_vol = -1
        for slug, mdata in slug_lookup.items():
            q = (mdata.get("question", "") + " " + slug).lower()
            for kw in keywords:
                if kw in q:
                    vol = mdata.get("volume", 0) or 0
                    if vol > best_vol:
                        matched = mdata
                        best_vol = vol
        if matched:
            print(f"  + {sid}: {matched['question'][:60]} (${best_vol:,.0f})")
            match_count += 1
        else:
            print(f"  - {sid}: no direct market match")

    print(f"\nMatched {match_count}/{len(SETTLEMENT_KEYWORDS)} settlements to active markets.")
    print("\nNote: Most Middle East markets are thematic (ceasefire, war) rather than")
    print("per-settlement. The GNN maps settlements to thematic market outcomes.")

    # 5. Auto-update if requested
    if "--update" in sys.argv:
        _auto_update_mapping(active_markets, client)
        print("\nUpdated polymarket_mapping_mideast.json")

    print("\nDone.")


def _auto_update_mapping(active_markets, client):
    """Generate polymarket_mapping_mideast.json with the best market matches."""
    # For Middle East, markets are thematic — not per-settlement
    # We map the most relevant market to each settlement region
    mapping = {}

    slug_lookup = {m["slug"]: m for m in active_markets}

    # Define which markets map to which graph nodes
    # Unlike Ukraine (1 market per city), Middle East uses region-level mapping
    REGION_MARKET_PATTERNS = {
        # Gaza settlements map to Hamas ceasefire markets
        "gaza_ceasefire": {
            "keywords": ["ceasefire", "hamas"],
            "settlements": ["gaza_city", "khan_younis", "rafah", "jabalia", "deir_al_balah", "netzarim"],
        },
        # Israel settlements map to Israel-Iran conflict markets
        "israel_iran": {
            "keywords": ["iran", "ceasefire", "broken"],
            "settlements": ["tel_aviv", "jerusalem", "haifa", "golan_heights"],
        },
        # Lebanon/Hezbollah settlements
        "lebanon_conflict": {
            "keywords": ["hezbollah", "lebanon"],
            "settlements": ["beirut", "south_lebanon", "nabatieh", "baalbek"],
        },
        # Iran nuclear
        "iran_nuclear": {
            "keywords": ["iran", "nuclear"],
            "settlements": ["tehran", "isfahan", "natanz"],
        },
        # Houthi/Red Sea
        "houthi_red_sea": {
            "keywords": ["houthi", "red sea"],
            "settlements": ["sanaa", "hodeidah", "bab_el_mandeb"],
        },
    }

    # Find best market for each region
    for region_key, region_cfg in REGION_MARKET_PATTERNS.items():
        best_market = None
        best_vol = -1
        for m in active_markets:
            q = (m.get("question", "") + " " + m.get("slug", "")).lower()
            if all(kw in q for kw in region_cfg["keywords"]):
                vol = m.get("volume", 0) or 0
                if vol > best_vol:
                    best_market = m
                    best_vol = vol

        if best_market:
            token_ids_raw = best_market.get("token_ids", [])
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

            for sid in region_cfg["settlements"]:
                mapping[sid] = {
                    "slug": best_market["slug"],
                    "description": best_market["question"],
                    "condition_id": best_market["condition_id"],
                    "token_id": token_id,
                    "region_key": region_key,
                }
                print(f"  {sid} -> {best_market['slug']}")

    # Also include any unmatched high-volume markets as standalone entries
    mapped_cids = {v["condition_id"] for v in mapping.values()}
    for m in active_markets[:5]:
        if m["condition_id"] not in mapped_cids:
            slug_key = m["slug"].replace("-", "_")[:30]
            token_ids_raw = m.get("token_ids", [])
            token_id = None
            if isinstance(token_ids_raw, list) and len(token_ids_raw) > 0:
                token_id = token_ids_raw[0]
            mapping[f"_extra_{slug_key}"] = {
                "slug": m["slug"],
                "description": m["question"],
                "condition_id": m["condition_id"],
                "token_id": token_id,
                "region_key": "extra",
            }
            print(f"  extra: {m['slug']}")

    # Write mapping
    mapping_path = DATA_DIR / "polymarket_mapping_mideast.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
