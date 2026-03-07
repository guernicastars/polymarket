"""Bali Real Estate Risk Intelligence — CLI entry point.

Usage:
    python -m bali.network                    # Full risk analysis + generate map
    python -m bali.network --mode risk        # Just risk scoring
    python -m bali.network --mode map         # Generate interactive map
    python -m bali.network --mode rank        # Rank districts by investment value
    python -m bali.network --mode scrape      # Run scrapers (BMKG + property)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_risk_analysis():
    """Compute risk scores for all 57 districts."""
    from .core.graph import BaliGraph
    from .risks.composite import compute_all_risks, rank_by_investment_value, rank_districts_by_risk

    graph = BaliGraph()
    logger.info("Loaded graph: %s", graph.summary())

    risks = compute_all_risks(graph.districts)
    graph.risk_scores = risks

    # Print ranked results
    ranked = rank_districts_by_risk(risks)
    print("\n" + "=" * 80)
    print("BALI REAL ESTATE RISK ASSESSMENT — 57 Kecamatan")
    print("=" * 80)
    print(f"{'Rank':<5} {'District':<25} {'Regency':<15} {'Score':<8} {'Grade':<7} {'Env':<7} {'Seis':<7} {'Legal':<7} {'Admin':<7}")
    print("-" * 80)

    for i, (district_id, risk) in enumerate(ranked, 1):
        d = graph.districts[district_id]
        print(
            f"{i:<5} {d.name:<25} {d.regency:<15} "
            f"{risk.composite_score:<8.1f} {risk.investment_grade:<7} "
            f"{risk.environmental.score:<7.1f} {risk.seismological.score:<7.1f} "
            f"{risk.legal.score:<7.1f} {risk.administrative.score:<7.1f}"
        )

    # Investment value ranking
    print("\n" + "=" * 80)
    print("INVESTMENT VALUE RANKING (Risk-Adjusted Price)")
    print("=" * 80)
    value_ranked = rank_by_investment_value(graph.districts, risks)
    print(f"{'Rank':<5} {'District':<25} {'Grade':<7} {'Risk':<8} {'$/m²':<10} {'Value':<8}")
    print("-" * 65)
    for i, entry in enumerate(value_ranked[:20], 1):
        print(
            f"{i:<5} {entry['name']:<25} {entry['investment_grade']:<7} "
            f"{entry['composite_risk']:<8.1f} ${entry['avg_price_usd_m2']:<9,.0f} "
            f"{entry['value_score']:<8.1f}"
        )

    # Grade distribution
    grades = {}
    for _, risk in ranked:
        g = risk.investment_grade
        grades[g] = grades.get(g, 0) + 1
    print(f"\nGrade Distribution: {dict(sorted(grades.items()))}")

    return graph, risks


def generate_map():
    """Generate interactive Leaflet.js risk map."""
    from .core.graph import BaliGraph
    from .risks.composite import compute_all_risks

    graph = BaliGraph()
    risks = compute_all_risks(graph.districts)
    graph.risk_scores = risks

    # Build district data for JS
    districts_js = []
    for d_id, district in graph.districts.items():
        risk = risks.get(d_id)
        districts_js.append({
            "id": district.id,
            "name": district.name,
            "regency": district.regency,
            "lat": district.center.lat,
            "lng": district.center.lng,
            "population": district.population,
            "elevation_m": district.elevation_m,
            "coastal": district.coastal,
            "volcanic_proximity_km": district.volcanic_proximity_km,
            "avg_land_price_usd_m2": district.avg_land_price_usd_m2,
            "tourism_intensity": district.tourism_intensity,
            "infrastructure_index": district.infrastructure_index,
            "foreign_investor_density": district.foreign_investor_density,
            "composite_risk": risk.composite_score if risk else None,
            "investment_grade": risk.investment_grade if risk else None,
            "env_risk": risk.environmental.score if risk else None,
            "seismic_risk": risk.seismological.score if risk else None,
            "legal_risk": risk.legal.score if risk else None,
            "admin_risk": risk.administrative.score if risk else None,
        })

    # Build edges for JS
    edges_data = json.loads((Path(__file__).parent / "data" / "edges.json").read_text())
    edges_js = edges_data.get("edges", [])

    # Read template and inject data
    template_path = Path(__file__).parent / "viz" / "bali-risk-map.html"
    template = template_path.read_text()

    html = template.replace(
        "%%DISTRICTS_JSON%%",
        json.dumps(districts_js, indent=2),
    )

    # Inject edges
    html = html.replace(
        "// Initialize\nrenderDistricts();",
        f"window.EDGES = {json.dumps(edges_js)};\n// Initialize\nrenderDistricts();",
    )

    output_path = Path(__file__).parent / "viz" / "bali-risk-map-live.html"
    output_path.write_text(html)
    logger.info("Generated interactive map: %s", output_path)
    print(f"\nMap generated: {output_path}")
    return str(output_path)


async def run_scrapers():
    """Run BMKG and property scrapers."""
    from .scrapers.bmkg import BMKGScraper

    print("\n=== BMKG Earthquake Feed ===")
    bmkg = BMKGScraper()
    events = await bmkg.fetch_all()
    print(f"Fetched {len(events)} seismic events near Bali region")
    for event in events[:10]:
        print(f"  M{event.magnitude} at ({event.lat:.2f}, {event.lng:.2f}) depth={event.depth_km}km — {event.region}")

    # Property scraper requires BeautifulSoup — report availability
    try:
        from .scrapers.property import PropertyScraper
        print("\n=== Property Listings ===")
        scraper = PropertyScraper(max_pages=2)
        listings = await scraper.scrape_all()
        print(f"Scraped {len(listings)} property listings")
        stats = scraper.get_district_stats()
        for district_id, s in sorted(stats.items(), key=lambda x: -x[1]["count"])[:10]:
            print(f"  {district_id}: {s['count']} listings, avg ${s['avg_price_usd']:,.0f}")
    except ImportError as e:
        print(f"Property scraper unavailable (missing dependency): {e}")
        print("Install with: pip install beautifulsoup4")


def main():
    parser = argparse.ArgumentParser(description="Bali Real Estate Risk Intelligence")
    parser.add_argument("--mode", choices=["risk", "map", "rank", "scrape", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ("risk", "all"):
        run_risk_analysis()

    if args.mode in ("map", "all"):
        generate_map()

    if args.mode in ("scrape", "all"):
        asyncio.run(run_scrapers())

    if args.mode == "rank":
        run_risk_analysis()


if __name__ == "__main__":
    main()
