# Bali Real Estate Risk Intelligence Platform — Implementation Plan

## Architecture Overview

```
bali/
├── network/                    # Python: Geographic risk network model
│   ├── core/
│   │   ├── graph.py            # NetworkX multigraph of 57 kecamatan
│   │   ├── types.py            # Data classes (District, RiskScore, Edge, etc.)
│   │   └── metrics.py          # Graph metrics (centrality, connectivity, risk propagation)
│   ├── data/
│   │   ├── districts.json      # 57 kecamatan with coordinates, elevation, attributes
│   │   ├── edges.json          # Road/proximity/risk-zone edges
│   │   └── risk_zones.json     # Volcanic, tsunami, flood, liquefaction zones
│   ├── risks/
│   │   ├── environmental.py    # Flood, volcanic, landslide, tsunami scoring
│   │   ├── seismological.py    # Fault proximity, historical quakes, liquefaction
│   │   ├── legal.py            # Ownership type, title, zoning, dispute risk
│   │   ├── administrative.py   # Permits, infrastructure, bureaucratic complexity
│   │   └── composite.py        # Weighted multi-factor risk aggregation
│   ├── scrapers/
│   │   ├── bmkg.py             # BMKG seismic + weather API (earthquake feed)
│   │   ├── property.py         # Property listing scraper (Rumah123, Lamudi)
│   │   ├── osm.py              # OpenStreetMap infrastructure data
│   │   └── scheduler.py        # Scraper orchestration
│   ├── viz/
│   │   └── bali-risk-map.html  # Leaflet.js interactive risk map
│   └── tests/
├── dashboard/                  # Next.js risk assessment UI
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx                    # Overview: island risk heatmap + stats
│   │   │   ├── districts/page.tsx          # All 57 districts ranked by risk
│   │   │   ├── district/[id]/page.tsx      # Individual district deep-dive
│   │   │   ├── seismic/page.tsx            # Seismic activity + historical data
│   │   │   ├── legal/page.tsx              # Legal risk guide for investors
│   │   │   └── market/page.tsx             # Property market data + trends
│   │   ├── components/
│   │   │   ├── risk-map.tsx                # Interactive Leaflet risk map
│   │   │   ├── risk-radar.tsx              # 4-axis radar chart per district
│   │   │   ├── risk-score-card.tsx         # Score display with color coding
│   │   │   ├── district-table.tsx          # Sortable district comparison table
│   │   │   ├── seismic-feed.tsx            # Live earthquake feed
│   │   │   └── property-trends.tsx         # Price/volume charts
│   │   ├── lib/
│   │   │   ├── risk-engine.ts              # Client-side risk calculations
│   │   │   └── api.ts                      # API client for backend data
│   │   └── types/
│   │       └── index.ts                    # TypeScript interfaces
│   ├── package.json
│   └── tsconfig.json
└── README.md
```

## Implementation Phases

### Phase 1: Network Model + Seed Data (Steps 1-3)
1. Create district seed data (57 kecamatan with geo coords, attributes)
2. Build NetworkX graph with edges (road, proximity, shared risk zones)
3. Implement 4 risk scoring engines + composite aggregation

### Phase 2: Scrapers (Steps 4-5)
4. BMKG earthquake feed scraper (real-time seismic data)
5. Property listing scraper (Rumah123/Lamudi prices + volumes)
6. OSM infrastructure extraction

### Phase 3: Visualization (Step 6)
7. Leaflet.js interactive risk map with district polygons + heatmap overlay

### Phase 4: Dashboard (Steps 7-10)
8. Next.js app with overview, district list, district detail pages
9. Seismic feed page, legal guide, property market page
10. Risk radar charts, comparison tools, investment scoring

## Risk Scoring Model

Each district scored 0-100 on 4 axes:

| Category | Weight | Factors |
|----------|--------|---------|
| Environmental (0.30) | Flood zone overlap, volcanic proximity (Mt. Agung/Batur), landslide terrain slope, tsunami inundation zone, coastal erosion |
| Seismological (0.25) | Fault line distance, historical earthquake frequency/magnitude, soil liquefaction susceptibility, ground acceleration (PGA) |
| Legal (0.25) | Foreign ownership pathway (Hak Pakai/PMA/nominee risk), land title type (SHM/HGB/Girik), zoning compliance, dispute history density |
| Administrative (0.20) | IMB/PBG permit availability, infrastructure development index, bureaucratic complexity score, utility access |

Composite: weighted sum → 0-100 scale (lower = safer for investment)

## Data Sources

| Source | Type | Refresh | Data |
|--------|------|---------|------|
| BMKG API | Seismic | Real-time | Earthquake events, magnitude, depth, location |
| BMKG Weather | Environmental | Daily | Rainfall, flood warnings |
| Rumah123.com | Property | Daily | Listings, prices, locations, property types |
| Lamudi.co.id | Property | Daily | Listings, prices, investment properties |
| OpenStreetMap | Infrastructure | Weekly | Roads, utilities, hospitals, schools |
| BNPB (Disaster Agency) | Risk zones | Monthly | Flood, landslide, tsunami hazard maps |
| BPN (Land Agency) | Legal | Static + updates | Title types by area, dispute records |
