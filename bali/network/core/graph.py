"""NetworkX multigraph model of Bali's 57 kecamatan."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import networkx as nx

from .types import CompositeRisk, District, Edge, GeoCoord, RiskScore

DATA_DIR = Path(__file__).parent.parent / "data"


def haversine_km(a: GeoCoord, b: GeoCoord) -> float:
    """Great-circle distance between two lat/lng points in km."""
    R = 6371.0
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlng = math.radians(b.lng - a.lng)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


class BaliGraph:
    """Geographic risk network of Bali's districts."""

    def __init__(self):
        self.G = nx.MultiGraph()
        self.districts: dict[str, District] = {}
        self.risk_scores: dict[str, CompositeRisk] = {}
        self._load_districts()
        self._load_edges()

    def _load_districts(self):
        """Load 57 kecamatan as graph nodes."""
        data = json.loads((DATA_DIR / "districts.json").read_text())
        for d in data["districts"]:
            district = District(
                id=d["id"],
                name=d["name"],
                regency=d["regency"],
                center=GeoCoord(d["center"]["lat"], d["center"]["lng"]),
                area_km2=d["area_km2"],
                population=d["population"],
                elevation_m=d["elevation_m"],
                coastal=d["coastal"],
                volcanic_proximity_km=d["volcanic_proximity_km"],
                dominant_zone=d["dominant_zone"],
                dominant_title=d["dominant_title"],
                infrastructure_index=d["infrastructure_index"],
                tourism_intensity=d["tourism_intensity"],
                foreign_investor_density=d["foreign_investor_density"],
                avg_land_price_usd_m2=d["avg_land_price_usd_m2"],
                tags=d.get("tags", []),
            )
            self.districts[district.id] = district
            self.G.add_node(
                district.id,
                name=district.name,
                regency=district.regency,
                lat=district.center.lat,
                lng=district.center.lng,
                area_km2=district.area_km2,
                population=district.population,
                elevation_m=district.elevation_m,
                coastal=district.coastal,
                volcanic_proximity_km=district.volcanic_proximity_km,
                infrastructure_index=district.infrastructure_index,
                tourism_intensity=district.tourism_intensity,
                foreign_investor_density=district.foreign_investor_density,
                avg_land_price_usd_m2=district.avg_land_price_usd_m2,
            )

    def _load_edges(self):
        """Load edges from JSON."""
        data = json.loads((DATA_DIR / "edges.json").read_text())
        for e in data["edges"]:
            self.G.add_edge(
                e["source"],
                e["target"],
                key=e["edge_type"],
                edge_type=e["edge_type"],
                weight=e["weight"],
                **e.get("properties", {}),
            )

    @property
    def num_districts(self) -> int:
        return self.G.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.G.number_of_edges()

    def get_district(self, district_id: str) -> Optional[District]:
        return self.districts.get(district_id)

    def get_neighbors(self, district_id: str, edge_type: Optional[str] = None) -> list[str]:
        """Get neighboring districts, optionally filtered by edge type."""
        neighbors = set()
        for u, v, data in self.G.edges(district_id, data=True):
            if edge_type is None or data.get("edge_type") == edge_type:
                other = v if u == district_id else u
                neighbors.add(other)
        return sorted(neighbors)

    def get_road_distance(self, src: str, dst: str) -> Optional[float]:
        """Shortest road distance between two districts (km)."""
        # Build road-only subgraph
        road_edges = [(u, v, k, d) for u, v, k, d in self.G.edges(keys=True, data=True)
                      if d.get("edge_type") == "road"]
        road_G = nx.Graph()
        for u, v, k, d in road_edges:
            dist = d.get("distance_km", 10)
            if road_G.has_edge(u, v):
                if dist < road_G[u][v]["weight"]:
                    road_G[u][v]["weight"] = dist
            else:
                road_G.add_edge(u, v, weight=dist)
        try:
            return nx.shortest_path_length(road_G, src, dst, weight="weight")
        except nx.NetworkXNoPath:
            return None

    def get_straight_distance(self, src: str, dst: str) -> float:
        """Straight-line distance between district centers (km)."""
        a = self.districts[src].center
        b = self.districts[dst].center
        return haversine_km(a, b)

    def betweenness_centrality(self) -> dict[str, float]:
        """Compute betweenness centrality (road network importance)."""
        road_G = nx.Graph()
        for u, v, d in self.G.edges(data=True):
            if d.get("edge_type") == "road":
                w = d.get("distance_km", 10)
                if not road_G.has_edge(u, v) or w < road_G[u][v]["weight"]:
                    road_G.add_edge(u, v, weight=w)
        return nx.betweenness_centrality(road_G, weight="weight")

    def articulation_points(self) -> list[str]:
        """Find critical districts whose removal disconnects the road network."""
        road_G = nx.Graph()
        for u, v, d in self.G.edges(data=True):
            if d.get("edge_type") == "road":
                road_G.add_edge(u, v)
        return sorted(nx.articulation_points(road_G))

    def risk_propagation(self, source_district: str, risk_type: str, decay: float = 0.7) -> dict[str, float]:
        """Simulate risk propagation from a source district through the network.

        Returns a dict of district_id -> propagated risk intensity (0-1).
        Risk decays by `decay` factor per hop through shared_risk_zone edges,
        and by `decay^2` per hop through road edges.
        """
        visited: dict[str, float] = {source_district: 1.0}
        queue = [(source_district, 1.0)]

        while queue:
            current, intensity = queue.pop(0)
            for u, v, data in self.G.edges(current, data=True):
                neighbor = v if u == current else u
                edge_type = data.get("edge_type", "road")

                if edge_type == "shared_risk_zone" and data.get("risk_type") == risk_type:
                    new_intensity = intensity * decay
                elif edge_type == "road":
                    new_intensity = intensity * decay * decay
                else:
                    continue

                if new_intensity < 0.05:
                    continue
                if neighbor not in visited or new_intensity > visited[neighbor]:
                    visited[neighbor] = new_intensity
                    queue.append((neighbor, new_intensity))

        return visited

    def get_regency_districts(self, regency: str) -> list[District]:
        """Get all districts in a regency."""
        return [d for d in self.districts.values() if d.regency == regency]

    def investment_hotspots(self, max_risk: float = 50, min_price: float = 0) -> list[District]:
        """Find districts with favorable risk/reward balance."""
        hotspots = []
        for d in self.districts.values():
            if d.avg_land_price_usd_m2 >= min_price:
                if d.id in self.risk_scores and self.risk_scores[d.id].composite_score <= max_risk:
                    hotspots.append(d)
        return sorted(hotspots, key=lambda x: self.risk_scores.get(x.id, CompositeRisk(
            district_id=x.id,
            environmental=RiskScore("environmental", 50, 0.5, {}),
            seismological=RiskScore("seismological", 50, 0.5, {}),
            legal=RiskScore("legal", 50, 0.5, {}),
            administrative=RiskScore("administrative", 50, 0.5, {}),
            composite_score=50,
            investment_grade="C",
            computed_at="",
        )).composite_score)

    def to_geojson(self) -> dict:
        """Export graph as GeoJSON for Leaflet visualization."""
        features = []
        for district_id, district in self.districts.items():
            risk = self.risk_scores.get(district_id)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [district.center.lng, district.center.lat],
                },
                "properties": {
                    "id": district.id,
                    "name": district.name,
                    "regency": district.regency,
                    "population": district.population,
                    "elevation_m": district.elevation_m,
                    "coastal": district.coastal,
                    "volcanic_proximity_km": district.volcanic_proximity_km,
                    "avg_land_price_usd_m2": district.avg_land_price_usd_m2,
                    "tourism_intensity": district.tourism_intensity,
                    "infrastructure_index": district.infrastructure_index,
                    "composite_risk": risk.composite_score if risk else None,
                    "investment_grade": risk.investment_grade if risk else None,
                    "env_risk": risk.environmental.score if risk else None,
                    "seismic_risk": risk.seismological.score if risk else None,
                    "legal_risk": risk.legal.score if risk else None,
                    "admin_risk": risk.administrative.score if risk else None,
                },
            })

        # Add edges as LineString features
        for u, v, data in self.G.edges(data=True):
            if u in self.districts and v in self.districts:
                du, dv = self.districts[u], self.districts[v]
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [du.center.lng, du.center.lat],
                            [dv.center.lng, dv.center.lat],
                        ],
                    },
                    "properties": {
                        "source": u,
                        "target": v,
                        "edge_type": data.get("edge_type", "unknown"),
                        "weight": data.get("weight", 0),
                    },
                })

        return {"type": "FeatureCollection", "features": features}

    def summary(self) -> dict:
        """Quick summary statistics."""
        regencies = set(d.regency for d in self.districts.values())
        coastal = sum(1 for d in self.districts.values() if d.coastal)
        edge_types = {}
        for _, _, d in self.G.edges(data=True):
            et = d.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "districts": self.num_districts,
            "regencies": len(regencies),
            "edges": self.num_edges,
            "edge_types": edge_types,
            "coastal_districts": coastal,
            "inland_districts": self.num_districts - coastal,
            "avg_price_usd_m2": sum(d.avg_land_price_usd_m2 for d in self.districts.values()) / max(len(self.districts), 1),
            "total_population": sum(d.population for d in self.districts.values()),
        }
