"""Tests for Bali risk network model."""

import pytest

from ..core.graph import BaliGraph, haversine_km
from ..core.types import GeoCoord
from ..risks.composite import compute_all_risks, investment_grade


class TestBaliGraph:
    def test_load_districts(self):
        g = BaliGraph()
        assert g.num_districts == 56

    def test_all_regencies_present(self):
        g = BaliGraph()
        regencies = set(d.regency for d in g.districts.values())
        expected = {"denpasar", "badung", "gianyar", "tabanan", "klungkung",
                    "bangli", "karangasem", "buleleng", "jembrana"}
        assert regencies == expected

    def test_edges_loaded(self):
        g = BaliGraph()
        assert g.num_edges > 50

    def test_neighbors(self):
        g = BaliGraph()
        neighbors = g.get_neighbors("ubud")
        assert "tegallalang" in neighbors
        assert "gianyar" in neighbors

    def test_road_distance(self):
        g = BaliGraph()
        dist = g.get_road_distance("kuta", "ubud")
        assert dist is not None
        assert dist > 20  # At least 20km by road

    def test_straight_distance(self):
        g = BaliGraph()
        dist = g.get_straight_distance("kuta", "ubud")
        assert 20 < dist < 40  # ~30km straight line

    def test_betweenness(self):
        g = BaliGraph()
        centrality = g.betweenness_centrality()
        assert len(centrality) > 0
        # Denpasar/Mengwi should be relatively central
        assert centrality.get("mengwi", 0) > 0

    def test_summary(self):
        g = BaliGraph()
        s = g.summary()
        assert s["districts"] == 56
        assert s["regencies"] == 9
        assert s["total_population"] > 3_000_000

    def test_geojson_export(self):
        g = BaliGraph()
        geojson = g.to_geojson()
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) > 57  # nodes + edges


class TestRiskScoring:
    def test_compute_all_risks(self):
        g = BaliGraph()
        risks = compute_all_risks(g.districts)
        assert len(risks) == 56

    def test_risk_scores_in_range(self):
        g = BaliGraph()
        risks = compute_all_risks(g.districts)
        for district_id, risk in risks.items():
            assert 0 <= risk.composite_score <= 100, f"{district_id}: {risk.composite_score}"
            assert 0 <= risk.environmental.score <= 100
            assert 0 <= risk.seismological.score <= 100
            assert 0 <= risk.legal.score <= 100
            assert 0 <= risk.administrative.score <= 100

    def test_volcanic_districts_higher_risk(self):
        g = BaliGraph()
        risks = compute_all_risks(g.districts)
        # Rendang (slope of Agung) should have higher environmental risk than Denpasar
        assert risks["rendang"].environmental.score > risks["denpasar_selatan"].environmental.score

    def test_investment_grades(self):
        assert investment_grade(20) == "A"
        assert investment_grade(30) == "B"
        assert investment_grade(50) == "C"
        assert investment_grade(60) == "D"
        assert investment_grade(80) == "F"

    def test_kuta_selatan_high_legal_risk(self):
        """Kuta Selatan has highest foreign investor density = higher legal risk."""
        g = BaliGraph()
        risks = compute_all_risks(g.districts)
        # High foreign density → ownership pathway risk
        assert risks["kuta_selatan"].legal.score > 30

    def test_risk_propagation(self):
        g = BaliGraph()
        risks = compute_all_risks(g.districts)
        g.risk_scores = risks
        prop = g.risk_propagation("rendang", "volcanic")
        assert "rendang" in prop
        assert prop["rendang"] == 1.0
        # Neighbors should have decayed risk
        assert any(v < 1.0 and v > 0 for k, v in prop.items() if k != "rendang")


class TestHaversine:
    def test_same_point(self):
        p = GeoCoord(-8.5, 115.2)
        assert haversine_km(p, p) == 0.0

    def test_known_distance(self):
        # Kuta to Ubud ~30km
        kuta = GeoCoord(-8.7230, 115.1750)
        ubud = GeoCoord(-8.5069, 115.2624)
        dist = haversine_km(kuta, ubud)
        assert 20 < dist < 35
