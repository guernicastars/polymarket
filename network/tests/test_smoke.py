"""Smoke tests for the Donbas network model."""

import pytest

from network.core.types import ControlStatus, EdgeType, Settlement, VulnerabilityScore
from network.core.graph import DonbasGraph
from network.core.metrics import GraphMetrics
from network.signals.vulnerability import VulnerabilityAnalyzer
from network.signals.supply_chain import SupplyChainAnalyzer
from network.signals.cascade import CascadeSimulator
from network.signals.market_signal import MarketSignalGenerator


@pytest.fixture
def graph():
    return DonbasGraph.from_seed_data()


@pytest.fixture
def metrics(graph):
    return GraphMetrics(graph)


# ------------------------------------------------------------------
# Test 1: Graph loads correctly
# ------------------------------------------------------------------
def test_graph_loads(graph):
    assert graph.node_count == 40
    assert graph.edge_count >= 55
    summary = graph.summary()
    assert summary["polymarket_targets"] == 10


# ------------------------------------------------------------------
# Test 2: Control status overlay works
# ------------------------------------------------------------------
def test_control_status(graph):
    # Hryshyne has control_override=CONTESTED in dynamic_state
    assert graph.get_effective_control("hryshyne") == ControlStatus.CONTESTED
    # Bakhmut is RU
    assert graph.get_effective_control("bakhmut") == ControlStatus.RU
    # Pokrovsk is UA
    assert graph.get_effective_control("pokrovsk") == ControlStatus.UA


# ------------------------------------------------------------------
# Test 3: Centrality — Pokrovsk should rank high
# ------------------------------------------------------------------
def test_centrality(metrics):
    top = metrics.top_centrality("betweenness", top_n=5)
    top_ids = [sid for sid, _ in top]
    # Pokrovsk should be in top 5 for betweenness (logistics hub)
    assert "pokrovsk" in top_ids or "kramatorsk" in top_ids


# ------------------------------------------------------------------
# Test 4: Cut vertices exist in UA subgraph
# ------------------------------------------------------------------
def test_articulation_points(metrics):
    points = metrics.ua_articulation_points()
    assert len(points) > 0
    # At least some of these expected cut points should appear
    expected = {"pavlohrad", "kupiansk", "borova", "izium", "orikhiv", "pokrovsk"}
    assert len(set(points) & expected) > 0


# ------------------------------------------------------------------
# Test 5: Vulnerability scoring
# ------------------------------------------------------------------
def test_vulnerability(graph, metrics):
    analyzer = VulnerabilityAnalyzer(graph, metrics)
    top = analyzer.top_vulnerable(top_n=5)
    assert len(top) > 0
    # All scores should be in [0, 1] range
    for sid, score in top:
        assert 0.0 <= score <= 1.0
    # Hryshyne or Pokrovsk should be among top vulnerable
    top_ids = [sid for sid, _ in top]
    assert "hryshyne" in top_ids or "pokrovsk" in top_ids or "chasiv_yar" in top_ids


# ------------------------------------------------------------------
# Test 6: Supply chain analysis
# ------------------------------------------------------------------
def test_supply_chain(graph):
    supply = SupplyChainAnalyzer(graph)
    # Pokrovsk should have a supply path from Dnipro
    path, cost = supply.shortest_supply_path("pokrovsk")
    assert len(path) > 0
    assert cost < float("inf")
    assert path[0] == "dnipro"
    assert path[-1] == "pokrovsk"

    # Redundancy should be > 0
    redundancy = supply.path_redundancy("pokrovsk")
    assert redundancy >= 1


# ------------------------------------------------------------------
# Test 7: Cascade simulation
# ------------------------------------------------------------------
def test_cascade(graph):
    cascade = CascadeSimulator(graph)
    result = cascade.simulate_fall("pokrovsk")
    assert result.trigger_node == "pokrovsk"
    assert "pokrovsk" in result.fallen_nodes
    assert result.severity > 0.0
    # Should produce a valid scenario report
    report = cascade.scenario_report("pokrovsk")
    assert "If Pokrovsk falls" in report["scenario"]


# ------------------------------------------------------------------
# Test 8: Market signal generation
# ------------------------------------------------------------------
def test_market_signals(graph, metrics):
    vuln = VulnerabilityAnalyzer(graph, metrics)
    supply = SupplyChainAnalyzer(graph)
    cascade = CascadeSimulator(graph)
    gen = MarketSignalGenerator(graph, vuln, supply, cascade)

    signals = gen.generate_all_signals()
    assert len(signals) > 0

    for sig in signals:
        assert sig.direction in ("BUY", "SELL", "HOLD")
        assert 0.0 <= sig.model_probability <= 1.0
        assert sig.kelly_fraction >= 0.0
