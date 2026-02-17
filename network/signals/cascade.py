"""Cascade simulation — 'what if city X falls?' scenario analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx

from ..core.types import ControlStatus

if TYPE_CHECKING:
    from ..core.graph import DonbasGraph


@dataclass
class CascadeResult:
    """Result of a cascade simulation."""
    trigger_node: str
    fallen_nodes: list[str] = field(default_factory=list)
    isolated_nodes: list[str] = field(default_factory=list)
    supply_cut_nodes: list[str] = field(default_factory=list)
    new_component_count: int = 0
    severity: float = 0.0  # 0-1, how bad is this cascade


class CascadeSimulator:
    """Simulate cascading effects of settlement losses."""

    SUPPLY_ORIGIN = "dnipro"

    def __init__(self, graph: "DonbasGraph") -> None:
        self.dg = graph

    def simulate_fall(self, node_id: str) -> CascadeResult:
        """Simulate what happens if a UA-held node falls to RU.

        Steps:
        1. Remove node from UA subgraph
        2. Check if remaining graph splits into components
        3. Identify nodes that lose supply connectivity to Dnipro
        4. Score severity based on isolated population and strategic value
        """
        result = CascadeResult(trigger_node=node_id)
        result.fallen_nodes.append(node_id)

        # Get current UA subgraph, then remove the fallen node
        ua_sub = self.dg.get_ua_subgraph()
        if node_id not in ua_sub:
            return result

        ua_sub.remove_node(node_id)
        simple = nx.Graph(ua_sub)

        # Find connected components
        components = list(nx.connected_components(simple))
        result.new_component_count = len(components)

        # Find which component has Dnipro (the supply origin)
        supply_component = set()
        for comp in components:
            if self.SUPPLY_ORIGIN in comp:
                supply_component = comp
                break

        # Nodes not connected to supply origin are isolated
        all_ua_nodes = set(simple.nodes())
        isolated = all_ua_nodes - supply_component - {node_id}
        result.isolated_nodes = sorted(isolated)

        # Supply cut = nodes that had a path to Dnipro but now don't
        result.supply_cut_nodes = sorted(isolated)

        # Severity: weighted by population and strategic value
        total_ua_pop = sum(
            s.population for s in self.dg.settlements.values()
            if self.dg.get_effective_control(s.id) in (ControlStatus.UA, ControlStatus.CONTESTED)
        )
        isolated_pop = sum(
            self.dg.settlements[n].population
            for n in result.isolated_nodes
            if n in self.dg.settlements
        )
        fallen_pop = self.dg.settlements.get(node_id, None)
        fallen_pop = fallen_pop.population if fallen_pop else 0

        if total_ua_pop > 0:
            result.severity = min((isolated_pop + fallen_pop) / total_ua_pop, 1.0)
        else:
            result.severity = 0.0

        # Bonus severity for polymarket targets in isolated set
        pm_isolated = [
            n for n in result.isolated_nodes
            if n in self.dg.settlements and self.dg.settlements[n].is_polymarket_target
        ]
        result.severity = min(result.severity + 0.1 * len(pm_isolated), 1.0)

        return result

    def simulate_multi_fall(self, node_ids: list[str]) -> CascadeResult:
        """Simulate multiple simultaneous losses."""
        result = CascadeResult(trigger_node=node_ids[0] if node_ids else "")
        result.fallen_nodes = list(node_ids)

        ua_sub = self.dg.get_ua_subgraph()
        for nid in node_ids:
            if nid in ua_sub:
                ua_sub.remove_node(nid)

        simple = nx.Graph(ua_sub)
        components = list(nx.connected_components(simple))
        result.new_component_count = len(components)

        supply_component = set()
        for comp in components:
            if self.SUPPLY_ORIGIN in comp:
                supply_component = comp
                break

        all_ua_nodes = set(simple.nodes())
        isolated = all_ua_nodes - supply_component
        result.isolated_nodes = sorted(isolated)
        result.supply_cut_nodes = sorted(isolated)

        total_ua_pop = sum(
            s.population for s in self.dg.settlements.values()
            if self.dg.get_effective_control(s.id) in (ControlStatus.UA, ControlStatus.CONTESTED)
        )
        affected_pop = sum(
            self.dg.settlements[n].population
            for n in list(isolated) + node_ids
            if n in self.dg.settlements
        )
        result.severity = min(affected_pop / max(total_ua_pop, 1), 1.0)

        return result

    def scenario_report(self, node_id: str) -> dict:
        """Generate a human-readable scenario report."""
        r = self.simulate_fall(node_id)
        name = self.dg.settlements[node_id].name if node_id in self.dg.settlements else node_id
        return {
            "scenario": f"If {name} falls",
            "fallen": r.fallen_nodes,
            "isolated_settlements": r.isolated_nodes,
            "isolated_names": [
                self.dg.settlements[n].name
                for n in r.isolated_nodes
                if n in self.dg.settlements
            ],
            "supply_cut": r.supply_cut_nodes,
            "new_components": r.new_component_count,
            "severity": round(r.severity, 3),
        }
