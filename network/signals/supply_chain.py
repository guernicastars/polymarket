"""Supply chain analysis — path redundancy, min-cut, and supply risk."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import networkx as nx

from ..core.types import ControlStatus

if TYPE_CHECKING:
    from ..core.graph import DonbasGraph


class SupplyChainAnalyzer:
    """Analyze supply chain vulnerability from rear bases to frontline."""

    SUPPLY_ORIGIN = "dnipro"  # primary supply origin node

    def __init__(self, graph: "DonbasGraph") -> None:
        self.dg = graph

    def shortest_supply_path(
        self, target: str, origin: Optional[str] = None
    ) -> tuple[list[str], float]:
        """Find shortest weighted path from origin to target.

        Returns (path, total_weight). Empty path if unreachable.
        """
        src = origin or self.SUPPLY_ORIGIN
        simple = self.dg.get_active_edges_graph()
        try:
            path = nx.shortest_path(simple, src, target, weight="weight")
            cost = nx.shortest_path_length(simple, src, target, weight="weight")
            return path, cost
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], float("inf")

    def all_simple_paths(
        self, target: str, origin: Optional[str] = None, cutoff: int = 8
    ) -> list[list[str]]:
        """Find all simple paths (up to cutoff length) — measures redundancy."""
        src = origin or self.SUPPLY_ORIGIN
        simple = self.dg.get_active_edges_graph()
        try:
            return list(nx.all_simple_paths(simple, src, target, cutoff=cutoff))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def path_redundancy(
        self, target: str, origin: Optional[str] = None, cutoff: int = 8
    ) -> int:
        """Count of distinct supply routes — higher = more resilient."""
        return len(self.all_simple_paths(target, origin, cutoff))

    def min_cut_nodes(
        self, target: str, origin: Optional[str] = None
    ) -> tuple[int, set[str]]:
        """Minimum node cut between origin and target.

        Returns (cut_size, cut_nodes). If cut_size is low, supply is fragile.
        """
        src = origin or self.SUPPLY_ORIGIN
        simple = self.dg.get_active_edges_graph()
        try:
            cut_value = nx.node_connectivity(simple, src, target)
            cut_nodes = nx.minimum_node_cut(simple, src, target)
            return cut_value, cut_nodes
        except (nx.NetworkXError, nx.NodeNotFound):
            return 0, set()

    def supply_risk_score(
        self, target: str, origin: Optional[str] = None
    ) -> dict:
        """Composite supply risk assessment for a settlement."""
        path, cost = self.shortest_supply_path(target, origin)
        redundancy = self.path_redundancy(target, origin)
        cut_size, cut_nodes = self.min_cut_nodes(target, origin)

        # Risk: inverse of redundancy and cut size, scaled by path cost
        risk = 0.0
        if redundancy > 0:
            risk = 1.0 - min(redundancy / 10.0, 1.0)
        if cut_size > 0:
            risk = max(risk, 1.0 - min(cut_size / 5.0, 1.0))

        return {
            "target": target,
            "origin": origin or self.SUPPLY_ORIGIN,
            "shortest_path": path,
            "shortest_cost": cost,
            "path_redundancy": redundancy,
            "min_cut_size": cut_size,
            "min_cut_nodes": list(cut_nodes),
            "supply_risk": round(risk, 3),
        }

    def score_all_targets(self) -> list[dict]:
        """Supply risk for all Polymarket target settlements."""
        targets = self.dg.get_polymarket_targets()
        results = []
        for t in targets:
            if self.dg.get_effective_control(t.id) != ControlStatus.RU:
                results.append(self.supply_risk_score(t.id))
        return sorted(results, key=lambda x: x["supply_risk"], reverse=True)
