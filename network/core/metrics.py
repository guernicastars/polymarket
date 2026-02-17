"""Graph metrics — centrality, cut vertices, component analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from .graph import DonbasGraph


class GraphMetrics:
    """Compute structural metrics on the Donbas graph."""

    def __init__(self, graph: "DonbasGraph") -> None:
        self.dg = graph
        # Use simple undirected graph for standard algorithms
        self._simple = self._to_simple()

    def _to_simple(self) -> nx.Graph:
        """Collapse multigraph to simple graph (min weight per edge pair)."""
        simple = nx.Graph()
        for u, v, data in self.dg.G.edges(data=True):
            w = data.get("weight", 1.0)
            if simple.has_edge(u, v):
                if w < simple[u][v]["weight"]:
                    simple[u][v]["weight"] = w
            else:
                simple.add_edge(u, v, weight=w)
        return simple

    # ------------------------------------------------------------------
    # Centrality
    # ------------------------------------------------------------------

    def betweenness_centrality(self, normalized: bool = True) -> dict[str, float]:
        return nx.betweenness_centrality(self._simple, weight="weight", normalized=normalized)

    def degree_centrality(self) -> dict[str, float]:
        return nx.degree_centrality(self._simple)

    def closeness_centrality(self) -> dict[str, float]:
        return nx.closeness_centrality(self._simple, distance="weight")

    def eigenvector_centrality(self, max_iter: int = 1000) -> dict[str, float]:
        try:
            return nx.eigenvector_centrality(self._simple, max_iter=max_iter, weight="weight")
        except nx.PowerIterationFailedConvergence:
            return {n: 0.0 for n in self._simple.nodes}

    def top_centrality(self, measure: str = "betweenness", top_n: int = 10) -> list[tuple[str, float]]:
        funcs = {
            "betweenness": self.betweenness_centrality,
            "degree": self.degree_centrality,
            "closeness": self.closeness_centrality,
            "eigenvector": self.eigenvector_centrality,
        }
        scores = funcs[measure]()
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ------------------------------------------------------------------
    # Cut vertices & bridges
    # ------------------------------------------------------------------

    def articulation_points(self) -> list[str]:
        """Nodes whose removal disconnects the graph."""
        return list(nx.articulation_points(self._simple))

    def bridges(self) -> list[tuple[str, str]]:
        """Edges whose removal disconnects the graph."""
        return list(nx.bridges(self._simple))

    # ------------------------------------------------------------------
    # UA-specific subgraph analysis
    # ------------------------------------------------------------------

    def ua_articulation_points(self) -> list[str]:
        """Cut vertices in the Ukrainian-held subgraph."""
        ua_sub = self.dg.get_ua_subgraph()
        simple_ua = nx.Graph()
        for u, v, data in ua_sub.edges(data=True):
            w = data.get("weight", 1.0)
            if simple_ua.has_edge(u, v):
                if w < simple_ua[u][v]["weight"]:
                    simple_ua[u][v]["weight"] = w
            else:
                simple_ua.add_edge(u, v, weight=w)
        return list(nx.articulation_points(simple_ua))

    def ua_components(self) -> list[set[str]]:
        """Connected components of UA subgraph."""
        ua_sub = self.dg.get_ua_subgraph()
        simple_ua = nx.Graph(ua_sub)
        return [c for c in nx.connected_components(simple_ua)]

    def component_count(self) -> int:
        return nx.number_connected_components(self._simple)
