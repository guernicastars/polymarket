"""Core Donbas multigraph built on NetworkX."""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import networkx as nx

from .types import ControlStatus, EdgeType, Settlement, Edge, DynamicState


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


class DonbasGraph:
    """Multigraph of Donbas settlements and connections."""

    def __init__(self) -> None:
        self.G: nx.MultiGraph = nx.MultiGraph()
        self.settlements: dict[str, Settlement] = {}
        self.dynamic_states: dict[str, DynamicState] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_seed_data(cls, data_dir: Optional[pathlib.Path] = None) -> "DonbasGraph":
        """Build graph from JSON seed files."""
        d = data_dir or DATA_DIR
        g = cls()
        g._load_settlements(d / "settlements.json")
        g._load_edges(d / "edges.json")
        g._load_dynamic_state(d / "dynamic_state.json")
        return g

    def _load_settlements(self, path: pathlib.Path) -> None:
        with open(path) as f:
            raw = json.load(f)
        for s in raw:
            settlement = Settlement(
                id=s["id"],
                name=s["name"],
                lat=s["lat"],
                lng=s["lng"],
                control=ControlStatus(s["control"]),
                population=s.get("population", 0),
                is_polymarket_target=s.get("is_polymarket_target", False),
                polymarket_slug=s.get("polymarket_slug"),
                oblast=s.get("oblast", ""),
                fortification_level=s.get("fortification_level", 0.5),
                terrain_difficulty=s.get("terrain_difficulty", 0.5),
                garrison_estimate=s.get("garrison_estimate", 0),
                tags=s.get("tags", []),
            )
            self.settlements[settlement.id] = settlement
            self.G.add_node(
                settlement.id,
                name=settlement.name,
                control=settlement.control.value,
                lat=settlement.lat,
                lng=settlement.lng,
                population=settlement.population,
                is_polymarket_target=settlement.is_polymarket_target,
                fortification_level=settlement.fortification_level,
                terrain_difficulty=settlement.terrain_difficulty,
            )

    def _load_edges(self, path: pathlib.Path) -> None:
        with open(path) as f:
            raw = json.load(f)
        for e in raw:
            edge = Edge(
                source=e["source"],
                target=e["target"],
                edge_type=EdgeType(e["edge_type"]),
                weight=e.get("weight", 1.0),
                capacity=e.get("capacity", 1.0),
                distance_km=e.get("distance_km", 0.0),
                is_active=e.get("is_active", True),
                is_frontline=e.get("is_frontline", False),
                description=e.get("description", ""),
            )
            self.G.add_edge(
                edge.source,
                edge.target,
                key=edge.edge_type.value,
                weight=edge.weight,
                capacity=edge.capacity,
                distance_km=edge.distance_km,
                is_active=edge.is_active,
                is_frontline=edge.is_frontline,
                edge_type=edge.edge_type.value,
            )

    def _load_dynamic_state(self, path: pathlib.Path) -> None:
        with open(path) as f:
            raw = json.load(f)
        for ds in raw:
            state = DynamicState(
                settlement_id=ds["settlement_id"],
                control_override=(
                    ControlStatus(ds["control_override"])
                    if ds.get("control_override")
                    else None
                ),
                assault_intensity=ds.get("assault_intensity", 0.0),
                shelling_intensity=ds.get("shelling_intensity", 0.0),
                supply_disruption=ds.get("supply_disruption", 0.0),
                frontline_distance_km=ds.get("frontline_distance_km", 50.0),
                last_updated=ds.get("last_updated", ""),
            )
            self.dynamic_states[state.settlement_id] = state
            # apply control override to graph
            if state.control_override and state.settlement_id in self.G:
                self.G.nodes[state.settlement_id]["control"] = state.control_override.value

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_settlement(self, sid: str) -> Optional[Settlement]:
        return self.settlements.get(sid)

    def get_effective_control(self, sid: str) -> ControlStatus:
        ds = self.dynamic_states.get(sid)
        if ds and ds.control_override:
            return ds.control_override
        s = self.settlements.get(sid)
        return s.control if s else ControlStatus.CONTESTED

    def get_polymarket_targets(self) -> list[Settlement]:
        return [s for s in self.settlements.values() if s.is_polymarket_target]

    def get_by_control(self, status: ControlStatus) -> list[Settlement]:
        return [
            s for s in self.settlements.values()
            if self.get_effective_control(s.id) == status
        ]

    def get_ua_subgraph(self) -> nx.MultiGraph:
        """Subgraph of Ukrainian-held + contested nodes."""
        nodes = [
            s.id for s in self.settlements.values()
            if self.get_effective_control(s.id) in (ControlStatus.UA, ControlStatus.CONTESTED)
        ]
        return self.G.subgraph(nodes).copy()

    def get_active_edges_graph(self) -> nx.Graph:
        """Simple graph with only active edges (for pathfinding)."""
        simple = nx.Graph()
        for u, v, key, data in self.G.edges(keys=True, data=True):
            if data.get("is_active", True):
                w = data.get("weight", 1.0)
                if simple.has_edge(u, v):
                    if w < simple[u][v]["weight"]:
                        simple[u][v]["weight"] = w
                else:
                    simple.add_edge(u, v, weight=w, capacity=data.get("capacity", 1.0))
        return simple

    @property
    def node_count(self) -> int:
        return self.G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.G.number_of_edges()

    def summary(self) -> dict:
        ua = len(self.get_by_control(ControlStatus.UA))
        ru = len(self.get_by_control(ControlStatus.RU))
        contested = len(self.get_by_control(ControlStatus.CONTESTED))
        targets = len(self.get_polymarket_targets())
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "ua_held": ua,
            "ru_held": ru,
            "contested": contested,
            "polymarket_targets": targets,
        }
