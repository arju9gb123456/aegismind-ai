"""Digital-twin topology model.

Everything here is a *virtual* representation of an enterprise-like network.
Nothing in this module touches a real host, socket or device.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

ASSET_KINDS = (
    "workstation",
    "admin_workstation",
    "web_server",
    "app_server",
    "file_server",
    "mail_server",
    "domain_controller",
    "db_server",
)

ZONES = ("user", "dmz", "server", "data")

RELATIONS = ("network", "credential", "trust")


@dataclass
class Service:
    name: str
    port: int


@dataclass
class Asset:
    id: str
    kind: str
    zone: str
    criticality: float  # 0..1, business importance
    exposure: float  # 0..1, simulated weakness proxy (NOT a real CVE score)
    services: list[Service] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in ASSET_KINDS:
            raise ValueError(f"unknown asset kind: {self.kind}")
        if self.zone not in ZONES:
            raise ValueError(f"unknown zone: {self.zone}")
        self.services = [s if isinstance(s, Service) else Service(**s) for s in self.services]


@dataclass
class Edge:
    """Directed relationship src -> dst that an attacker could try to traverse."""

    src: str
    dst: str
    relation: str  # network | credential | trust
    service: str  # e.g. smb, rdp, ssh, http, sql, cached-admin
    port: int | None
    exploitability: float  # 0..1, simulated probability the step succeeds
    allowed: bool = True  # False once a (simulated) policy blocks it

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"unknown relation: {self.relation}")
        if not 0.0 < self.exploitability <= 1.0:
            raise ValueError("exploitability must be in (0, 1]")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.dst, self.service)


@dataclass
class Topology:
    name: str
    assets: dict[str, Asset] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    # ---------------------------------------------------------------- building
    def add_asset(self, asset: Asset) -> None:
        if asset.id in self.assets:
            raise ValueError(f"duplicate asset id: {asset.id}")
        self.assets[asset.id] = asset

    def add_edge(self, edge: Edge) -> None:
        if edge.src not in self.assets or edge.dst not in self.assets:
            raise ValueError(f"edge references unknown asset: {edge.src}->{edge.dst}")
        if edge.src == edge.dst:
            raise ValueError("self-loops are not allowed")
        self.edges.append(edge)

    # ----------------------------------------------------------------- queries
    def by_kind(self, *kinds: str) -> list[Asset]:
        return [a for a in self.assets.values() if a.kind in kinds]

    def by_zone(self, *zones: str) -> list[Asset]:
        return [a for a in self.assets.values() if a.zone in zones]

    def crown_jewels(self, threshold: float = 0.8) -> list[Asset]:
        return [a for a in self.assets.values() if a.criticality >= threshold]

    def to_networkx(self, only_allowed: bool = True) -> nx.DiGraph:
        """Collapse parallel edges into one DiGraph edge, keeping the easiest step.

        Edge attribute ``cost = -log(exploitability)`` so that the shortest
        path by ``cost`` is the most probable path.
        """
        import math

        g = nx.DiGraph(name=self.name)
        for a in self.assets.values():
            g.add_node(a.id, kind=a.kind, zone=a.zone, criticality=a.criticality,
                       exposure=a.exposure)
        for e in self.edges:
            if only_allowed and not e.allowed:
                continue
            cost = -math.log(e.exploitability)
            if g.has_edge(e.src, e.dst):
                cur = g[e.src][e.dst]
                if cur["cost"] <= cost:
                    cur["alternatives"].append(e.service)
                    continue
                alts = cur["alternatives"] + [cur["service"]]
            else:
                alts = []
            g.add_edge(e.src, e.dst, relation=e.relation, service=e.service, port=e.port,
                       exploitability=e.exploitability, cost=cost, hops=1,
                       alternatives=alts)
        return g

    # --------------------------------------------------------------- serialise
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "assets": [asdict(a) for a in self.assets.values()],
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Topology":
        topo = cls(name=data["name"])
        for a in data["assets"]:
            topo.add_asset(Asset(**a))
        for e in data["edges"]:
            topo.add_edge(Edge(**e))
        return topo

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Topology":
        return cls.from_dict(json.loads(Path(path).read_text()))
