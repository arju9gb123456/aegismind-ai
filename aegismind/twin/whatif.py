"""Minimal what-if engine for the digital twin (expanded in Sprint 5).

Actions are applied to a *copy* of the virtual topology only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from aegismind.graph.paths import blast_radius, risk_weighted_paths

from .topology import Topology


@dataclass
class Action:
    kind: str  # block_edge | isolate_asset
    src: str | None = None
    dst: str | None = None
    asset: str | None = None

    def describe(self) -> str:
        if self.kind == "block_edge":
            return f"Block {self.src} -> {self.dst} (simulated)"
        return f"Isolate {self.asset} (simulated)"


def apply(t: Topology, action: Action) -> Topology:
    new = copy.deepcopy(t)
    for e in new.edges:
        if action.kind == "block_edge" and e.src == action.src and e.dst == action.dst:
            e.allowed = False
        elif action.kind == "isolate_asset" and action.asset in (e.src, e.dst):
            e.allowed = False
    return new


def compare(t: Topology, action: Action, entry: str, target: str) -> dict:
    """Baseline vs mitigation: best-path probability, path count and blast radius."""
    before_g, after_g = t.to_networkx(), apply(t, action).to_networkx()
    before = risk_weighted_paths(before_g, entry, target, k=10)
    after = risk_weighted_paths(after_g, entry, target, k=10)
    # legitimate-service impact proxy: allowed edges removed
    lost = before_g.number_of_edges() - after_g.number_of_edges()
    p0 = before[0].probability if before else 0.0
    p1 = after[0].probability if after else 0.0
    return {
        "action": action.describe(),
        "best_path_probability_before": round(p0, 6),
        "best_path_probability_after": round(p1, 6),
        "risk_reduction_pct": round(100 * (p0 - p1) / p0, 2) if p0 else 0.0,
        "paths_before": len(before),
        "paths_after": len(after),
        "blast_radius_before": blast_radius(before_g, entry)["reachable_count"],
        "blast_radius_after": blast_radius(after_g, entry)["reachable_count"],
        "edges_disabled": lost,
        "note": "Simulated on the digital twin only; requires human review.",
    }
