"""Attack-path baselines and graph analytics (NetworkX).

Baselines
---------
* ``hop_shortest_paths``  – fewest hops, ignores how easy each step is.
* ``risk_weighted_paths`` – most probable paths, cost = -log(exploitability).

Analytics
---------
* ``blast_radius`` – what an attacker could reach from a foothold.
* ``choke_points`` – edges whose removal cuts the most entry->target routes.

Outputs are *plausible* routes with scores, never a claim that an attack
will happen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice

import networkx as nx


@dataclass
class RankedPath:
    path: list[str]
    probability: float  # product of step exploitabilities (defender's estimate)
    hops: int
    evidence: list[dict]  # per-edge: service, relation, exploitability

    def to_dict(self) -> dict:
        return {"path": self.path, "probability": round(self.probability, 6),
                "hops": self.hops, "evidence": self.evidence}


def _rank(g: nx.DiGraph, paths) -> list[RankedPath]:
    out = []
    for p in paths:
        ev = [{"src": u, "dst": v, "service": g[u][v]["service"],
               "relation": g[u][v]["relation"],
               "exploitability": g[u][v]["exploitability"]} for u, v in zip(p, p[1:])]
        prob = math.exp(-sum(g[u][v]["cost"] for u, v in zip(p, p[1:])))
        out.append(RankedPath(path=list(p), probability=prob, hops=len(p) - 1, evidence=ev))
    return out


def _k_paths(g: nx.DiGraph, source: str, target: str, k: int, weight: str,
             max_hops: int | None) -> list[list[str]]:
    if source not in g or target not in g:
        return []
    try:
        gen = nx.shortest_simple_paths(g, source, target, weight=weight)
        result = []
        for p in islice(gen, k * 5):
            if max_hops is None or len(p) - 1 <= max_hops:
                result.append(p)
            if len(result) == k:
                break
        return result
    except nx.NetworkXNoPath:
        return []


def hop_shortest_paths(g: nx.DiGraph, source: str, target: str, k: int = 5,
                       max_hops: int | None = None) -> list[RankedPath]:
    return _rank(g, _k_paths(g, source, target, k, "hops", max_hops))


def risk_weighted_paths(g: nx.DiGraph, source: str, target: str, k: int = 5,
                        max_hops: int | None = None) -> list[RankedPath]:
    return _rank(g, _k_paths(g, source, target, k, "cost", max_hops))


def blast_radius(g: nx.DiGraph, source: str, max_hops: int = 4,
                 min_probability: float = 0.0) -> dict:
    """Assets reachable from ``source`` within ``max_hops`` and their best-path probability."""
    lengths = nx.single_source_dijkstra_path_length(g, source, weight="cost")
    hops = nx.single_source_shortest_path_length(g, source, cutoff=max_hops)
    reach = []
    for node, cost in lengths.items():
        if node == source or node not in hops:
            continue
        p = math.exp(-cost)
        if p >= min_probability:
            reach.append({"asset": node, "probability": round(p, 6), "hops": hops[node],
                          "criticality": g.nodes[node]["criticality"]})
    reach.sort(key=lambda r: (-r["probability"] * r["criticality"], r["asset"]))
    return {
        "source": source,
        "reachable": reach,
        "reachable_count": len(reach),
        "expected_critical_exposure": round(sum(r["probability"] * r["criticality"]
                                                for r in reach), 4),
    }


def choke_points(g: nx.DiGraph, sources: list[str], targets: list[str],
                 top_n: int = 5) -> list[dict]:
    """Rank edges by how much entry->target risk flows through them.

    Uses edge betweenness restricted to (sources, targets) on the cost-weighted
    graph, then reports the drop in best-path probability if the edge is blocked.
    """
    if not sources or not targets:
        return []
    bc = nx.edge_betweenness_centrality_subset(g, sources, targets, weight="cost",
                                               normalized=True)
    ranked = sorted(bc.items(), key=lambda kv: -kv[1])[: top_n * 3]

    def total_best(gr: nx.DiGraph) -> float:
        s = 0.0
        for src in sources:
            lengths = nx.single_source_dijkstra_path_length(gr, src, weight="cost")
            for t in targets:
                if t in lengths:
                    s += math.exp(-lengths[t]) * gr.nodes[t]["criticality"]
        return s

    base = total_best(g)
    out = []
    for (u, v), score in ranked:
        h = g.copy()
        h.remove_edge(u, v)
        after = total_best(h)
        out.append({"edge": [u, v], "service": g[u][v]["service"], "betweenness": round(score, 5),
                    "risk_before": round(base, 6), "risk_after": round(after, 6),
                    "risk_reduction_pct": round(100 * (base - after) / base, 2) if base else 0.0})
    out.sort(key=lambda r: (-r["risk_reduction_pct"], -r["betweenness"]))
    return out[:top_n]


def minimum_cut(g: nx.DiGraph, sources: list[str], targets: list[str]) -> list[tuple[str, str]]:
    """Smallest set of edges that disconnects all sources from all targets."""
    h = g.copy()
    h.add_node("__S__")
    h.add_node("__T__")
    for s in sources:
        h.add_edge("__S__", s, capacity=float("inf"))
    for t in targets:
        h.add_edge(t, "__T__", capacity=float("inf"))
    for u, v in g.edges:
        h[u][v]["capacity"] = 1
    _, (reach, non_reach) = nx.minimum_cut(h, "__S__", "__T__")
    return sorted((u, v) for u, v in g.edges if u in reach and v in non_reach)
