"""Graph-context attack-path model (Sprint 4, main research contribution).

Idea
----
Instead of ranking paths only by edge exploitability, learn *which next hop an
attacker takes* from the context around each candidate edge, then decode
whole paths with beam search.

For an attacker at ``u`` heading to target ``T`` every unvisited successor
``v`` is a candidate. Each candidate edge gets features in three groups:

* edge      : exploitability, relation type (network / credential / trust)
* graph     : destination zone, criticality, exposure, out-degree, hop and
              cost distance from ``v`` to ``T``, progress made towards ``T``,
              whether ``v`` is the target
* evidence  : detector alerts on ``u -> v``, alert rate, alerts on edges
              leaving ``v`` (only in the ``+evidence`` variant)

A gradient-boosted classifier is trained on synthetic campaigns (true next
hop = 1, other candidates = 0) from *training seeds only*. At test time the
per-candidate scores are turned into a softmax over the candidates of each
step, and beam search returns the top-k complete paths ``entry -> target``.

The evidence-free variant is the ablation that answers RQ1: does graph
context alone beat the exploitability-only baseline, and how much do alerts
add on top?
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .evidence import EdgeEvidence

ZONES = ("user", "dmz", "server", "data")
RELATIONS = ("network", "credential", "trust")

GRAPH_FEATURES = [
    "log_exploitability", "rel_network", "rel_credential", "rel_trust",
    "dst_criticality", "dst_exposure", "zone_user", "zone_dmz", "zone_server", "zone_data",
    "dst_is_target", "log_out_degree", "hops_to_target", "cost_to_target",
    "hop_progress", "cost_progress", "step_index",
]
EVIDENCE_FEATURES = ["log_edge_alerts", "edge_alert_rate", "edge_has_alert",
                     "log_downstream_alerts"]
UNREACHABLE = 99.0


@dataclass
class Context:
    """Per-scenario quantities shared by all decisions."""
    g: nx.DiGraph
    target: str
    hops_to: dict
    cost_to: dict
    evidence: EdgeEvidence | None


def make_context(g: nx.DiGraph, target: str, evidence: EdgeEvidence | None) -> Context:
    rev = g.reverse(copy=False)
    hops_to = nx.single_source_shortest_path_length(rev, target)
    cost_to = nx.single_source_dijkstra_path_length(rev, target, weight="cost")
    return Context(g=g, target=target, hops_to=hops_to, cost_to=cost_to, evidence=evidence)


def candidates(ctx: Context, u: str, visited: set[str]) -> list[str]:
    """Unvisited successors from which the target is still reachable."""
    return [v for v in ctx.g.successors(u) if v not in visited and v in ctx.hops_to]


def edge_features(ctx: Context, u: str, v: str, step: int, use_evidence: bool) -> list[float]:
    g, d, nd = ctx.g, ctx.g[u][v], ctx.g.nodes[v]
    hu = ctx.hops_to.get(u, UNREACHABLE)
    hv = ctx.hops_to.get(v, UNREACHABLE)
    cu = ctx.cost_to.get(u, UNREACHABLE)
    cv = ctx.cost_to.get(v, UNREACHABLE)
    f = [
        math.log(d["exploitability"]),
        *(1.0 if d["relation"] == r else 0.0 for r in RELATIONS),
        nd["criticality"], nd["exposure"],
        *(1.0 if nd["zone"] == z else 0.0 for z in ZONES),
        1.0 if v == ctx.target else 0.0,
        math.log1p(g.out_degree(v)),
        float(hv), float(cv),
        float(hu - hv), float(cu - cv),
        float(step),
    ]
    if use_evidence:
        ev = ctx.evidence
        a = ev.alert_count(u, v) if ev else 0
        f += [math.log1p(a), ev.alert_rate(u, v) if ev else 0.0, 1.0 if a else 0.0,
              math.log1p(ev.outgoing_alerts(g, v)) if ev else 0.0]
    return f


def decision_rows(ctx: Context, path: list[str], use_evidence: bool):
    """Training rows for every step of a ground-truth path."""
    X, y, groups = [], [], []
    visited = {path[0]}
    for step, (u, v_true) in enumerate(zip(path, path[1:])):
        cands = candidates(ctx, u, visited)
        if v_true not in cands:
            cands.append(v_true)  # keep the true hop even if pruned
        for v in cands:
            X.append(edge_features(ctx, u, v, step, use_evidence))
            y.append(1 if v == v_true else 0)
            groups.append(step)
        visited.add(v_true)
    return X, y, groups


class GraphContextPathModel:
    def __init__(self, use_evidence: bool = True, seed: int = 42, beam_width: int = 12,
                 max_hops: int = 6):
        self.use_evidence = use_evidence
        self.seed = seed
        self.beam_width = beam_width
        self.max_hops = max_hops
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                  max_leaf_nodes=31, random_state=seed)
        self.feature_names = GRAPH_FEATURES + (EVIDENCE_FEATURES if use_evidence else [])

    @property
    def name(self) -> str:
        return "graph_context+evidence" if self.use_evidence else "graph_context"

    # ------------------------------------------------------------- training
    def fit(self, training: list[tuple[nx.DiGraph, list[str], EdgeEvidence | None]]):
        X, y = [], []
        for g, path, ev in training:
            ctx = make_context(g, path[-1], ev)
            xs, ys, _ = decision_rows(ctx, path, self.use_evidence)
            X += xs
            y += ys
        self.n_train_rows_ = len(y)
        self.clf.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        return self

    # -------------------------------------------------------------- scoring
    def step_log_probs(self, ctx: Context, u: str, visited: set[str], step: int):
        cands = candidates(ctx, u, visited)
        if not cands:
            return []
        X = np.asarray([edge_features(ctx, u, v, step, self.use_evidence) for v in cands],
                       dtype=np.float32)
        p = np.clip(self.clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p))
        logits -= logits.max()
        logp = logits - math.log(np.exp(logits).sum())
        return list(zip(cands, logp.tolist()))

    def predict_paths(self, g: nx.DiGraph, source: str, target: str,
                      evidence: EdgeEvidence | None = None, k: int = 10) -> list[tuple[list[str], float]]:
        """Beam search; returns up to k (path, log_prob) pairs, best first."""
        if source not in g or target not in g:
            return []
        ctx = make_context(g, target, evidence if self.use_evidence else None)
        if source not in ctx.hops_to:
            return []
        beam = [(0.0, [source])]
        done: list[tuple[float, list[str]]] = []
        for step in range(self.max_hops):
            nxt = []
            for lp, path in beam:
                for v, l in self.step_log_probs(ctx, path[-1], set(path), step):
                    new = (lp + l, path + [v])
                    if v == target:
                        done.append(new)
                    elif step + 1 < self.max_hops and ctx.hops_to.get(v, UNREACHABLE) <= self.max_hops - step - 1:
                        nxt.append(new)
            if not nxt:
                break
            beam = heapq.nlargest(self.beam_width, nxt, key=lambda t: t[0])
        done.sort(key=lambda t: -t[0])
        seen, out = set(), []
        for lp, p in done:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                out.append((p, lp))
            if len(out) == k:
                break
        return out

    def feature_importance(self, training_sample, n_repeats: int = 3) -> list[dict]:
        """Permutation importance on next-hop decisions (for the paper / XAI)."""
        from sklearn.inspection import permutation_importance
        X, y = [], []
        for g, path, ev in training_sample:
            xs, ys, _ = decision_rows(make_context(g, path[-1], ev), path, self.use_evidence)
            X += xs
            y += ys
        r = permutation_importance(self.clf, np.asarray(X, np.float32), np.asarray(y),
                                   scoring="average_precision", n_repeats=n_repeats,
                                   random_state=self.seed)
        order = np.argsort(r.importances_mean)[::-1]
        return [{"feature": self.feature_names[i],
                 "importance": round(float(r.importances_mean[i]), 5)} for i in order]
