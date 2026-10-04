"""Alert evidence on the attack graph.

Links the detector (Sprint 3) to the graph (Sprint 4): every telemetry event
of a scenario passes through a *simulated* detector with a given recall
(attack events flagged) and false-positive rate (benign events flagged). The
flagged events are aggregated per graph edge ``(src, dst)``.

Recall / FPR should come from measured detector results, e.g. the Sprint 3
CICIDS2017 report, so that graph experiments inherit realistic alert noise.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx


@dataclass
class EdgeEvidence:
    alerts: dict[tuple[str, str], int] = field(default_factory=dict)
    events: dict[tuple[str, str], int] = field(default_factory=dict)
    recall: float = 0.0
    fpr: float = 0.0

    def alert_count(self, u: str, v: str) -> int:
        return self.alerts.get((u, v), 0)

    def event_count(self, u: str, v: str) -> int:
        return self.events.get((u, v), 0)

    def alert_rate(self, u: str, v: str) -> float:
        n = self.event_count(u, v)
        return self.alert_count(u, v) / n if n else 0.0

    def outgoing_alerts(self, g: nx.DiGraph, v: str) -> int:
        return sum(self.alert_count(v, w) for w in g.successors(v))

    @property
    def total_alerts(self) -> int:
        return sum(self.alerts.values())


def simulate_alerts(events: list[dict], recall: float, fpr: float, seed: int) -> EdgeEvidence:
    if not (0 <= recall <= 1 and 0 <= fpr <= 1):
        raise ValueError("recall and fpr must be within [0, 1]")
    rng = random.Random(seed)
    alerts: dict = defaultdict(int)
    counts: dict = defaultdict(int)
    for e in events:
        key = (e["src"], e["dst"])
        counts[key] += 1
        p = recall if e["label"] == "attack" else fpr
        if rng.random() < p:
            alerts[key] += 1
    return EdgeEvidence(alerts=dict(alerts), events=dict(counts), recall=recall, fpr=fpr)


def alert_weighted_graph(g: nx.DiGraph, ev: EdgeEvidence, alpha: float = 1.0) -> nx.DiGraph:
    """Heuristic baseline: edges with alerts become 'cheaper' for the attacker.

    cost' = cost / (1 + alpha * log1p(alerts)); stays non-negative for Dijkstra.
    """
    h = g.copy()
    for u, v, d in h.edges(data=True):
        a = ev.alert_count(u, v)
        d["cost"] = d["cost"] / (1.0 + alpha * math.log1p(a))
    return h
