import math

import networkx as nx
import pytest

from aegismind.graph import metrics
from aegismind.graph.paths import (blast_radius, choke_points, hop_shortest_paths,
                                   minimum_cut, risk_weighted_paths)
from aegismind.twin.generator import GeneratorConfig, Scenario, generate_scenario
from aegismind.twin.topology import Asset, Edge, Topology
from aegismind.twin.whatif import Action, apply, compare


def tiny_topology() -> Topology:
    """PC -> APP -> DB (easy) and PC -> FILE -> DB (hard)."""
    t = Topology("tiny")
    t.add_asset(Asset("PC", "workstation", "user", 0.2, 0.6))
    t.add_asset(Asset("APP", "app_server", "server", 0.6, 0.5))
    t.add_asset(Asset("FILE", "file_server", "server", 0.5, 0.5))
    t.add_asset(Asset("DB", "db_server", "data", 1.0, 0.3))
    t.add_edge(Edge("PC", "APP", "network", "http", 443, 0.5))
    t.add_edge(Edge("APP", "DB", "network", "sql", 5432, 0.5))
    t.add_edge(Edge("PC", "FILE", "network", "smb", 445, 0.2))
    t.add_edge(Edge("FILE", "DB", "network", "sql", 5432, 0.2))
    return t


# ---------------------------------------------------------------- topology
def test_topology_roundtrip(tmp_path):
    t = tiny_topology()
    f = tmp_path / "t.json"
    t.save(f)
    t2 = Topology.load(f)
    assert t2.to_dict() == t.to_dict()


def test_invalid_inputs_rejected():
    t = tiny_topology()
    with pytest.raises(ValueError):
        t.add_edge(Edge("PC", "NOPE", "network", "http", 443, 0.5))
    with pytest.raises(ValueError):
        Edge("PC", "APP", "network", "http", 443, 0.0)
    with pytest.raises(ValueError):
        Asset("X", "toaster", "user", 0.1, 0.1)


def test_parallel_edges_keep_easiest():
    t = tiny_topology()
    t.add_edge(Edge("PC", "APP", "credential", "cached-admin", None, 0.9))
    g = t.to_networkx()
    assert g["PC"]["APP"]["service"] == "cached-admin"
    assert "http" in g["PC"]["APP"]["alternatives"]


# ------------------------------------------------------------------- paths
def test_risk_weighted_prefers_probable_path():
    g = tiny_topology().to_networkx()
    best = risk_weighted_paths(g, "PC", "DB", k=2)
    assert best[0].path == ["PC", "APP", "DB"]
    assert math.isclose(best[0].probability, 0.25, rel_tol=1e-9)
    assert len(best) == 2 and best[1].probability < best[0].probability


def test_hop_baseline_returns_two_hop_paths():
    g = tiny_topology().to_networkx()
    paths = hop_shortest_paths(g, "PC", "DB", k=5)
    assert {tuple(p.path) for p in paths} == {("PC", "APP", "DB"), ("PC", "FILE", "DB")}


def test_no_path_returns_empty():
    g = tiny_topology().to_networkx()
    assert risk_weighted_paths(g, "DB", "PC") == []


def test_blast_radius():
    br = blast_radius(tiny_topology().to_networkx(), "PC")
    assert br["reachable_count"] == 3
    assert br["reachable"][0]["asset"] in {"APP", "DB"}


def test_choke_points_and_min_cut():
    g = tiny_topology().to_networkx()
    cps = choke_points(g, ["PC"], ["DB"], top_n=2)
    assert cps[0]["edge"] in (["PC", "APP"], ["APP", "DB"])
    assert cps[0]["risk_reduction_pct"] > 0
    cut = minimum_cut(g, ["PC"], ["DB"])
    h = g.copy()
    h.remove_edges_from(cut)
    assert not nx.has_path(h, "PC", "DB")
    assert len(cut) == 2


# ------------------------------------------------------------------ whatif
def test_whatif_block_edge_reduces_risk():
    t = tiny_topology()
    res = compare(t, Action("block_edge", src="APP", dst="DB"), "PC", "DB")
    assert res["best_path_probability_after"] < res["best_path_probability_before"]
    assert res["risk_reduction_pct"] > 0
    assert all(e.allowed for e in t.edges), "original topology must not be modified"


def test_isolate_asset():
    t2 = apply(tiny_topology(), Action("isolate_asset", asset="DB"))
    g = t2.to_networkx()
    assert g.in_degree("DB") == 0


# ----------------------------------------------------------------- metrics
def test_metrics():
    truth = ["A", "B", "C"]
    preds = [["A", "X", "C"], ["A", "B", "C"]]
    assert metrics.topk_hit(preds, truth, 1) == 0.0
    assert metrics.topk_hit(preds, truth, 2) == 1.0
    assert metrics.reciprocal_rank(preds, truth) == 0.5
    p, r, f1 = metrics.edge_precision_recall(["A", "B", "X"], truth)
    assert (p, r) == (0.5, 0.5) and f1 == 0.5
    assert metrics.next_hop_accuracy(["A", "B", "C"], truth, observed_steps=0) == 1.0


# --------------------------------------------------------------- generator
def test_generator_is_deterministic():
    a, b = generate_scenario(7), generate_scenario(7)
    assert a.topology.to_dict() == b.topology.to_dict()
    assert a.campaign["path"] == b.campaign["path"]
    assert a.events == b.events


def test_generated_scenario_is_consistent(tmp_path):
    s = generate_scenario(3)
    path = s.campaign["path"]
    g = s.topology.to_networkx()
    assert path[0] == s.campaign["entry"] and path[-1] == s.campaign["target"]
    assert all(g.has_edge(u, v) for u, v in zip(path, path[1:]))
    assert len(path) - 1 <= GeneratorConfig().max_hops
    assert s.topology.assets[path[-1]].criticality >= 0.85
    # no direct user-zone -> data-zone network edges (firewall policy)
    for e in s.topology.edges:
        if e.relation == "network":
            assert not (s.topology.assets[e.src].zone == "user"
                        and s.topology.assets[e.dst].zone == "data")
    attack_steps = {r["campaign_step"] for r in s.events if r["label"] == "attack"}
    assert attack_steps == set(range(1, len(path)))
    # defender view: same structure, different numbers
    dv = s.defender_view()
    assert [e.key for e in dv.edges] == [e.key for e in s.topology.edges]
    # save / load
    d = s.save(tmp_path)
    s2 = Scenario.load(d)
    assert s2.campaign["path"] == path
    assert (d / "events.csv").read_text().startswith("ts,event_type")


def test_steps_tagged_with_attack_ids():
    s = generate_scenario(11)
    assert all(st["technique_id"].startswith("T") for st in s.campaign["steps"])
    assert s.campaign["steps"][0]["tactic"] == "Initial Access"
