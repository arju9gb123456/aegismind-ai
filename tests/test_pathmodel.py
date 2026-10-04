import math

import networkx as nx
import pytest

from aegismind.cli import main
from aegismind.graph import pathbench
from aegismind.graph.evidence import EdgeEvidence, alert_weighted_graph, simulate_alerts
from aegismind.graph.learned import (EVIDENCE_FEATURES, GRAPH_FEATURES, GraphContextPathModel,
                                     candidates, decision_rows, edge_features, make_context)
from aegismind.twin.generator import GeneratorConfig, generate_scenario


def tiny_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for n, z, c in [("PC", "user", 0.2), ("APP", "server", 0.6), ("FILE", "server", 0.5),
                    ("DB", "data", 1.0)]:
        g.add_node(n, zone=z, criticality=c, exposure=0.5, kind="x")
    for u, v, e in [("PC", "APP", 0.5), ("APP", "DB", 0.5), ("PC", "FILE", 0.2), ("FILE", "DB", 0.2)]:
        g.add_edge(u, v, exploitability=e, cost=-math.log(e), relation="network", service="x")
    return g


# ---------------------------------------------------------------- evidence
def test_simulate_alerts_extremes():
    events = [{"src": "A", "dst": "B", "label": "attack"}] * 10 + \
             [{"src": "B", "dst": "C", "label": "benign"}] * 10
    ev = simulate_alerts(events, recall=1.0, fpr=0.0, seed=0)
    assert ev.alert_count("A", "B") == 10 and ev.alert_count("B", "C") == 0
    assert ev.event_count("B", "C") == 10 and ev.alert_rate("A", "B") == 1.0
    ev = simulate_alerts(events, recall=0.0, fpr=1.0, seed=0)
    assert ev.alert_count("A", "B") == 0 and ev.alert_count("B", "C") == 10
    with pytest.raises(ValueError):
        simulate_alerts(events, recall=1.5, fpr=0, seed=0)


def test_simulate_alerts_deterministic_and_rate():
    events = [{"src": "A", "dst": "B", "label": "attack"}] * 2000
    a = simulate_alerts(events, 0.3, 0.0, seed=5)
    b = simulate_alerts(events, 0.3, 0.0, seed=5)
    assert a.alerts == b.alerts
    assert 0.25 < a.alert_rate("A", "B") < 0.35


def test_alert_weighted_graph_lowers_cost_only_where_alerts():
    g = tiny_graph()
    ev = EdgeEvidence(alerts={("PC", "FILE"): 5})
    h = alert_weighted_graph(g, ev)
    assert h["PC"]["FILE"]["cost"] < g["PC"]["FILE"]["cost"]
    assert h["PC"]["APP"]["cost"] == g["PC"]["APP"]["cost"]
    assert all(d["cost"] >= 0 for *_, d in h.edges(data=True))


# ---------------------------------------------------------------- features
def test_feature_vector_lengths_and_context():
    g = tiny_graph()
    ctx = make_context(g, "DB", EdgeEvidence(alerts={("PC", "FILE"): 3}, events={("PC", "FILE"): 3}))
    assert ctx.hops_to["PC"] == 2 and ctx.hops_to["DB"] == 0
    f0 = edge_features(ctx, "PC", "FILE", 0, use_evidence=False)
    f1 = edge_features(ctx, "PC", "FILE", 0, use_evidence=True)
    assert len(f0) == len(GRAPH_FEATURES)
    assert len(f1) == len(GRAPH_FEATURES) + len(EVIDENCE_FEATURES)
    assert f1[GRAPH_FEATURES.__len__() + 2] == 1.0  # edge_has_alert
    assert f0[GRAPH_FEATURES.index("hop_progress")] == 1.0


def test_candidates_skip_visited_and_dead_ends():
    g = tiny_graph()
    g.add_node("DEAD", zone="user", criticality=0.1, exposure=0.1, kind="x")
    g.add_edge("PC", "DEAD", exploitability=0.9, cost=0.1, relation="network", service="x")
    ctx = make_context(g, "DB", None)
    assert set(candidates(ctx, "PC", {"PC"})) == {"APP", "FILE"}
    assert candidates(ctx, "PC", {"PC", "APP", "FILE"}) == []


def test_decision_rows_one_positive_per_step():
    g = tiny_graph()
    X, y, groups = decision_rows(make_context(g, "DB", None), ["PC", "APP", "DB"], False)
    assert sum(y) == 2 and len(X) == len(y) == len(groups)


# ---------------------------------------------------------------- model
@pytest.fixture(scope="module")
def trained():
    cfg = GeneratorConfig()
    models, train = pathbench.train_models(cfg, range(2000, 2060), recall=0.8, fpr=0.01)
    return cfg, models


def test_model_paths_are_valid(trained):
    cfg, models = trained
    sc = generate_scenario(5, cfg)
    g = sc.defender_view().to_networkx()
    ev = simulate_alerts(sc.events, 0.8, 0.01, seed=1)
    s, t = sc.campaign["entry"], sc.campaign["target"]
    for m in models:
        paths = m.predict_paths(g, s, t, ev, k=5)
        assert paths, m.name
        lps = [lp for _, lp in paths]
        assert lps == sorted(lps, reverse=True)
        for p, lp in paths:
            assert p[0] == s and p[-1] == t and len(set(p)) == len(p)
            assert len(p) - 1 <= cfg.max_hops and lp <= 0
            assert all(g.has_edge(u, v) for u, v in zip(p, p[1:]))
        assert len({tuple(p) for p, _ in paths}) == len(paths)


def test_unknown_or_unreachable_nodes(trained):
    _, models = trained
    g = tiny_graph()
    assert models[0].predict_paths(g, "NOPE", "DB") == []
    assert models[0].predict_paths(g, "DB", "PC") == []


def test_evidence_model_beats_heuristic_with_strong_alerts(trained):
    cfg, models = trained
    test = pathbench._load(range(1, 41), cfg, recall=0.8, fpr=0.01)
    res = pathbench.evaluate(models, test, cfg)
    assert res["graph_context+evidence"]["mrr"] > res["risk_weighted"]["mrr"]
    assert set(res) == set(pathbench.METHODS)


# ------------------------------------------------------------------ cli
def test_cli_pathbench(tmp_path, capsys):
    main(["pathbench", "--count", "10", "--train-count", "30", "--test-alerts", "0.6:0.01,0:0",
          "--no-importance", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "graph_context+evidence" in out and "recall=0.0" in out
    assert len(list(tmp_path.glob("pathbench_*.json"))) == 1


def test_cli_pathbench_rejects_seed_overlap(tmp_path):
    with pytest.raises(SystemExit):
        main(["pathbench", "--count", "10", "--seed", "1", "--train-seed", "5",
              "--out", str(tmp_path)])
