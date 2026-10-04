import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from aegismind.cli import main
from aegismind.defense.feedback import FeedbackStore
from aegismind.defense.optimizer import (DefenseAction, OptimizerConfig, apply_action,
                                         check_business_constraints, evaluate_action,
                                         incident_risk, recommend, risk_graph)
from aegismind.explain import explain_detector, explain_path, path_summary
from aegismind.graph.evidence import EdgeEvidence, simulate_alerts
from aegismind.graph.paths import risk_weighted_paths
from aegismind.twin.generator import generate_scenario
from aegismind.twin.topology import Asset, Edge, Topology


def small_enterprise() -> Topology:
    """PC-01 -(smb)-> PC-02 -(cached-admin)-> APP-01 -(sql)-> DB-01, plus business links."""
    t = Topology("small")
    for a in [Asset("PC-01", "workstation", "user", 0.2, 0.6),
              Asset("PC-02", "workstation", "user", 0.2, 0.6),
              Asset("APP-01", "app_server", "server", 0.6, 0.5),
              Asset("DC-01", "domain_controller", "server", 0.9, 0.2),
              Asset("DB-01", "db_server", "data", 1.0, 0.3)]:
        t.add_asset(a)
    for e in [Edge("PC-01", "PC-02", "network", "smb", 445, 0.5),
              Edge("PC-02", "APP-01", "credential", "cached-admin", None, 0.8),
              Edge("PC-01", "APP-01", "network", "http", 443, 0.2),
              Edge("PC-02", "APP-01", "network", "http", 443, 0.2),
              Edge("PC-01", "DC-01", "network", "ldap", 389, 0.1),
              Edge("APP-01", "DB-01", "network", "sql", 5432, 0.4),
              Edge("APP-01", "DB-01", "credential", "service-account", None, 0.6)]:
        t.add_edge(e)
    return t


CFG = OptimizerConfig(mc_samples=5)


# ------------------------------------------------------------ apply_action
def test_apply_actions_do_not_touch_original():
    t = small_enterprise()
    t2, n = apply_action(t, DefenseAction("reset_credentials", src="PC-02"))
    assert n == 1 and all(e.allowed for e in t.edges)
    assert not any(e.allowed and e.service == "cached-admin" for e in t2.edges)
    t3, n = apply_action(t, DefenseAction("require_mfa", dst="DB-01"))
    sa = [e for e in t3.edges if e.service == "service-account"][0]
    assert n == 1 and sa.allowed and sa.exploitability == pytest.approx(0.12)
    t4, n = apply_action(t, DefenseAction("restrict_service", dst="APP-01", service="http"))
    assert n == 2
    t5, n = apply_action(t, DefenseAction("isolate_asset", asset="PC-02"))
    assert n == 3


def test_business_constraints():
    t = small_enterprise()
    assert check_business_constraints(t, set()) is None
    t2, _ = apply_action(t, DefenseAction("restrict_service", dst="APP-01", service="http"))
    assert "HTTP" in check_business_constraints(t2, set())
    t3, _ = apply_action(t, DefenseAction("restrict_service", dst="DB-01", service="sql"))
    assert "SQL" in check_business_constraints(t3, set())
    t4, _ = apply_action(t, DefenseAction("isolate_asset", asset="PC-02"))
    assert "protected" in check_business_constraints(t4, {"PC-02"})


# ---------------------------------------------------------------- evaluate
def test_misconfiguration_fix_has_no_business_disruption():
    t = small_enterprise()
    ev = evaluate_action(t, DefenseAction("reset_credentials", src="PC-02"), "PC-01",
                         ["DB-01"], {"DB-01"}, CFG)
    assert ev.feasible and ev.risk_reduction > 0 and ev.service_disruption == 0
    assert "T1078.002" in ev.attack_techniques_mitigated


def test_infeasible_actions_rejected_with_reason():
    t = small_enterprise()
    ev = evaluate_action(t, DefenseAction("isolate_asset", asset="DB-01"), "PC-01",
                         ["DB-01"], {"DB-01"}, CFG)
    assert not ev.feasible and ev.utility == float("-inf") and ev.rejected_reason
    ev = evaluate_action(t, DefenseAction("reset_credentials", src="PC-01"), "PC-01",
                         ["DB-01"], {"DB-01"}, CFG)
    assert not ev.feasible and "no effect" in ev.rejected_reason


def test_evidence_raises_risk_on_alerted_edges():
    t = small_enterprise()
    plain = incident_risk(risk_graph(t), "PC-01", ["DB-01"])
    alerted = incident_risk(risk_graph(t, EdgeEvidence(alerts={("PC-01", "PC-02"): 5})),
                            "PC-01", ["DB-01"])
    assert alerted > plain


# --------------------------------------------------------------- recommend
def test_recommend_plan_is_feasible_and_reduces_risk():
    t = small_enterprise()
    paths = [p.path for p in risk_weighted_paths(t.to_networkx(), "PC-01", "DB-01", k=3)]
    rec = recommend(t, "PC-01", paths, CFG)
    ps = rec["plan_summary"]
    assert rec["plan"] and ps["risk_after"] < ps["risk_before"]
    assert all(a["feasible"] for a in rec["plan"])
    assert "DB-01" in rec["protected_assets"] and "DC-01" in rec["protected_assets"]
    utils = [a["utility"] for a in rec["ranked_actions"] if a["feasible"]]
    assert utils == sorted(utils, reverse=True)
    assert rec["note"].startswith("Simulated")


def test_preference_penalty_changes_ranking():
    t = small_enterprise()
    paths = [p.path for p in risk_weighted_paths(t.to_networkx(), "PC-01", "DB-01", k=3)]
    base = recommend(t, "PC-01", paths, CFG)
    top_kind = base["ranked_actions"][0]["action"]["kind"]
    pen = recommend(t, "PC-01", paths, CFG, preference_penalty={top_kind: 5.0})
    assert pen["ranked_actions"][0]["action"]["kind"] != top_kind


def test_recommend_on_generated_scenario_respects_constraints():
    sc = generate_scenario(3)
    t = sc.defender_view()
    ev = simulate_alerts(sc.events, 0.6, 0.01, seed=1)
    e, tg = sc.campaign["entry"], sc.campaign["target"]
    paths = [p.path for p in risk_weighted_paths(t.to_networkx(), e, tg, k=3)]
    rec = recommend(t, e, paths, OptimizerConfig(mc_samples=3, max_actions=2), evidence=ev)
    assert rec["uses_alert_evidence"]
    for a in rec["plan"]:
        t, _ = apply_action(t, DefenseAction(**{k: v for k, v in a["action"].items()
                                                 if k in ("kind", "src", "dst", "asset", "service")}))
    assert check_business_constraints(t, set(rec["protected_assets"])) is None


# ---------------------------------------------------------------- feedback
def test_feedback_store(tmp_path):
    fs = FeedbackStore(tmp_path / "fb.jsonl", rho=0.2)
    assert fs.preference_penalty() == {}
    act = {"kind": "isolate_asset", "description": "Isolate PC-01"}
    for _ in range(3):
        fs.record("scn-1", act, "reject")
    fs.record("scn-1", {"kind": "require_mfa"}, "approve")
    pen = fs.preference_penalty()
    assert pen["isolate_asset"] > 0 > pen["require_mfa"]
    assert len(fs.entries()) == 4
    with pytest.raises(ValueError):
        fs.record("scn-1", act, "maybe")


# ----------------------------------------------------------------- explain
def test_explain_path():
    t = small_enterprise()
    g = t.to_networkx()
    ev = EdgeEvidence(alerts={("PC-01", "PC-02"): 2}, events={("PC-01", "PC-02"): 4})
    steps = explain_path(g, ["PC-01", "PC-02", "APP-01", "DB-01"], ev)
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert steps[0]["alerts"] == 2 and "2 detector alert(s)" in steps[0]["reasons"][2]
    assert "no detector alerts" in steps[1]["reasons"][2]
    assert any("crown-jewel" in r for r in steps[2]["reasons"])
    assert "1 of 3 steps have detector alerts" in path_summary(steps)


def test_explain_path_with_learned_model_impact():
    from aegismind.graph.learned import GraphContextPathModel
    from aegismind.graph.pathbench import _load
    from aegismind.twin.generator import GeneratorConfig
    cfg = GeneratorConfig()
    train = _load(range(3000, 3030), cfg, 0.9, 0.0)
    m = GraphContextPathModel(use_evidence=True).fit(train)
    g, path, ev = train[0]
    steps = explain_path(g, path, ev, model=m)
    assert all("evidence_impact_logp" in s for s in steps)


@pytest.mark.parametrize("model_cls", [LogisticRegression, DecisionTreeClassifier])
def test_explain_detector_occlusion(model_cls):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4)).astype(np.float32)
    y = (X[:, 2] > 0.3).astype(int)  # only feature f2 matters
    m = model_cls().fit(X, y)
    out = explain_detector(m, X, X[y == 1][:5], ["f0", "f1", "f2", "f3"], method="occlusion")
    assert out["method"] == "occlusion"
    assert out["global"][0]["feature"] == "f2"
    assert len(out["rows"]) == 5 and out["rows"][0][0]["feature"] == "f2"


def test_explain_detector_shap_if_installed():
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (X[:, 1] > 0).astype(int)
    m = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    out = explain_detector(m, X[:50], X[y == 1][:3], ["a", "b", "c", "d"], method="shap")
    assert out["method"] == "shap" and out["global"][0]["feature"] == "b"


def test_explain_detector_bad_method():
    with pytest.raises(ValueError):
        explain_detector(LogisticRegression().fit([[0], [1]], [0, 1]), np.zeros((2, 1)),
                         np.zeros((1, 1)), ["x"], method="magic")


# --------------------------------------------------------------------- cli
def test_cli_recommend_and_feedback(tmp_path, capsys):
    main(["generate", "--count", "1", "--out", str(tmp_path / "scn")])
    scn = str(tmp_path / "scn" / "scn-0001")
    fb = str(tmp_path / "fb.jsonl")
    main(["recommend", scn, "--out", str(tmp_path), "--feedback-file", fb])
    out = capsys.readouterr().out
    assert "Most likely path" in out and "Ranked actions" in out and "requires human approval" in out
    rec = json.loads((tmp_path / "recommend_scn-0001.json").read_text())
    assert rec["feasible_actions"] and rec["path_explanation"]
    main(["feedback", scn, "--action", "1", "--decision", "reject", "--out", str(tmp_path),
          "--feedback-file", fb])
    assert "Recorded: reject" in capsys.readouterr().out
    main(["recommend", scn, "--out", str(tmp_path), "--feedback-file", fb])
    assert "Analyst feedback applied" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["feedback", scn, "--action", "999", "--decision", "approve", "--out", str(tmp_path),
              "--feedback-file", fb])


def test_cli_detect_explain(tmp_path, capsys):
    from .fixtures import write_cicids
    raw = write_cicids(tmp_path / "raw")
    proc = tmp_path / "proc"
    main(["prepare", "cicids2017", "--raw", str(raw), "--out", str(proc)])
    main(["detect", "cicids2017", "--data", str(proc), "--models", "random_forest",
          "--explain", "3", "--out", str(tmp_path / "exp")])
    assert "Why random_forest flags flows as attacks" in capsys.readouterr().out
    rep = json.loads(next((tmp_path / "exp").glob("detectors_*.json")).read_text())
    assert len(rep["results"][0]["explanation"]["explained_rows"]) == 3
