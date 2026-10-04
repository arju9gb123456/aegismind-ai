import json

import pytest
from fastapi.testclient import TestClient

import aegismind.api.app as api
from aegismind.twin.generator import generate_scenario


@pytest.fixture()
def client(tmp_path, monkeypatch):
    scn, exp = tmp_path / "scenarios", tmp_path / "experiments"
    monkeypatch.setattr(api, "SCENARIO_DIR", scn)
    monkeypatch.setattr(api, "EXPERIMENT_DIR", exp)
    for seed in (1, 2):
        generate_scenario(seed).save(scn)
    api._load.cache_clear()
    return TestClient(api.app)


def test_health_and_list(client):
    assert client.get("/api/health").json()["status"] == "ok"
    sc = client.get("/api/scenarios").json()
    assert [s["id"] for s in sc] == ["scn-0001", "scn-0002"]
    assert sc[0]["entry"] and sc[0]["assets"] > 0


def test_auto_creates_demo_scenarios(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "SCENARIO_DIR", tmp_path / "empty")
    api._load.cache_clear()
    c = TestClient(api.app)
    assert len(c.get("/api/scenarios").json()) == 12


def test_analysis(client):
    r = client.get("/api/scenarios/scn-0001/analysis", params={"recall": 0.6, "fpr": 0.01}).json()
    ids = {n["id"] for n in r["nodes"]}
    assert all(l["source"] in ids and l["target"] in ids for l in r["links"])
    assert r["paths"] and r["paths"][0]["path"][0] == r["scenario"]["entry"]
    assert r["paths"][0]["steps"][0]["technique_id"].startswith("T")
    k = r["kpis"]
    assert k["threat_level"] in ("Low", "Guarded", "Elevated", "Critical")
    assert 0 <= k["risk_score"] <= 1 and k["alerts"] > 0
    entry = [n for n in r["nodes"] if n["is_entry"]][0]
    assert entry["reach_probability"] == 1.0 and entry["path_rank"] == 1
    assert any(l["path_rank"] == 1 for l in r["links"])
    assert r["ground_truth"]["path"][-1] == r["scenario"]["target"]


def test_analysis_without_alerts_has_lower_risk(client):
    a = client.get("/api/scenarios/scn-0001/analysis", params={"recall": 0.9, "fpr": 0.0}).json()
    b = client.get("/api/scenarios/scn-0001/analysis", params={"recall": 0.0, "fpr": 0.0}).json()
    assert b["kpis"]["alerts"] == 0
    assert a["kpis"]["risk_score"] >= b["kpis"]["risk_score"]


def test_unknown_scenario(client):
    assert client.get("/api/scenarios/scn-9999/analysis").status_code == 404
    assert client.get("/api/scenarios/../etc/analysis").status_code == 404


def test_recommend_and_feedback_roundtrip(client):
    r = client.post("/api/scenarios/scn-0001/recommend", json={"recall": 0.6, "fpr": 0.01}).json()
    assert r["actions"] and r["plan_summary"]["risk_after"] <= r["plan_summary"]["risk_before"]
    top = r["actions"][0]["action"]
    body = {"scenario": "scn-0001", "decision": "reject", "description": top["description"],
            "action": {k: top[k] for k in ("kind", "src", "dst", "asset", "service")}}
    fb = client.post("/api/feedback", json=body).json()
    assert fb["preference_penalty"][top["kind"]] > 0
    r2 = client.post("/api/scenarios/scn-0001/recommend", json={}).json()
    assert r2["preference_penalty"][top["kind"]] > 0
    assert len(client.get("/api/feedback").json()["entries"]) == 1
    bad = dict(body, decision="maybe")
    assert client.post("/api/feedback", json=bad).status_code == 422


def test_recommend_protect(client):
    r = client.post("/api/scenarios/scn-0001/recommend", json={"protect": ["PC-16"]}).json()
    assert "PC-16" in r["protected_assets"]
    assert not any(a["action"]["kind"] == "isolate_asset" and a["action"]["asset"] == "PC-16"
                   for a in r["actions"])


def test_whatif(client):
    an = client.get("/api/scenarios/scn-0001/analysis").json()
    u, v = an["paths"][0]["path"][:2]
    r = client.post("/api/scenarios/scn-0001/whatif",
                    json={"action": {"kind": "block_edge", "src": u, "dst": v}}).json()
    assert r["changed_edges"] and r["changed_edges"][0]["change"] == "blocked"
    assert r["evaluation"]["risk_after"] <= r["evaluation"]["risk_before"]
    assert client.post("/api/scenarios/scn-0001/whatif",
                       json={"action": {"kind": "explode"}}).status_code == 422


def test_experiments(client, tmp_path):
    exp = api.EXPERIMENT_DIR
    exp.mkdir(parents=True, exist_ok=True)
    assert client.get("/api/experiments").json()["detectors"] == []
    rep = {"dataset": "cicids2017", "split": "day", "date_utc": "x", "results": [
        {"model": "logreg", "test": {"precision": 1, "f1": 0.5},
         "test_per_family": {"Benign": {"flagged_rate": 0.1}}}]}
    (exp / "detectors_cicids2017_1.json").write_text(json.dumps(rep))
    (exp / "pathbench_1.json").write_text(json.dumps({"runs": [], "train_alerts": {}}))
    out = client.get("/api/experiments").json()
    assert out["detectors"][0]["models"][0]["strategies"]["val_f1"]["metrics"]["f1"] == 0.5
    assert out["pathbench"]["runs"] == []


def test_spa_fallback_does_not_swallow_api_or_leak_files(client):
    if not api.FRONTEND_DIST.exists():
        pytest.skip("frontend not built")
    assert client.get("/api/does-not-exist").status_code == 404
    r = client.get("/some/client/route")
    assert r.status_code == 200 and "<div id=\"root\">" in r.text
    r = client.get("/..%2F..%2Fpyproject.toml")
    assert "aegismind" not in r.text.lower() or "<div id=\"root\">" in r.text
