"""AegisMind REST API (Sprint 6).

Run:
    uvicorn aegismind.api.app:app --reload --port 8000

Endpoints (all JSON):
    GET  /api/health
    GET  /api/scenarios                         list scenario folders
    GET  /api/scenarios/{sid}/analysis          graph, alerts, predicted paths, KPIs
    POST /api/scenarios/{sid}/recommend         ranked defensive actions + plan
    POST /api/scenarios/{sid}/whatif            simulate one action on the twin
    POST /api/feedback                          approve / reject an action
    GET  /api/feedback                          feedback log + preference penalties
    GET  /api/experiments                       latest detector / path benchmark reports

If ``frontend/dist`` exists it is served at ``/``.
Everything is simulated on the digital twin; nothing touches a real network.
"""
from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aegismind import __version__
from aegismind.defense.feedback import FeedbackStore
from aegismind.defense.optimizer import (DefenseAction, OptimizerConfig, apply_action,
                                         evaluate_action, incident_risk, recommend, risk_graph)
from aegismind.explain import explain_path, path_summary
from aegismind.graph.evidence import alert_weighted_graph, simulate_alerts
from aegismind.graph.paths import blast_radius, risk_weighted_paths
from aegismind.twin.attack_kb import technique_for
from aegismind.twin.generator import Scenario, generate_scenario

ROOT = Path(os.environ.get("AEGISMIND_ROOT", Path.cwd()))
SCENARIO_DIR = Path(os.environ.get("AEGISMIND_SCENARIOS", ROOT / "data" / "scenarios"))
EXPERIMENT_DIR = Path(os.environ.get("AEGISMIND_EXPERIMENTS", ROOT / "experiments"))
FRONTEND_DIST = Path(os.environ.get("AEGISMIND_FRONTEND", ROOT / "frontend" / "dist"))

app = FastAPI(title="AegisMind AI", version=__version__,
              description="Attack-path prediction and digital-twin defense (research prototype)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ models
class DetectorParams(BaseModel):
    recall: float = Field(0.6, ge=0, le=1)
    fpr: float = Field(0.01, ge=0, le=1)


class RecommendRequest(DetectorParams):
    protect: list[str] = []
    lam: float = 0.6
    mu: float = 0.1
    nu: float = 0.5
    max_actions: int = Field(3, ge=1, le=6)


class ActionModel(BaseModel):
    kind: str
    src: str | None = None
    dst: str | None = None
    asset: str | None = None
    service: str | None = None


class WhatIfRequest(DetectorParams):
    action: ActionModel
    protect: list[str] = []


class FeedbackRequest(BaseModel):
    scenario: str
    action: ActionModel
    description: str | None = None
    decision: str
    note: str = ""


# ----------------------------------------------------------------- helpers
def ensure_scenarios(n: int = 12) -> None:
    """Create demo scenarios on first run so the dashboard works out of the box."""
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    if not any(SCENARIO_DIR.glob("scn-*/campaign.json")):
        for seed in range(1, n + 1):
            generate_scenario(seed).save(SCENARIO_DIR)


def _scenario_path(sid: str) -> Path:
    p = SCENARIO_DIR / sid
    if not sid.startswith("scn-") or not (p / "campaign.json").exists():
        raise HTTPException(404, f"scenario {sid} not found")
    return p


@lru_cache(maxsize=64)
def _load(sid: str, mtime: float):
    p = SCENARIO_DIR / sid
    s = Scenario.load(p)
    with (p / "events.csv").open(newline="") as fh:
        events = list(csv.DictReader(fh))
    return s, events


def load_scenario(sid: str):
    p = _scenario_path(sid)
    return _load(sid, (p / "campaign.json").stat().st_mtime)


def _context(sid: str, recall: float, fpr: float):
    s, events = load_scenario(sid)
    t = s.defender_view()
    ev = simulate_alerts(events, recall, fpr, seed=s.seed + 7919)
    return s, t, ev


def threat_level(score: float) -> str:
    if score >= 0.6:
        return "Critical"
    if score >= 0.35:
        return "Elevated"
    if score >= 0.15:
        return "Guarded"
    return "Low"


def feedback_store() -> FeedbackStore:
    return FeedbackStore(EXPERIMENT_DIR / "feedback.jsonl")


# --------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "scenarios_dir": str(SCENARIO_DIR)}


@app.get("/api/scenarios")
def list_scenarios():
    ensure_scenarios()
    out = []
    for d in sorted(SCENARIO_DIR.glob("scn-*")):
        f = d / "campaign.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        topo = json.loads((d / "topology.json").read_text())
        out.append({"id": d.name, "entry": c["entry"], "entry_vector": c["entry_vector"],
                    "target": c["target"], "hops": len(c["path"]) - 1,
                    "assets": len(topo["assets"]), "edges": len(topo["edges"])})
    return out


@app.get("/api/scenarios/{sid}/analysis")
def analysis(sid: str, recall: float = 0.6, fpr: float = 0.01, k: int = 5):
    s, t, ev = _context(sid, recall, fpr)
    g = t.to_networkx()
    entry, target = s.campaign["entry"], s.campaign["target"]
    ranked = risk_weighted_paths(alert_weighted_graph(g, ev), entry, target, k=k)
    paths = []
    for i, rp in enumerate(ranked):
        steps = explain_path(g, rp.path, ev)
        paths.append({"rank": i + 1, "path": rp.path, "probability": round(rp.probability, 5),
                      "summary": path_summary(steps), "steps": steps})
    on_path = {}
    for p in paths:
        for u, v in zip(p["path"], p["path"][1:]):
            on_path.setdefault((u, v), p["rank"])
    path_nodes = {n: min(p["rank"] for p in paths if n in p["path"])
                  for p0 in paths for n in p0["path"]}

    jewels = [a.id for a in t.crown_jewels(0.85)]
    rg = risk_graph(t, ev)
    risk = incident_risk(rg, entry, jewels)
    max_risk = sum(g.nodes[j]["criticality"] for j in jewels) or 1.0
    score = risk / max_risk
    br = blast_radius(rg, entry)
    reach = {r["asset"]: r["probability"] for r in br["reachable"]}

    nodes = []
    for a in t.assets.values():
        nodes.append({
            "id": a.id, "kind": a.kind, "zone": a.zone, "criticality": a.criticality,
            "exposure": a.exposure, "is_crown": a.criticality >= 0.85,
            "is_entry": a.id == entry, "is_target": a.id == target,
            "path_rank": path_nodes.get(a.id),
            "reach_probability": 1.0 if a.id == entry else round(reach.get(a.id, 0.0), 4),
            "alerts_in": sum(ev.alert_count(u, a.id) for u in g.predecessors(a.id)),
            "alerts_out": sum(ev.alert_count(a.id, v) for v in g.successors(a.id)),
            "services": [x.name for x in a.services],
        })
    links = []
    for u, v, d in g.edges(data=True):
        links.append({
            "source": u, "target": v, "service": d["service"], "relation": d["relation"],
            "exploitability": d["exploitability"], **technique_for(d["service"]),
            "alerts": ev.alert_count(u, v), "events": ev.event_count(u, v),
            "path_rank": on_path.get((u, v)),
        })
    affected = sum(1 for p in reach.values() if p >= 0.05)
    return {
        "scenario": {"id": sid, "entry": entry, "entry_vector": s.campaign["entry_vector"],
                     "target": target, "synthetic": True},
        "detector": {"recall": recall, "fpr": fpr},
        "kpis": {
            "threat_level": threat_level(score),
            "risk_score": round(score, 4),
            "alerts": ev.total_alerts,
            "alerted_connections": sum(1 for x in links if x["alerts"]),
            "affected_assets": affected,
            "assets": len(nodes),
            "crown_jewels": jewels,
            "crown_jewels_reachable": sum(1 for j in jewels if reach.get(j, 0) >= 0.05),
            "blast_radius": br["reachable_count"],
        },
        "nodes": nodes,
        "links": links,
        "paths": paths,
        "ground_truth": {"path": s.campaign["path"], "steps": s.campaign["steps"],
                         "note": "Known only because the scenario is synthetic; for evaluation."},
    }


@app.post("/api/scenarios/{sid}/recommend")
def recommend_actions(sid: str, req: RecommendRequest):
    s, t, ev = _context(sid, req.recall, req.fpr)
    g = t.to_networkx()
    entry, target = s.campaign["entry"], s.campaign["target"]
    paths = [rp.path for rp in risk_weighted_paths(alert_weighted_graph(g, ev), entry, target, k=5)]
    if not paths:
        raise HTTPException(422, "no path from the foothold to the target")
    cfg = OptimizerConfig(lam=req.lam, mu=req.mu, nu=req.nu, max_actions=req.max_actions,
                          mc_samples=10)
    pen = feedback_store().preference_penalty()
    rec = recommend(t, entry, paths, cfg, protected=set(req.protect),
                    preference_penalty=pen, evidence=ev)
    feasible = [a for a in rec["ranked_actions"] if a["feasible"]][:10]
    rejected = [a for a in rec["ranked_actions"] if not a["feasible"]][:10]
    return {"scenario": sid, "config": rec["config"], "preference_penalty": pen,
            "candidates_considered": rec["candidates_considered"],
            "actions": feasible, "rejected": rejected, "plan": rec["plan"],
            "plan_summary": rec["plan_summary"], "protected_assets": rec["protected_assets"],
            "note": rec["note"]}


@app.post("/api/scenarios/{sid}/whatif")
def whatif(sid: str, req: WhatIfRequest):
    s, t, ev = _context(sid, req.recall, req.fpr)
    a = DefenseAction(**req.action.model_dump())
    if a.kind not in ("block_edge", "restrict_service", "require_mfa", "reset_credentials",
                      "isolate_asset"):
        raise HTTPException(422, f"unknown action kind {a.kind}")
    jewels = [x.id for x in t.crown_jewels(0.85)]
    protected = set(req.protect) | set(jewels) | {x.id for x in t.by_kind("domain_controller")}
    ev_res = evaluate_action(t, a, s.campaign["entry"], jewels, protected,
                             OptimizerConfig(mc_samples=10), evidence=ev)
    t1, _ = apply_action(t, a)
    before = {(e.src, e.dst, e.service): e for e in t.edges}
    changed = []
    for e in t1.edges:
        b = before[(e.src, e.dst, e.service)]
        if b.allowed and not e.allowed:
            changed.append({"source": e.src, "target": e.dst, "service": e.service, "change": "blocked"})
        elif e.exploitability < b.exploitability:
            changed.append({"source": e.src, "target": e.dst, "service": e.service,
                            "change": "hardened", "exploitability": e.exploitability})
    g1 = t1.to_networkx()
    new_paths = risk_weighted_paths(alert_weighted_graph(g1, ev), s.campaign["entry"],
                                    s.campaign["target"], k=3)
    return {"evaluation": ev_res.to_dict(), "changed_edges": changed,
            "new_top_paths": [{"path": p.path, "probability": round(p.probability, 5)}
                              for p in new_paths],
            "note": "Simulated on the digital twin only."}


@app.post("/api/feedback")
def post_feedback(req: FeedbackRequest):
    _scenario_path(req.scenario)
    a = DefenseAction(**req.action.model_dump())
    try:
        entry = feedback_store().record(req.scenario, {"kind": a.kind,
                                                       "description": req.description or a.describe()},
                                        req.decision, req.note)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"recorded": entry, "preference_penalty": feedback_store().preference_penalty()}


@app.get("/api/feedback")
def get_feedback():
    fs = feedback_store()
    return {"entries": fs.entries()[-50:], "preference_penalty": fs.preference_penalty()}


@app.get("/api/experiments")
def experiments():
    """Latest report of each kind, compacted for charts."""
    out: dict = {"detectors": [], "pathbench": None, "benchmark": None}
    if not EXPERIMENT_DIR.exists():
        return out
    latest: dict = {}
    for f in sorted(EXPERIMENT_DIR.glob("detectors_*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        latest[(r.get("dataset"), r.get("split"))] = (f.name, r)
    for (dataset, split), (name, r) in sorted(latest.items(), key=lambda kv: str(kv[0])):
        models = []
        for m in r["results"]:
            test = m["test"]
            fam = m["test_per_family"]
            if "val_f1" in test:  # newer reports: one entry per threshold strategy
                strategies = {k: {"metrics": test[k], "families": fam[k]} for k in test}
            else:
                strategies = {"val_f1": {"metrics": test, "families": fam}}
            models.append({"model": m["model"], "kind": m.get("kind", "supervised"),
                           "strategies": strategies})
        out["detectors"].append({"file": name, "dataset": dataset, "split": split,
                                 "date_utc": r.get("date_utc"), "models": models})
    pbs = sorted(EXPERIMENT_DIR.glob("pathbench_*.json"))
    if pbs:
        r = json.loads(pbs[-1].read_text())
        out["pathbench"] = {"file": pbs[-1].name, "runs": r["runs"],
                            "train_alerts": r["train_alerts"],
                            "feature_importance": r.get("feature_importance")}
    bms = sorted(EXPERIMENT_DIR.glob("benchmark_*.json"))
    if bms:
        r = json.loads(bms[-1].read_text())
        out["benchmark"] = {"file": bms[-1].name, "results": r["results"]}
    return out


# ------------------------------------------------------------- frontend
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/") or full_path == "api":
            raise HTTPException(404, "unknown API route")
        root = FRONTEND_DIST.resolve()
        f = (root / full_path).resolve()
        if full_path and f.is_file() and f.is_relative_to(root):
            return FileResponse(f)
        return FileResponse(root / "index.html")
