"""Explanations (report section 6.4, Sprint 5).

Two kinds of explanation, both meant for a human reviewer:

1. Path explanations: for every step of a predicted attack path, which edge
   evidence supports it (service, ATT&CK technique, exploitability, alerts),
   and, when a learned path model is given, how much the alerts changed that
   step's probability (evidence impact = log p with alerts - log p without).

2. Detector explanations: which features pushed a flow towards "attack".
   Uses SHAP when the optional ``shap`` package is installed (TreeExplainer
   for tree models, LinearExplainer for logistic regression). Without SHAP it
   falls back to an occlusion method: replace one feature at a time with its
   training median and measure how much the attack score drops.

Feature attributions show what the model relied on. They do not prove cause.
SHAP units differ by model: log-odds for logistic regression and gradient
boosting, probability for random forests and decision trees, so compare
feature rankings across models, not raw values.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np

from aegismind.graph.evidence import EdgeEvidence
from aegismind.twin.attack_kb import technique_for


# ------------------------------------------------------------------ paths
def explain_path(g: nx.DiGraph, path: list[str], evidence: EdgeEvidence | None = None,
                 model=None) -> list[dict]:
    steps = []
    impact = _evidence_impact(g, path, evidence, model) if (model and evidence) else None
    for i, (u, v) in enumerate(zip(path, path[1:])):
        d = g[u][v]
        tech = technique_for(d["service"])
        alerts = evidence.alert_count(u, v) if evidence else 0
        events = evidence.event_count(u, v) if evidence else 0
        reasons = [f"{d['relation']} access over {d['service']} "
                   f"({tech['technique_id']} {tech['technique']})",
                   f"estimated step success {d['exploitability']:.0%}"]
        if alerts:
            reasons.append(f"{alerts} detector alert(s) on this connection out of {events} event(s)")
        elif evidence is not None:
            reasons.append("no detector alerts on this connection")
        if g.nodes[v].get("criticality", 0) >= 0.85:
            reasons.append(f"{v} is a crown-jewel asset")
        step = {"step": i + 1, "src": u, "dst": v, "service": d["service"],
                "relation": d["relation"], **tech, "exploitability": d["exploitability"],
                "alerts": alerts, "events": events, "reasons": reasons}
        if impact is not None:
            step["evidence_impact_logp"] = round(impact[i], 4)
        steps.append(step)
    return steps


def _evidence_impact(g, path, evidence, model) -> list[float]:
    from aegismind.graph.learned import make_context
    with_ev = make_context(g, path[-1], evidence)
    without = make_context(g, path[-1], EdgeEvidence())
    out, visited = [], {path[0]}
    for step, (u, v) in enumerate(zip(path, path[1:])):
        a = dict(model.step_log_probs(with_ev, u, visited, step))
        b = dict(model.step_log_probs(without, u, visited, step))
        out.append(a.get(v, -math.inf) - b.get(v, -math.inf))
        visited.add(v)
    return out


def path_summary(steps: list[dict]) -> str:
    alerted = [s for s in steps if s["alerts"]]
    tactics = list(dict.fromkeys(s["tactic"] for s in steps))
    return (f"{len(steps)}-step route; {len(alerted)} of {len(steps)} steps have detector "
            f"alerts; tactics: {', '.join(tactics)}.")


# ---------------------------------------------------------------- detector
def has_shap() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def explain_detector(model, X_background: np.ndarray, X_explain: np.ndarray,
                     features: list[str], top_k: int = 5, method: str = "auto") -> dict:
    """Global and per-row feature attributions for a fitted sklearn classifier."""
    if method == "auto":
        method = "shap" if has_shap() else "occlusion"
    if method == "shap":
        contrib = _shap_values(model, X_background, X_explain)
    elif method == "occlusion":
        contrib = _occlusion(model, X_background, X_explain)
    else:
        raise ValueError("method must be auto, shap or occlusion")
    global_imp = np.abs(contrib).mean(axis=0)
    order = np.argsort(global_imp)[::-1][:15]
    rows = []
    for i in range(len(X_explain)):
        top = np.argsort(contrib[i])[::-1][:top_k]
        rows.append([{"feature": features[j], "contribution": round(float(contrib[i, j]), 5)}
                     for j in top])
    return {"method": method,
            "global": [{"feature": features[j], "mean_abs_contribution": round(float(global_imp[j]), 5)}
                       for j in order],
            "rows": rows,
            "note": "Attributions show what the model relied on; they do not prove cause."}


def _attack_score(model, X):
    return model.predict_proba(X)[:, 1]


def _occlusion(model, X_bg, X):
    med = np.median(X_bg, axis=0)
    base = _attack_score(model, X)
    contrib = np.zeros_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        Xo = X.copy()
        Xo[:, j] = med[j]
        contrib[:, j] = base - _attack_score(model, Xo)
    return contrib


def _shap_values(model, X_bg, X):
    import shap
    from sklearn.linear_model import LogisticRegression
    if isinstance(model, LogisticRegression):
        exp = shap.LinearExplainer(model, X_bg)
        vals = exp.shap_values(X)
    else:
        exp = shap.TreeExplainer(model)
        vals = exp.shap_values(X, check_additivity=False)
    vals = np.asarray(vals[1] if isinstance(vals, list) else vals)
    if vals.ndim == 3:  # (rows, features, classes)
        vals = vals[:, :, 1]
    return vals
