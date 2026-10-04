"""Defense optimizer (report section 6.5, Sprint 5).

Compares candidate defensive actions on the digital twin and ranks them with
the report's utility:

    Utility(a) = risk_reduction(a) - lambda * service_disruption(a)
                 - mu * action_cost(a) - nu * uncertainty(a)

* risk                : sum over crown jewels of criticality x best-path
                        probability from the incident entry point; when
                        detector alerts are given, edges with alerts are
                        treated as more likely (same weighting as the
                        risk_weighted+alerts path baseline)
* risk_reduction      : relative drop in risk after the action (0..1)
* service_disruption  : weighted share of *business* connectivity disabled
                        (credential misconfigurations such as cached admin
                        credentials count as zero business value)
* action_cost         : fixed per action type, normalised to 0..1
* uncertainty         : std-dev of risk_reduction when the defender's
                        exploitability estimates are perturbed (Monte Carlo)

Hard constraints reject an action that would cut a required business service
(every app server reachable from workstations over HTTP, every database
reachable from an app server over SQL, the domain controller reachable over
LDAP) or isolate a crown-jewel asset.

Everything runs on a copy of the virtual topology. Output is a ranked list of
recommendations for a human to approve; nothing is executed.
"""
from __future__ import annotations

import copy
import math
import random
import statistics
from dataclasses import asdict, dataclass, field

import networkx as nx

from aegismind.graph.evidence import EdgeEvidence, alert_weighted_graph
from aegismind.twin.attack_kb import technique_for
from aegismind.twin.topology import Topology

ACTION_COST = {  # relative effort / operational cost, 0..1
    "reset_credentials": 0.15,
    "block_edge": 0.20,
    "require_mfa": 0.30,
    "restrict_service": 0.40,
    "isolate_asset": 0.70,
}
MFA_FACTOR = 0.2  # MFA multiplies credential-step exploitability by this
CREDENTIAL_MISCONFIG = {"cached-admin", "service-account"}

# business value of a connectivity edge, by service (misconfigurations = 0)
BUSINESS_WEIGHT = {
    "http": 1.0, "sql": 3.0, "ldap": 2.0, "smb": 1.0, "smtp": 1.0,
    "ssh": 1.0, "rdp": 0.5, "winrm": 0.5,
    "cached-admin": 0.0, "service-account": 0.0, "domain-trust": 0.5,
}


@dataclass
class OptimizerConfig:
    lam: float = 0.6  # service disruption weight (lambda)
    mu: float = 0.1  # action cost weight
    nu: float = 0.5  # uncertainty weight
    max_actions: int = 3
    mc_samples: int = 20
    noise: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class DefenseAction:
    kind: str
    src: str | None = None
    dst: str | None = None
    asset: str | None = None
    service: str | None = None

    def describe(self) -> str:
        if self.kind == "block_edge":
            return f"Block traffic {self.src} -> {self.dst}"
        if self.kind == "restrict_service":
            return f"Restrict {self.service.upper()} access to {self.dst}"
        if self.kind == "require_mfa":
            return f"Require MFA for privileged logins to {self.dst}"
        if self.kind == "reset_credentials":
            return f"Reset cached admin credentials on {self.src}"
        if self.kind == "isolate_asset":
            return f"Isolate {self.asset} from the network"
        return self.kind

    def rollback(self) -> str:
        return {
            "block_edge": "Remove the firewall rule.",
            "restrict_service": "Re-enable the service rule for the affected sources.",
            "require_mfa": "Disable the MFA requirement for this host.",
            "reset_credentials": "No rollback needed; users sign in again with new credentials.",
            "isolate_asset": "Reconnect the asset after investigation.",
        }[self.kind]


@dataclass
class Evaluation:
    action: DefenseAction
    risk_before: float
    risk_after: float
    risk_reduction: float
    service_disruption: float
    action_cost: float
    uncertainty: float
    utility: float
    edges_changed: int
    feasible: bool
    rejected_reason: str | None = None
    attack_techniques_mitigated: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = {**asdict(self.action), "description": self.action.describe(),
                       "rollback": self.action.rollback()}
        for k in ("risk_before", "risk_after", "risk_reduction", "service_disruption",
                  "action_cost", "uncertainty", "utility"):
            d[k] = round(d[k], 4) if math.isfinite(d[k]) else None  # JSON has no infinity
        return d


# ------------------------------------------------------------- apply action
def apply_action(t: Topology, a: DefenseAction) -> tuple[Topology, int]:
    """Return a modified copy of the topology and the number of edges changed."""
    new = copy.deepcopy(t)
    changed = 0
    for e in new.edges:
        if not e.allowed:
            continue
        hit = False
        if a.kind == "block_edge":
            hit = e.src == a.src and e.dst == a.dst
        elif a.kind == "restrict_service":
            hit = e.dst == a.dst and e.service == a.service
        elif a.kind == "reset_credentials":
            hit = e.src == a.src and e.service == "cached-admin"
        elif a.kind == "isolate_asset":
            hit = a.asset in (e.src, e.dst)
        elif a.kind == "require_mfa":
            if e.dst == a.dst and e.relation == "credential":
                e.exploitability = max(0.001, round(e.exploitability * MFA_FACTOR, 6))
                changed += 1
            continue
        if hit:
            e.allowed = False
            changed += 1
    return new, changed


# ----------------------------------------------------------------- measures
def risk_graph(t: Topology, evidence: EdgeEvidence | None = None) -> nx.DiGraph:
    g = t.to_networkx()
    return alert_weighted_graph(g, evidence) if evidence is not None else g


def incident_risk(g: nx.DiGraph, entry: str, jewels: list[str]) -> float:
    if entry not in g:
        return 0.0
    lengths = nx.single_source_dijkstra_path_length(g, entry, weight="cost")
    return sum(math.exp(-lengths[j]) * g.nodes[j]["criticality"] for j in jewels if j in lengths)


def business_weight(t: Topology) -> float:
    return sum(BUSINESS_WEIGHT.get(e.service, 1.0) for e in t.edges if e.allowed)


def business_violations(t: Topology, protected: set[str]) -> list[str]:
    """Every required business service that is currently cut."""
    allowed = [e for e in t.edges if e.allowed]
    ws = {a.id for a in t.by_kind("workstation")}
    out = []
    for app in t.by_kind("app_server"):
        if not any(e.dst == app.id and e.service == "http" and e.src in ws for e in allowed):
            out.append(f"cuts HTTP access from workstations to {app.id}")
    for db in t.by_kind("db_server"):
        if not any(e.dst == db.id and e.service == "sql" for e in allowed):
            out.append(f"cuts all SQL access to {db.id}")
    for dc in t.by_kind("domain_controller"):
        if not any(e.dst == dc.id and e.service == "ldap" for e in allowed):
            out.append(f"cuts LDAP (logins) to {dc.id}")
    for p in sorted(protected):
        if not any(p in (e.src, e.dst) for e in allowed):
            out.append(f"isolates protected asset {p}")
    return out


def check_business_constraints(t: Topology, protected: set[str],
                               baseline: Topology | None = None) -> str | None:
    """Reason string for the first business rule the topology breaks, or None.

    With ``baseline``, rules that were already broken there are ignored, so a
    hand-written network that never offered a service is not held against
    every action.
    """
    already = set(business_violations(baseline, protected)) if baseline is not None else set()
    new = [v for v in business_violations(t, protected) if v not in already]
    return new[0] if new else None


def _perturb(t: Topology, rng: random.Random, noise: float) -> Topology:
    new = copy.deepcopy(t)
    for e in new.edges:
        e.exploitability = min(0.99, max(0.001, e.exploitability * math.exp(rng.gauss(0, noise))))
    return new


# ---------------------------------------------------------------- candidates
def candidate_actions(t: Topology, paths: list[list[str]], entry: str,
                      protected: set[str]) -> list[DefenseAction]:
    """Actions that touch at least one edge or node on the predicted paths."""
    on_path_edges = {(u, v) for p in paths for u, v in zip(p, p[1:])}
    on_path_nodes = {n for p in paths for n in p}
    cands: set[DefenseAction] = set()
    for e in t.edges:
        if not e.allowed or (e.src, e.dst) not in on_path_edges:
            continue
        cands.add(DefenseAction("block_edge", src=e.src, dst=e.dst))
        if e.relation == "network":
            cands.add(DefenseAction("restrict_service", dst=e.dst, service=e.service))
        if e.relation == "credential":
            cands.add(DefenseAction("require_mfa", dst=e.dst))
        if e.service == "cached-admin":
            cands.add(DefenseAction("reset_credentials", src=e.src))
    for n in on_path_nodes:
        if n not in protected:
            cands.add(DefenseAction("isolate_asset", asset=n))
    return sorted(cands, key=lambda a: (a.kind, a.src or "", a.dst or "", a.asset or "",
                                        a.service or ""))


# ------------------------------------------------------------------ evaluate
def evaluate_action(t: Topology, a: DefenseAction, entry: str, jewels: list[str],
                    protected: set[str], cfg: OptimizerConfig,
                    evidence: EdgeEvidence | None = None) -> Evaluation:
    r0 = incident_risk(risk_graph(t, evidence), entry, jewels)
    t1, changed = apply_action(t, a)
    r1 = incident_risk(risk_graph(t1, evidence), entry, jewels)
    rr = (r0 - r1) / r0 if r0 > 0 else 0.0
    bw0 = business_weight(t)
    disruption = (bw0 - business_weight(t1)) / bw0 if bw0 else 0.0
    if a.kind == "require_mfa":
        disruption += 0.01  # small login friction
    cost = ACTION_COST[a.kind]

    # Monte Carlo uncertainty of the risk reduction
    rng = random.Random(f"{cfg.seed}|{a}")
    samples = []
    for _ in range(cfg.mc_samples):
        tp = _perturb(t, rng, cfg.noise)
        p0 = incident_risk(risk_graph(tp, evidence), entry, jewels)
        p1 = incident_risk(risk_graph(apply_action(tp, a)[0], evidence), entry, jewels)
        samples.append((p0 - p1) / p0 if p0 > 0 else 0.0)
    unc = statistics.pstdev(samples) if len(samples) > 1 else 0.0

    reason = check_business_constraints(t1, protected, baseline=t)
    if a.kind == "isolate_asset" and a.asset in protected:
        reason = f"{a.asset} is a protected asset"
    if changed == 0:
        reason = reason or "no effect on the current topology"
    utility = rr - cfg.lam * disruption - cfg.mu * cost - cfg.nu * unc

    techniques = sorted({technique_for(e.service)["technique_id"] for e in t.edges
                         if e.allowed and _touches(a, e)})
    return Evaluation(action=a, risk_before=r0, risk_after=r1, risk_reduction=rr,
                      service_disruption=disruption, action_cost=cost, uncertainty=unc,
                      utility=utility if reason is None else float("-inf"),
                      edges_changed=changed, feasible=reason is None,
                      rejected_reason=reason, attack_techniques_mitigated=techniques)


def _touches(a: DefenseAction, e) -> bool:
    if a.kind == "block_edge":
        return e.src == a.src and e.dst == a.dst
    if a.kind == "restrict_service":
        return e.dst == a.dst and e.service == a.service
    if a.kind == "reset_credentials":
        return e.src == a.src and e.service == "cached-admin"
    if a.kind == "require_mfa":
        return e.dst == a.dst and e.relation == "credential"
    return a.asset in (e.src, e.dst)


# ---------------------------------------------------------------- recommend
def recommend(t: Topology, entry: str, paths: list[list[str]], cfg: OptimizerConfig | None = None,
              jewels: list[str] | None = None, protected: set[str] | None = None,
              preference_penalty: dict[str, float] | None = None,
              evidence: EdgeEvidence | None = None) -> dict:
    """Rank single actions and build a greedy multi-action plan.

    ``preference_penalty`` (from analyst feedback) is subtracted from the
    utility of each action type, e.g. {"isolate_asset": 0.1}.
    """
    cfg = cfg or OptimizerConfig()
    jewels = jewels or [a.id for a in t.crown_jewels(0.85)]
    protected = set(protected or []) | set(jewels) | {a.id for a in t.by_kind("domain_controller")}
    pen = preference_penalty or {}

    def score(ev: Evaluation) -> Evaluation:
        if ev.feasible:
            ev.utility -= pen.get(ev.action.kind, 0.0)
        return ev

    cands = candidate_actions(t, paths, entry, protected)
    singles = [score(evaluate_action(t, a, entry, jewels, protected, cfg, evidence)) for a in cands]
    singles.sort(key=lambda e: (-e.utility, e.action.describe()))

    plan, current = [], t
    remaining = [e.action for e in singles if e.feasible]
    base_risk = incident_risk(risk_graph(t, evidence), entry, jewels)
    for _ in range(cfg.max_actions):
        evs = [score(evaluate_action(current, a, entry, jewels, protected, cfg, evidence))
               for a in remaining]
        evs = [e for e in evs if e.feasible and e.utility > 0]
        if not evs:
            break
        best = max(evs, key=lambda e: (e.utility, e.action.describe()))
        plan.append(best)
        current, _ = apply_action(current, best.action)
        remaining = [a for a in remaining if a != best.action]
    final_risk = incident_risk(risk_graph(current, evidence), entry, jewels)
    bw0 = business_weight(t)
    return {
        "entry": entry,
        "crown_jewels": jewels,
        "protected_assets": sorted(protected),
        "config": asdict(cfg),
        "preference_penalty": pen,
        "uses_alert_evidence": evidence is not None,
        "candidates_considered": len(cands),
        "ranked_actions": [e.to_dict() for e in singles],
        "plan": [e.to_dict() for e in plan],
        "plan_summary": {
            "risk_before": round(base_risk, 4),
            "risk_after": round(final_risk, 4),
            "risk_reduction": round((base_risk - final_risk) / base_risk, 4) if base_risk else 0.0,
            "service_disruption": round((bw0 - business_weight(current)) / bw0, 4) if bw0 else 0.0,
            "actions": len(plan),
        },
        "note": "Simulated on the digital twin only. Every action requires human approval.",
    }
