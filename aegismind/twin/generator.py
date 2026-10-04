"""Synthetic scenario generator.

Public intrusion datasets (CICIDS2017, UNSW-NB15) label individual flows but
do not contain *ground-truth attack paths*. To evaluate attack-path
prediction (Top-k hit rate, MRR, edge precision/recall) we generate
virtual enterprise topologies plus simulated campaigns whose true path is
known by construction.

All output is clearly labelled synthetic. No real hosts are involved.

Validity note (report this in the paper): campaign paths are sampled from
the same ``exploitability`` values that a risk-weighted path baseline uses,
which favours that baseline. ``attacker_temperature`` > 1 makes the simulated
attacker less optimal, and ``observation_noise`` perturbs the exploitability
values the defender sees, to reduce this circularity.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path

import networkx as nx

from .attack_kb import technique_for
from .topology import Asset, Edge, Service, Topology

BASE_EXPLOITABILITY = {
    "smb": 0.35,
    "rdp": 0.30,
    "ssh": 0.25,
    "winrm": 0.30,
    "http": 0.30,
    "sql": 0.25,
    "smtp": 0.20,
    "ldap": 0.15,
    "cached-admin": 0.70,
    "service-account": 0.60,
    "domain-trust": 0.90,
}

PORTS = {"smb": 445, "rdp": 3389, "ssh": 22, "winrm": 5985, "http": 443, "sql": 5432,
         "smtp": 25, "ldap": 389}


@dataclass
class GeneratorConfig:
    workstations: tuple[int, int] = (8, 25)
    admin_workstations: tuple[int, int] = (1, 2)
    web_servers: tuple[int, int] = (1, 2)
    app_servers: tuple[int, int] = (2, 4)
    db_servers: tuple[int, int] = (1, 2)
    p_lateral_smb: float = 0.15  # workstation -> workstation SMB reachability
    p_cached_admin: float = 0.20  # workstation holds cached admin credentials
    p_ws_rdp_admin: float = 0.10  # workstation can RDP to an admin workstation
    max_hops: int = 6
    candidate_paths: int = 30
    attacker_temperature: float = 1.5
    observation_noise: float = 0.15  # std-dev of multiplicative noise on defender view
    background_events: tuple[int, int] = (300, 800)
    window_hours: int = 24


@dataclass
class Scenario:
    scenario_id: str
    seed: int
    topology: Topology
    campaign: dict
    events: list[dict] = field(default_factory=list)

    def defender_view(self) -> Topology:
        """Topology as the defender sees it: same structure, noisy exploitability."""
        return Topology.from_dict(self.campaign["defender_topology"])

    def save(self, out_dir: str | Path) -> Path:
        d = Path(out_dir) / self.scenario_id
        d.mkdir(parents=True, exist_ok=True)
        self.topology.save(d / "topology_true.json")
        self.defender_view().save(d / "topology.json")
        campaign = {k: v for k, v in self.campaign.items() if k != "defender_topology"}
        (d / "campaign.json").write_text(json.dumps(campaign, indent=2))
        _write_csv(d / "events.csv", self.events)
        return d

    @classmethod
    def load(cls, scenario_dir: str | Path) -> "Scenario":
        d = Path(scenario_dir)
        campaign = json.loads((d / "campaign.json").read_text())
        campaign["defender_topology"] = json.loads((d / "topology.json").read_text())
        return cls(scenario_id=campaign["scenario_id"], seed=campaign["seed"],
                   topology=Topology.load(d / "topology_true.json"), campaign=campaign)


# --------------------------------------------------------------------------- utils
def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _clip(x: float, lo: float = 0.01, hi: float = 0.95) -> float:
    return max(lo, min(hi, x))


def _exploitability(service: str, dst: Asset, rng: random.Random) -> float:
    base = BASE_EXPLOITABILITY[service]
    return round(_clip(base * (0.5 + dst.exposure) * rng.uniform(0.85, 1.15)), 4)


# ------------------------------------------------------------------ topology build
def generate_topology(rng: random.Random, cfg: GeneratorConfig, name: str) -> Topology:
    t = Topology(name=name)

    def n(rangepair: tuple[int, int]) -> int:
        return rng.randint(*rangepair)

    def add(prefix: str, count: int, kind: str, zone: str, crit: tuple[float, float],
            expo: tuple[float, float], services: list[str]) -> list[Asset]:
        out = []
        for i in range(1, count + 1):
            a = Asset(id=f"{prefix}-{i:02d}", kind=kind, zone=zone,
                      criticality=round(rng.uniform(*crit), 3),
                      exposure=round(rng.uniform(*expo), 3),
                      services=[Service(s, PORTS[s]) for s in services])
            t.add_asset(a)
            out.append(a)
        return out

    ws = add("PC", n(cfg.workstations), "workstation", "user", (0.1, 0.3), (0.3, 0.8),
             ["smb", "rdp"])
    admins = add("ADMIN-PC", n(cfg.admin_workstations), "admin_workstation", "user",
                 (0.55, 0.7), (0.2, 0.5), ["rdp"])
    webs = add("WEB", n(cfg.web_servers), "web_server", "dmz", (0.4, 0.6), (0.4, 0.9), ["http"])
    mail = add("MAIL", 1, "mail_server", "dmz", (0.4, 0.6), (0.3, 0.7), ["smtp"])
    apps = add("APP", n(cfg.app_servers), "app_server", "server", (0.5, 0.75), (0.3, 0.7),
               ["http", "ssh"])
    files = add("FILE", 1, "file_server", "server", (0.4, 0.6), (0.3, 0.6), ["smb"])
    dcs = add("DC", 1, "domain_controller", "server", (0.88, 0.95), (0.1, 0.35), ["ldap", "rdp"])
    dbs = add("DB", n(cfg.db_servers), "db_server", "data", (0.9, 1.0), (0.15, 0.45), ["sql", "ssh"])

    def link(src: Asset, dst: Asset, service: str, relation: str = "network") -> None:
        t.add_edge(Edge(src=src.id, dst=dst.id, relation=relation, service=service,
                        port=PORTS.get(service), exploitability=_exploitability(service, dst, rng)))

    servers = apps + files + dcs + dbs
    # user zone
    for w in ws:
        for other in ws:
            if other is not w and rng.random() < cfg.p_lateral_smb:
                link(w, other, "smb")
        for f in files:
            link(w, f, "smb")
        for a in apps + webs:
            link(w, a, "http")
        for dc in dcs:
            link(w, dc, "ldap")
        for adm in admins:
            if rng.random() < cfg.p_ws_rdp_admin:
                link(w, adm, "rdp")
        if rng.random() < cfg.p_cached_admin:
            for target in rng.sample(servers + admins, k=min(2, len(servers + admins))):
                link(w, target, "cached-admin", relation="credential")
    # admin workstations manage servers
    for adm in admins:
        for s in apps + dbs:
            link(adm, s, "ssh", relation="credential")
        for s in files + dcs:
            link(adm, s, "rdp", relation="credential")
    # dmz -> server zone (firewall allows only app traffic)
    for wsrv in webs:
        for a in rng.sample(apps, k=max(1, len(apps) // 2)):
            link(wsrv, a, "http")
    for m in mail:
        for dc in dcs:
            link(m, dc, "ldap")
    # server -> data zone
    for a in apps:
        for db in rng.sample(dbs, k=rng.randint(1, len(dbs))):
            link(a, db, "sql")
            if rng.random() < 0.5:
                link(a, db, "service-account", relation="credential")
        for f in files:
            if rng.random() < 0.4:
                link(a, f, "smb")
    # domain controller trust: domain admin reaches every managed server
    for dc in dcs:
        for s in apps + files + dbs + admins:
            link(dc, s, "domain-trust", relation="trust")
    return t


def _noisy_copy(t: Topology, rng: random.Random, noise: float) -> Topology:
    d = t.to_dict()
    for e in d["edges"]:
        e["exploitability"] = round(_clip(e["exploitability"] * math.exp(rng.gauss(0, noise))), 4)
    return Topology.from_dict(d)


# ---------------------------------------------------------------------- campaigns
def sample_campaign(t: Topology, rng: random.Random, cfg: GeneratorConfig) -> dict | None:
    g = t.to_networkx()
    entries = [(a, "phishing") for a in t.by_kind("workstation")]
    entries += [(a, "public-web") for a in t.by_kind("web_server")]
    targets = [a for a in t.crown_jewels(0.85)]
    rng.shuffle(entries)
    for entry, vector in entries:
        target = rng.choice(targets)
        try:
            gen = nx.shortest_simple_paths(g, entry.id, target.id, weight="cost")
            cands = [p for p in islice(gen, cfg.candidate_paths) if len(p) - 1 <= cfg.max_hops]
        except nx.NetworkXNoPath:
            continue
        if not cands:
            continue
        logp = [sum(-g[u][v]["cost"] for u, v in zip(p, p[1:])) for p in cands]
        mx = max(logp)
        weights = [math.exp((lp - mx) / cfg.attacker_temperature) for lp in logp]
        path = rng.choices(cands, weights=weights, k=1)[0]
        steps = [{"step": 0, "src": None, "dst": entry.id, "service": vector,
                  **technique_for(vector)}]
        for i, (u, v) in enumerate(zip(path, path[1:]), start=1):
            ed = g[u][v]
            steps.append({"step": i, "src": u, "dst": v, "service": ed["service"],
                          "relation": ed["relation"], "exploitability": ed["exploitability"],
                          **technique_for(ed["service"])})
        return {"entry": entry.id, "entry_vector": vector, "target": target.id, "path": path,
                "steps": steps, "path_log_prob": round(logp[cands.index(path)], 4),
                "candidate_count": len(cands)}
    return None


# ------------------------------------------------------------------------- events
def generate_events(t: Topology, campaign: dict, rng: random.Random,
                    cfg: GeneratorConfig) -> list[dict]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=rng.randint(0, 300))
    window = timedelta(hours=cfg.window_hours)
    rows: list[dict] = []

    def row(ts, src, dst, service, port, label, step=-1, technique="", relation="network"):
        is_auth = relation != "network"
        return {
            "ts": ts.isoformat(),
            "event_type": "auth" if is_auth else "flow",
            "src": src, "dst": dst, "service": service, "port": port if port else "",
            "bytes": 0 if is_auth else int(rng.lognormvariate(8, 1.6)),
            "packets": 0 if is_auth else max(1, int(rng.lognormvariate(3, 1.1))),
            "duration_s": 0.0 if is_auth else round(rng.expovariate(1 / 4.0), 3),
            "label": label, "campaign_step": step, "technique_id": technique,
        }

    benign_edges = [e for e in t.edges if e.relation in ("network", "credential")]
    for _ in range(rng.randint(*cfg.background_events)):
        e = rng.choice(benign_edges)
        ts = start + timedelta(seconds=rng.uniform(0, window.total_seconds()))
        rows.append(row(ts, e.src, e.dst, e.service, e.port, "benign", relation=e.relation))

    ts = start + timedelta(seconds=rng.uniform(0.1, 0.4) * window.total_seconds())
    for s in campaign["steps"]:
        if s["src"] is None:
            continue
        ts += timedelta(minutes=rng.uniform(3, 90))  # attacker dwell time
        rows.append(row(ts, s["src"], s["dst"], s["service"], None, "attack", s["step"],
                        s["technique_id"], relation=s["relation"]))
        # a little reconnaissance noise around each step
        for _ in range(rng.randint(0, 3)):
            ts2 = ts - timedelta(seconds=rng.uniform(5, 600))
            rows.append(row(ts2, s["src"], s["dst"], s["service"], None, "attack", s["step"],
                            s["technique_id"]))
    rows.sort(key=lambda r: r["ts"])
    return rows


# ------------------------------------------------------------------------ driver
def generate_scenario(seed: int, cfg: GeneratorConfig | None = None) -> Scenario:
    cfg = cfg or GeneratorConfig()
    for attempt in range(20):
        rng = random.Random(seed * 1000 + attempt)
        sid = f"scn-{seed:04d}"
        topo = generate_topology(rng, cfg, name=sid)
        campaign = sample_campaign(topo, rng, cfg)
        if campaign is None:
            continue
        campaign.update({"scenario_id": sid, "seed": seed, "synthetic": True,
                         "generator_config": asdict(cfg)})
        campaign["defender_topology"] = _noisy_copy(topo, rng, cfg.observation_noise).to_dict()
        events = generate_events(topo, campaign, rng, cfg)
        return Scenario(scenario_id=sid, seed=seed, topology=topo, campaign=campaign,
                        events=events)
    raise RuntimeError(f"could not build a scenario with a reachable target for seed {seed}")
