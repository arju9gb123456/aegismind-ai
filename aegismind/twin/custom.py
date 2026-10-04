"""Hand-written networks for the digital twin.

Describe a network in YAML (see ``examples/``) and turn it into a scenario
that the CLI and the dashboard can analyse like a generated one:

    name: Small clinic
    incident: {entry: RECEPTION-PC, target: PATIENT-DB}   # optional
    assets:
      - {id: RECEPTION-PC, kind: workstation, zone: user}
      - {id: PATIENT-DB, kind: db_server, zone: data, criticality: 1.0}
    links:
      - {src: RECEPTION-PC, dst: PATIENT-DB, service: sql}

Defaults keep the file short: criticality and exposure come from the asset
kind, the relation comes from the service (cached-admin / service-account ->
credential, domain-trust -> trust, everything else -> network), ports come
from the service, and exploitability is derived the same way as in the
generator unless given. Use generic names: never put real hostnames, IPs,
URLs, keys or personal data in these files.
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from .generator import (BASE_EXPLOITABILITY, PORTS, GeneratorConfig, Scenario, _noisy_copy,
                        generate_events, sample_campaign)
from .topology import ASSET_KINDS, RELATIONS, ZONES, Asset, Edge, Service, Topology

KIND_DEFAULTS = {  # (criticality, exposure, default services)
    "workstation": (0.2, 0.55, ["smb", "rdp"]),
    "admin_workstation": (0.65, 0.35, ["rdp"]),
    "web_server": (0.5, 0.65, ["http"]),
    "app_server": (0.65, 0.45, ["http", "ssh"]),
    "file_server": (0.5, 0.45, ["smb"]),
    "mail_server": (0.5, 0.5, ["smtp"]),
    "domain_controller": (0.9, 0.25, ["ldap", "rdp"]),
    "db_server": (0.95, 0.3, ["sql", "ssh"]),
}
SERVICE_RELATION = {"cached-admin": "credential", "service-account": "credential",
                    "domain-trust": "trust"}


class NetworkError(ValueError):
    """A problem in a network file, with a message that says how to fix it."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "custom"


def load_network(path: str | Path) -> tuple[Topology, dict]:
    """Parse and validate a network YAML file. Returns (topology, incident dict)."""
    p = Path(path)
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise NetworkError(f"{p.name}: not valid YAML ({e})") from e
    if not isinstance(doc, dict) or "assets" not in doc or "links" not in doc:
        raise NetworkError(f"{p.name}: needs top-level 'assets' and 'links' lists")
    name = str(doc.get("name") or p.stem)
    t = Topology(name=name)
    rng = random.Random(f"network|{name}")

    for i, a in enumerate(doc["assets"], 1):
        where = f"{p.name}: asset #{i}"
        if not isinstance(a, dict) or "id" not in a:
            raise NetworkError(f"{where} needs an 'id'")
        kind = a.get("kind", "workstation")
        if kind not in ASSET_KINDS:
            raise NetworkError(f"{where} ({a['id']}): kind '{kind}' is not one of {', '.join(ASSET_KINDS)}")
        zone = a.get("zone", "user")
        if zone not in ZONES:
            raise NetworkError(f"{where} ({a['id']}): zone '{zone}' is not one of {', '.join(ZONES)}")
        crit, expo, services = KIND_DEFAULTS[kind]
        svcs = a.get("services", services)
        unknown = [s for s in svcs if s not in PORTS]
        if unknown:
            raise NetworkError(f"{where} ({a['id']}): unknown services {unknown}; use {', '.join(PORTS)}")
        try:
            t.add_asset(Asset(id=str(a["id"]), kind=kind, zone=zone,
                              criticality=float(a.get("criticality", crit)),
                              exposure=float(a.get("exposure", expo)),
                              services=[Service(s, PORTS[s]) for s in svcs]))
        except ValueError as e:
            raise NetworkError(f"{where}: {e}") from e

    for i, l in enumerate(doc["links"], 1):
        where = f"{p.name}: link #{i}"
        if not isinstance(l, dict) or "src" not in l or "dst" not in l or "service" not in l:
            raise NetworkError(f"{where} needs 'src', 'dst' and 'service'")
        src, dst, svc = str(l["src"]), str(l["dst"]), l["service"]
        for end in (src, dst):
            if end not in t.assets:
                raise NetworkError(f"{where}: '{end}' is not a listed asset id")
        if svc not in BASE_EXPLOITABILITY:
            raise NetworkError(f"{where}: service '{svc}' is not one of {', '.join(BASE_EXPLOITABILITY)}")
        rel = l.get("relation", SERVICE_RELATION.get(svc, "network"))
        if rel not in RELATIONS:
            raise NetworkError(f"{where}: relation '{rel}' is not one of {', '.join(RELATIONS)}")
        if "exploitability" in l:
            ex = float(l["exploitability"])
        else:
            dst_expo = t.assets[dst].exposure
            ex = round(min(0.95, max(0.01, BASE_EXPLOITABILITY[svc] * (0.5 + dst_expo)
                                     * rng.uniform(0.9, 1.1))), 4)
        try:
            t.add_edge(Edge(src=src, dst=dst, relation=rel, service=svc, port=PORTS.get(svc),
                            exploitability=ex))
        except ValueError as e:
            raise NetworkError(f"{where}: {e}") from e
        if l.get("bidirectional"):
            t.add_edge(Edge(src=dst, dst=src, relation=rel, service=svc, port=PORTS.get(svc),
                            exploitability=ex))

    incident = doc.get("incident") or {}
    for k in ("entry", "target"):
        if k in incident and incident[k] not in t.assets:
            raise NetworkError(f"{p.name}: incident {k} '{incident[k]}' is not a listed asset id")
    if not t.crown_jewels(0.85) and "target" not in incident:
        raise NetworkError(f"{p.name}: no crown jewel (criticality >= 0.85) and no incident target")
    return t, incident


def scenario_from_network(path: str | Path, seed: int = 1,
                          cfg: GeneratorConfig | None = None) -> Scenario:
    cfg = cfg or GeneratorConfig()
    t, inc = load_network(path)
    sid = f"scn-{_slug(t.name)}"
    for attempt in range(20):
        rng = random.Random(seed * 1000 + attempt)
        campaign = sample_campaign(t, rng, cfg, entry=inc.get("entry"), target=inc.get("target"),
                                   vector=inc.get("vector"))
        if campaign is not None:
            break
    else:
        where = f"{inc.get('entry', 'any entry')} to {inc.get('target', 'a crown jewel')}"
        raise NetworkError(f"no attack route from {where} within {cfg.max_hops} hops; "
                           "add links or pick another entry/target")
    campaign.update({"scenario_id": sid, "seed": seed, "synthetic": True,
                     "source_network": str(Path(path).name),
                     "generator_config": asdict(cfg)})
    campaign["defender_topology"] = _noisy_copy(t, rng, cfg.observation_noise).to_dict()
    events = generate_events(t, campaign, rng, cfg)
    return Scenario(scenario_id=sid, seed=seed, topology=t, campaign=campaign, events=events)
