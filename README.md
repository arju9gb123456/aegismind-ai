# AegisMind AI

**Graph-based cyber attack-path prediction with a digital-twin simulator and explainable, human-reviewed defense recommendations.**

B.Tech CSE major project and research prototype. AegisMind models an enterprise-like network as a graph, ranks plausible attack paths from a suspected foothold to critical assets, and tests candidate mitigations on an isolated *digital twin* before a human approves anything.

> ⚠️ **Scope and safety.** This is a research prototype, not a production security product. It works only on public datasets and synthetic or virtual lab data. It never scans, exploits or changes real systems, and every recommendation needs human review.

---

## Status

| Sprint | Work | Status |
|---|---|---|
| 1 | Repository, environment, data dictionary, threat model | ✅ Done |
| 1+ | Synthetic scenario generator with ground-truth attack paths | ✅ Done |
| 1+ | NetworkX baselines: hop-shortest and risk-weighted paths, blast radius, choke points, min-cut | ✅ Done |
| 1+ | Minimal digital-twin what-if engine and path metrics (Top-k, MRR, edge F1) | ✅ Done |
| 2 | CICIDS2017 / UNSW-NB15 loader and preprocessing with leakage controls | ⏳ Next |
| 3 | Baseline detector (scikit-learn) with metrics and confusion matrix | ⏳ |
| 4 | Graph-context path scoring and MITRE ATT&CK evidence | ⏳ |
| 5 | Defense optimizer, explainability layer and analyst feedback loop | ⏳ |
| 6 | FastAPI and React dashboard | ⏳ |
| 7 | Evaluation, ablations and paper draft | ⏳ |

## Why synthetic scenarios?

Public intrusion datasets label individual *flows*, but they don't contain ground-truth *attack paths*. Without known paths you can't measure path prediction (Top-k hit rate, MRR, edge precision/recall). `aegismind.twin.generator` builds random enterprise topologies and simulated campaigns whose true path is known by construction:

- **Zones:** user, DMZ, server, data. The firewall policy blocks direct user→data traffic.
- **Assets:** workstations, admin workstations, web, mail, app, file, domain controller and database servers.
- **Relationships:** network services (SMB, RDP, SSH, HTTP, SQL, LDAP), credential reuse and domain trust.
- Each step carries a simulated `exploitability` and a MITRE ATT&CK technique tag.
- **Defender view:** the defender sees the same structure with noisy exploitability values, and the simulated attacker is deliberately not optimal (`attacker_temperature`). Both reduce the circularity between how paths are generated and how they're predicted.
- **Telemetry:** labelled flow and auth events (benign background traffic plus attack steps) are written as CSV.

## Quick start

```bash
git clone https://github.com/arju9gb123456/aegismind-ai.git
cd aegismind-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

pytest                                            # run the test suite

python -m aegismind.cli generate --count 20       # synthetic scenarios -> data/scenarios/
python -m aegismind.cli analyze data/scenarios/scn-0001
python -m aegismind.cli benchmark --count 200     # baseline comparison -> experiments/
```

Example `analyze` output (excerpt):

```
Top predicted paths (risk-weighted, defender view):
  1. p=0.2138  PC-16 -> PC-21 -> DC-01 -> DB-01
  2. p=0.1895  PC-16 -> APP-01 -> DB-01
Ground truth: PC-16 -> PC-10 -> APP-02 -> DB-01
  step 1: PC-16 -> PC-10  [smb] T1021.002 Remote Services: SMB/Windows Admin Shares
Incident choke points (PC-16 -> DB-01):
  PC-16 -> PC-21 [smb]  -11.38%
```

Each benchmark run saves a JSON record with the date, git commit, Python version, seeds, generator config and results, so every number in the paper can be traced back to a run.

## Repository layout

```
aegismind/
  twin/
    topology.py     # Asset / Edge / Topology model, JSON I/O, NetworkX export
    generator.py    # synthetic topologies, campaigns (ground truth), telemetry
    attack_kb.py    # MITRE ATT&CK technique labels for simulated steps
    whatif.py       # apply simulated actions to a copy of the twin, compare
  graph/
    paths.py        # path baselines, blast radius, choke points, min-cut
    metrics.py      # Top-k hit, MRR, edge P/R/F1, next-hop accuracy
  cli.py            # generate | analyze | benchmark
configs/default.yaml
docs/               # data dictionary, threat model
tests/
data/               # raw/, processed/, scenarios/ (git-ignored)
experiments/        # benchmark records (git-ignored)
```

## Documentation

- [Data dictionary](docs/data_dictionary.md)
- [Threat model and scope](docs/threat_model.md)

## Tech stack

Python · NetworkX · pandas · scikit-learn · FastAPI · PostgreSQL · React + TypeScript · Cytoscape.js · Docker

## License

MIT. See [LICENSE](LICENSE).

---

Made by Arjun Ahirwar
