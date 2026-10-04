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
| 2 | CICIDS2017 / UNSW-NB15 loaders and leakage-aware preprocessing | ✅ Done |
| 3 | Baseline detectors (4 supervised + Isolation Forest), F1 and FPR-budget thresholds, per-family detection | ✅ Done |
| 4 | Graph-context path scoring and MITRE ATT&CK evidence | ⏳ Next |
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

## Public datasets

Download the datasets yourself (they're free, but too large to keep in git) and follow each dataset's terms of use:

| Dataset | Put files in | Source |
|---|---|---|
| CICIDS2017 (MachineLearningCSV, 8 files) | `data/raw/cicids2017/` | https://www.unb.ca/cic/datasets/ids-2017.html |
| UNSW-NB15 (training-set and testing-set CSVs) | `data/raw/unsw-nb15/` | https://research.unsw.edu.au/projects/unsw-nb15-dataset |

```bash
python -m aegismind.cli prepare cicids2017                 # day split (default)
python -m aegismind.cli prepare cicids2017 --sample 0.3    # 8 GB RAM laptops
python -m aegismind.cli prepare cicids2017 --split random  # optimistic comparison only
python -m aegismind.cli prepare unsw-nb15                  # official train/test partition
```

`prepare` writes `train/val/test.csv.gz` and a `metadata.json` (row counts, dropped rows, features, scaler statistics, fingerprints) to `data/processed/<dataset>/`.

**Leakage controls:**
- Split first, then fit everything (scaler, constant-column removal, category vocabularies) on the training split only.
- Split by capture day (CICIDS2017) or by the official partition (UNSW-NB15) instead of random rows.
- Drop exact duplicate rows, plus val/test rows that are identical to a training row.
- Remove `Infinity`/NaN rows and identifier columns (IPs, Flow ID, Timestamp), and count everything that was dropped.

> **Note for the paper:** with the CICIDS2017 day split (train Mon–Wed, test Fri), PortScan and DDoS appear only in the test set. Binary detection on that split therefore measures generalization to *unseen* attack families. Report it as the main result and the random split as an in-distribution upper bound.

## Baseline detectors

```bash
python -m aegismind.cli detect cicids2017 --max-train-rows 300000   # quick first run
python -m aegismind.cli detect cicids2017                           # full training split
python -m aegismind.cli detect unsw-nb15
python -m aegismind.cli detect cicids2017 --fpr-budget 0.005        # stricter alert budget
python -m aegismind.cli detect cicids2017 --models random_forest,iforest
```

| Model | Type | Trained on |
|---|---|---|
| Logistic Regression, Decision Tree, Random Forest, Histogram Gradient Boosting | Supervised, class-balanced | Labelled train rows |
| Isolation Forest | Anomaly detection | **Benign train rows only**: it never sees an attack, so it targets unseen attack types |

**Two threshold strategies.** Both are chosen on the validation split and then frozen for the test split:
- `val_f1`: the threshold that maximizes F1 on validation.
- `fpr_budget`: the highest-recall threshold that keeps false positives on validation benign traffic within a budget (default **1%**). This is how alerting is tuned in practice.

**Output per strategy:** precision, recall, F1, PR-AUC, ROC-AUC, false-positive rate and the confusion matrix. Each run also reports the **detection rate per attack family**, with families never seen in training marked `NO`, and the top features (impurity importance or |coefficient|). Every run saves a JSON report to `experiments/` with the data fingerprint, seed and git commit.

> **First finding (CICIDS2017, day split, all test attacks unseen in training):** with the `val_f1` threshold the best test F1 was 0.68 (Logistic Regression, but at 10.6% FPR). Random Forest had the best ranking (PR-AUC 0.88) but missed most attacks, because the threshold tuned on Thursday's web attacks did not transfer to Friday's DDoS, PortScan and Bot traffic. The `fpr_budget` strategy and the benign-only Isolation Forest were added to study this threshold-transfer problem.

## Repository layout

```
aegismind/
  twin/
    topology.py     # Asset / Edge / Topology model, JSON I/O, NetworkX export
    generator.py    # synthetic topologies, campaigns (ground truth), telemetry
    attack_kb.py    # MITRE ATT&CK technique labels for simulated steps
    whatif.py       # apply simulated actions to a copy of the twin, compare
  data/
    cicids.py       # CICIDS2017 loader (header, Infinity, label-encoding quirks)
    unsw.py         # UNSW-NB15 loader (official partition, categoricals)
    preprocess.py   # clean -> split -> fit-on-train -> save, with metadata
  detect/
    baseline.py     # classical detectors, thresholds, per-family metrics
  graph/
    paths.py        # path baselines, blast radius, choke points, min-cut
    metrics.py      # Top-k hit, MRR, edge P/R/F1, next-hop accuracy
  cli.py            # generate | analyze | prepare | detect | benchmark
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
