# Data dictionary

All synthetic records carry `synthetic: true` in `campaign.json`. Never mix synthetic data with real or public data without recording where each record came from.

## Scenario folder (`data/scenarios/scn-XXXX/`)

| File | Contents |
|---|---|
| `topology_true.json` | Ground-truth virtual topology (exact exploitability values) |
| `topology.json` | Defender view: same assets and edges, noisy exploitability |
| `campaign.json` | Entry, target, ground-truth path, ATT&CK-tagged steps, generator config, seed |
| `events.csv` | Labelled telemetry: benign background and attack-step events |

## Asset

| Field | Type | Description |
|---|---|---|
| `id` | str | Unique asset ID, e.g. `PC-03`, `DB-01` |
| `kind` | enum | workstation, admin_workstation, web_server, app_server, file_server, mail_server, domain_controller, db_server |
| `zone` | enum | user, dmz, server, data |
| `criticality` | float 0–1 | Business importance. Crown jewels are ≥ 0.85 |
| `exposure` | float 0–1 | Simulated weakness proxy. **Not** a real CVE or CVSS score |
| `services` | list | `{name, port}` |

## Edge (directed, `src → dst`)

| Field | Type | Description |
|---|---|---|
| `relation` | enum | `network` (service reachability), `credential` (reusable creds), `trust` (domain trust) |
| `service` | str | smb, rdp, ssh, winrm, http, sql, smtp, ldap, cached-admin, service-account, domain-trust |
| `port` | int or null | Port for network services |
| `exploitability` | float (0, 1] | Simulated probability that this step succeeds |
| `allowed` | bool | False when a simulated policy blocks the edge |

In graph form, `cost = -log(exploitability)`, so the shortest path by cost is the most probable path.

## Campaign step

| Field | Description |
|---|---|
| `step` | 0 is initial access (external → entry), then 1..n |
| `src`, `dst`, `service`, `relation`, `exploitability` | Edge used |
| `technique_id`, `technique`, `tactic` | MITRE ATT&CK label (vocabulary only) |

## events.csv

| Column | Description |
|---|---|
| `ts` | ISO-8601 UTC timestamp |
| `event_type` | `flow` (network) or `auth` (credential or trust use) |
| `src`, `dst`, `service`, `port` | Endpoints and service |
| `bytes`, `packets`, `duration_s` | Flow volume. 0 for auth events |
| `label` | `benign` or `attack` |
| `campaign_step` | Attack step index, or -1 for benign events |
| `technique_id` | ATT&CK ID for attack events |

## Public datasets (Sprint 2)

| Dataset | Use | Mapping notes |
|---|---|---|
| CICIDS2017 | Flow-level detector training and evaluation | IPs → pseudonymous asset IDs; split by capture day to avoid leakage |
| UNSW-NB15 | Generalization test | Different feature schema; map only shared fields and document every mapping |
| MITRE ATT&CK | Technique vocabulary | A knowledge base, not training data |

Raw files go in `data/raw/` and are git-ignored. Follow each dataset's terms of use.
