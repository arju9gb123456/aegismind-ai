# Threat model and scope

## System under study
A virtual, enterprise-like network (the digital twin) with four zones (user, DMZ, server, data), together with telemetry from public intrusion datasets.

## Attacker assumptions (simulated)
- Gains initial access through a user workstation (phishing) or a public-facing web server.
- Moves laterally over reachable network services, reused credentials and domain trust.
- Aims for a crown-jewel asset (database server or domain controller).
- Is not perfectly optimal: path choice is stochastic (`attacker_temperature`).

## Defender assumptions
- Knows the asset inventory and allowed connectivity.
- Has only *estimates* of how easy each step is (noisy exploitability).
- Can observe flow and auth telemetry, with some noise.
- Can only take simulated actions on the twin: block an edge, isolate an asset, and (later) restrict a service or require MFA.

## Trust boundaries and safety rules
1. Nothing in this repository connects to, scans or changes a real system.
2. Recommendations are proposals for a human reviewer and are never executed automatically.
3. Real organizational data (for example hospital or employee traffic) is out of scope unless there is written authorization and a privacy review.
4. Simulated values are labelled as simulated everywhere, including the dashboard and paper figures.

## Out of scope
Exploit development, malware analysis, real-time production blocking, and claims of real-world protection.

## Threats to validity (track in the paper)
- **Circularity:** campaigns are generated from the same exploitability model that the risk-weighted baseline uses. This is partly addressed by defender-view noise and attacker temperature.
- **Simulation fidelity:** the topology generator is simpler than real enterprises.
- **Dataset shift:** models trained on public datasets may not transfer to other networks.
