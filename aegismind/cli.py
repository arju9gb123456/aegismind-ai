"""Command-line entry point.

    python -m aegismind.cli generate --count 50 --seed 1
    python -m aegismind.cli analyze data/scenarios/scn-0001
    python -m aegismind.cli benchmark --count 100 --seed 1
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from aegismind import __version__
from aegismind.graph import metrics
from aegismind.graph.paths import (blast_radius, choke_points, hop_shortest_paths,
                                   risk_weighted_paths)
from aegismind.twin.generator import GeneratorConfig, Scenario, generate_scenario
from aegismind.twin.whatif import Action, compare


def load_config(path: str | None) -> GeneratorConfig:
    if not path:
        return GeneratorConfig()
    raw = yaml.safe_load(Path(path).read_text()).get("generator", {})
    tuples = {k: tuple(v) for k, v in raw.items() if isinstance(v, list)}
    return GeneratorConfig(**{**raw, **tuples})


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "uncommitted"


# --------------------------------------------------------------------- commands
def cmd_generate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    out = Path(args.out)
    for i in range(args.count):
        s = generate_scenario(args.seed + i, cfg)
        d = s.save(out)
        print(f"{s.scenario_id}: {len(s.topology.assets)} assets, {len(s.topology.edges)} edges, "
              f"path {' -> '.join(s.campaign['path'])}  [{d}]")


def cmd_analyze(args: argparse.Namespace) -> None:
    s = Scenario.load(args.scenario)
    t = s.defender_view()
    g = t.to_networkx()
    entry, target = s.campaign["entry"], s.campaign["target"]
    ranked = risk_weighted_paths(g, entry, target, k=args.k)
    truth = s.campaign["path"]

    print(f"Scenario {s.scenario_id}  entry={entry} ({s.campaign['entry_vector']})  target={target}")
    print("\nTop predicted paths (risk-weighted, defender view):")
    for i, rp in enumerate(ranked, 1):
        mark = "  <-- ground truth" if rp.path == truth else ""
        print(f"  {i}. p={rp.probability:.4f}  {' -> '.join(rp.path)}{mark}")
    print(f"\nGround truth: {' -> '.join(truth)}")
    for st in s.campaign["steps"]:
        print(f"  step {st['step']}: {st['src'] or 'external'} -> {st['dst']}  "
              f"[{st['service']}] {st['technique_id']} {st['technique']}")

    br = blast_radius(g, entry)
    print(f"\nBlast radius from {entry}: {br['reachable_count']} assets reachable, "
          f"expected critical exposure {br['expected_critical_exposure']}")

    jewels = [a.id for a in t.crown_jewels(0.85)]
    entries = [a.id for a in t.by_kind("workstation", "web_server")]
    print("\nNetwork-wide choke points (all entry points -> all crown jewels):")
    for c in choke_points(g, entries, jewels, top_n=5):
        print(f"  {c['edge'][0]} -> {c['edge'][1]} [{c['service']}]  "
              f"-{c['risk_reduction_pct']}%  (betweenness {c['betweenness']})")

    incident = choke_points(g, [entry], [target], top_n=3)
    print(f"\nIncident choke points ({entry} -> {target}):")
    for c in incident:
        print(f"  {c['edge'][0]} -> {c['edge'][1]} [{c['service']}]  -{c['risk_reduction_pct']}%")
    if incident:
        u, v = incident[0]["edge"]
        print("\nWhat-if on top incident choke point:")
        print(json.dumps(compare(t, Action("block_edge", src=u, dst=v), entry, target), indent=2))


def cmd_benchmark(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    methods = {"hop_shortest": hop_shortest_paths, "risk_weighted": risk_weighted_paths}
    scores: dict[str, dict[str, list[float]]] = {
        m: {"top1": [], "top3": [], "top5": [], "mrr": [], "edge_f1@1": []} for m in methods}
    for i in range(args.count):
        s = generate_scenario(args.seed + i, cfg)
        g = s.defender_view().to_networkx()
        truth = s.campaign["path"]
        for name, fn in methods.items():
            ranked = [rp.path for rp in fn(g, truth[0], truth[-1], k=10, max_hops=cfg.max_hops)]
            sc = scores[name]
            sc["top1"].append(metrics.topk_hit(ranked, truth, 1))
            sc["top3"].append(metrics.topk_hit(ranked, truth, 3))
            sc["top5"].append(metrics.topk_hit(ranked, truth, 5))
            sc["mrr"].append(metrics.reciprocal_rank(ranked, truth))
            sc["edge_f1@1"].append(metrics.edge_precision_recall(ranked[0], truth)[2]
                                   if ranked else 0.0)

    summary = {m: {k: round(statistics.mean(v), 4) for k, v in sc.items()}
               for m, sc in scores.items()}
    print(f"Benchmark on {args.count} synthetic scenarios (seeds {args.seed}..{args.seed + args.count - 1})")
    print(f"{'method':<15}" + "".join(f"{k:>11}" for k in next(iter(summary.values()))))
    for m, row in summary.items():
        print(f"{m:<15}" + "".join(f"{v:>11.4f}" for v in row.values()))

    record = {
        "experiment": "attack_path_baselines",
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aegismind_version": __version__,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "seeds": [args.seed, args.seed + args.count - 1],
        "data": "synthetic (aegismind.twin.generator)",
        "generator_config": {k: v for k, v in vars(cfg).items()},
        "results": summary,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"benchmark_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    f.write_text(json.dumps(record, indent=2, default=list))
    print(f"\nSaved experiment record -> {f}")


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aegismind", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"aegismind {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create synthetic scenarios")
    g.add_argument("--count", type=int, default=10)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--config", default="configs/default.yaml")
    g.add_argument("--out", default="data/scenarios")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("analyze", help="paths, blast radius and choke points for one scenario")
    a.add_argument("scenario")
    a.add_argument("-k", type=int, default=5)
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("benchmark", help="compare path baselines on synthetic scenarios")
    b.add_argument("--count", type=int, default=100)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--config", default="configs/default.yaml")
    b.add_argument("--out", default="experiments")
    b.set_defaults(func=cmd_benchmark)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
