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


def cmd_prepare(args: argparse.Namespace) -> None:
    from aegismind.data import preprocess
    from aegismind.data.cicids import load_cicids2017
    from aegismind.data.unsw import load_unsw_nb15

    conf = yaml.safe_load(Path(args.config).read_text())["datasets"][args.dataset]
    raw_dir = args.raw or conf["raw_dir"]
    out_dir = args.out or conf["out_dir"]
    sample = args.sample if args.sample is not None else conf.get("sample_frac")
    seed = conf.get("seed", 42)
    if args.dataset == "cicids2017":
        df = load_cicids2017(raw_dir, sample_frac=sample, seed=seed)
    else:
        df = load_unsw_nb15(raw_dir, sample_frac=sample, seed=seed)
    fields = preprocess.PrepConfig.__dataclass_fields__
    cfg = preprocess.PrepConfig(dataset=args.dataset,
                                **{k: v for k, v in conf.items() if k in fields})
    if args.split:
        cfg.split = args.split
    data = preprocess.prepare(df, cfg)
    data.metadata["sample_frac"] = sample
    path = preprocess.save(data, out_dir)
    m = data.metadata
    print(f"{args.dataset}: {m['cleaning']['rows_loaded']} rows loaded, "
          f"{m['n_features']} features, split={cfg.split}")
    for k, v in m["cleaning"].items():
        if k != "rows_loaded":
            print(f"  {k}: {v}")
    for name, s in m["splits"].items():
        print(f"  {name:<5} rows={s['rows']:<9} attack_rate={s['attack_rate']:<7} {s['label_family']}")
    if any(m["families_unseen_in_train"].values()):
        print(f"  families not seen in train: {m['families_unseen_in_train']}")
    print(f"Saved -> {path}")


def cmd_detect(args: argparse.Namespace) -> None:
    from aegismind.data import preprocess
    from aegismind.detect import baseline

    conf = yaml.safe_load(Path(args.config).read_text())["datasets"][args.dataset]
    proc = Path(args.data or conf["out_dir"])
    if not (proc / "metadata.json").exists():
        sys.exit(f"No prepared data in {proc}. Run: python -m aegismind.cli prepare {args.dataset}")
    print(f"Loading {proc} ...")
    data = preprocess.load(proc)
    models = [m.strip() for m in args.models.split(",")]
    print(f"Training {', '.join(models)} on {len(data.train):,} train rows "
          f"(max {args.max_train_rows or 'all'}), {len(data.features)} features")
    report = baseline.run_detectors(data, models, seed=args.seed,
                                    max_train_rows=args.max_train_rows,
                                    fpr_budget=args.fpr_budget,
                                    explain_rows=args.explain)
    report["git_commit"] = _git_commit()
    report["aegismind_version"] = __version__
    for strat in baseline.STRATEGIES:
        title = baseline.STRATEGY_TITLES[strat]
        if strat == "fpr_budget":
            title += f" ({args.fpr_budget:.1%})"
        print(f"\n=== {title} ===")
        print("Test results:")
        print(baseline.summary_table(report, strat))
        print("\nTest detection rate by family (Benign row = false positive rate):")
        print(baseline.family_table(report, strat))
    for r in report["results"]:
        exp = r.get("explanation")
        if not exp:
            continue
        print(f"\nWhy {r['model']} flags flows as attacks ({exp['method']}), top features overall:")
        for f in exp["global"][:5]:
            print(f"  {f['feature']:<32} {f['mean_abs_contribution']:.4f}")
        row = exp["explained_rows"][0]
        print(f"  example: test row {row['test_row']} ({row['true_family']}, score {row['score']}) <- "
              + ", ".join(f"{t['feature']} {t['contribution']:+.3f}" for t in row["top"][:3]))
    print(f"\nSaved report -> {baseline.save_report(report, args.out)}")


def cmd_recommend(args: argparse.Namespace) -> None:
    from aegismind.defense.feedback import FeedbackStore
    from aegismind.defense.optimizer import OptimizerConfig, recommend
    from aegismind.explain import explain_path, path_summary
    import csv

    from aegismind.graph.evidence import alert_weighted_graph, simulate_alerts

    s = Scenario.load(args.scenario)
    with (Path(args.scenario) / "events.csv").open(newline="") as fh:
        events = list(csv.DictReader(fh))
    t = s.defender_view()
    g = t.to_networkx()
    ev = simulate_alerts(events, args.recall, args.fpr, seed=s.seed + 7919)
    entry, target = s.campaign["entry"], s.campaign["target"]
    ranked = risk_weighted_paths(alert_weighted_graph(g, ev), entry, target, k=args.k)
    paths = [rp.path for rp in ranked]
    print(f"Scenario {s.scenario_id}: suspected foothold {entry}, likely target {target}, "
          f"{ev.total_alerts} alerts (detector recall={args.recall}, fpr={args.fpr})")
    if not paths:
        sys.exit("No path from the foothold to the target; nothing to recommend.")
    steps = explain_path(g, paths[0], ev)
    print(f"\nMost likely path: {' -> '.join(paths[0])}")
    print(f"  {path_summary(steps)}")
    for st in steps:
        print(f"  {st['step']}. {st['src']} -> {st['dst']}: " + "; ".join(st["reasons"]))

    store = FeedbackStore(args.feedback_file)
    pen = store.preference_penalty()
    cfg = OptimizerConfig(lam=args.lam, mu=args.mu, nu=args.nu, max_actions=args.max_actions)
    protected = {p.strip() for p in args.protect.split(",")} if args.protect else set()
    rec = recommend(t, entry, paths, cfg, protected=protected, preference_penalty=pen,
                    evidence=ev)
    if pen:
        print(f"\nAnalyst feedback applied: {pen}")
    print(f"\nRanked actions ({rec['candidates_considered']} considered; "
          f"utility = risk_reduction - {cfg.lam}*disruption - {cfg.mu}*cost - {cfg.nu}*uncertainty):")
    print(f"  {'#':>2} {'utility':>8} {'risk-':>7} {'disrupt':>8} {'uncert':>7}  action")
    feasible = [a for a in rec["ranked_actions"] if a["feasible"]]
    for i, a in enumerate(feasible[:args.top], 1):
        print(f"  {i:>2} {a['utility']:>8.3f} {a['risk_reduction']:>7.1%} "
              f"{a['service_disruption']:>8.1%} {a['uncertainty']:>7.3f}  {a['action']['description']}")
    rejected = [a for a in rec["ranked_actions"] if not a["feasible"]]
    if rejected:
        print(f"  ({len(rejected)} rejected by constraints, e.g. "
              f"'{rejected[0]['action']['description']}': {rejected[0]['rejected_reason']})")
    ps = rec["plan_summary"]
    print(f"\nRecommended plan ({ps['actions']} action(s)): risk {ps['risk_before']} -> "
          f"{ps['risk_after']} (-{ps['risk_reduction']:.0%}), service disruption {ps['service_disruption']:.1%}")
    for i, a in enumerate(rec["plan"], 1):
        print(f"  {i}. {a['action']['description']}  [rollback: {a['action']['rollback']}]")
    print(f"\n{rec['note']}")

    rec.update({"scenario": s.scenario_id, "paths": paths, "path_explanation": steps,
                "feasible_actions": feasible[:args.top],
                "detector": {"recall": args.recall, "fpr": args.fpr}})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"recommend_{s.scenario_id}.json"
    f.write_text(json.dumps(rec, indent=2, default=str))
    print(f"Saved -> {f}\nRecord a decision: python -m aegismind.cli feedback {args.scenario} "
          f"--action 1 --decision approve")


def cmd_feedback(args: argparse.Namespace) -> None:
    from aegismind.defense.feedback import FeedbackStore

    s = Scenario.load(args.scenario)
    f = Path(args.out) / f"recommend_{s.scenario_id}.json"
    if not f.exists():
        sys.exit(f"No recommendations saved for {s.scenario_id}. Run the recommend command first.")
    rec = json.loads(f.read_text())
    actions = rec["feasible_actions"]
    if not 1 <= args.action <= len(actions):
        sys.exit(f"--action must be between 1 and {len(actions)}")
    a = actions[args.action - 1]["action"]
    store = FeedbackStore(args.feedback_file)
    store.record(s.scenario_id, a, args.decision, args.note or "")
    print(f"Recorded: {args.decision} '{a['description']}'")
    print(f"Current preference adjustments: {store.preference_penalty()}")


def _parse_settings(text: str | None) -> list[tuple[float, float]] | None:
    """'0.6:0.01,0.3:0.05' -> [(0.6, 0.01), (0.3, 0.05)]"""
    if not text:
        return None
    out = []
    for part in text.split(","):
        r, f = part.split(":")
        out.append((float(r), float(f)))
    return out


def cmd_pathbench(args: argparse.Namespace) -> None:
    from aegismind.graph import pathbench

    cfg = load_config(args.config)
    train_seeds = range(args.train_seed, args.train_seed + args.train_count)
    test_seeds = range(args.seed, args.seed + args.count)
    overlap = set(train_seeds) & set(test_seeds)
    if overlap:
        sys.exit(f"train and test seeds overlap ({len(overlap)} seeds); change --train-seed")
    settings = _parse_settings(args.test_alerts)
    print(f"Attack-path benchmark: train seeds {train_seeds.start}..{train_seeds.stop - 1}, "
          f"test seeds {test_seeds.start}..{test_seeds.stop - 1}, "
          f"training alerts recall={args.recall} fpr={args.fpr}")
    report = pathbench.run_pathbench(cfg, train_seeds, test_seeds, args.recall, args.fpr,
                                     test_settings=settings, seed=42,
                                     importance=not args.no_importance)
    report["git_commit"] = _git_commit()
    report["aegismind_version"] = __version__
    for run in report["runs"]:
        print(f"\nTest alerts: recall={run['test_recall']} fpr={run['test_fpr']}")
        print(pathbench.results_table(run))
    if "feature_importance" in report:
        top = report["feature_importance"]["graph_context+evidence"][:6]
        print("\nMost useful features (graph_context+evidence, permutation importance):")
        for f in top:
            print(f"  {f['feature']:<24} {f['importance']:.4f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"pathbench_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    f.write_text(json.dumps(report, indent=2))
    print(f"\nSaved experiment record -> {f}")


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

    pr = sub.add_parser("prepare", help="clean and split a public dataset (leakage-aware)")
    pr.add_argument("dataset", choices=["cicids2017", "unsw-nb15"])
    pr.add_argument("--config", default="configs/default.yaml")
    pr.add_argument("--raw", help="override raw_dir from config")
    pr.add_argument("--out", help="override out_dir from config")
    pr.add_argument("--split", choices=["day", "official", "random"])
    pr.add_argument("--sample", type=float, help="keep this fraction of rows per file")
    pr.set_defaults(func=cmd_prepare)

    d = sub.add_parser("detect", help="train and evaluate baseline intrusion detectors")
    d.add_argument("dataset", choices=["cicids2017", "unsw-nb15"])
    d.add_argument("--models", default="logreg,decision_tree,random_forest,hist_gb,iforest")
    d.add_argument("--explain", type=int, default=0,
                   help="explain the N most attack-like test flows per supervised model")
    d.add_argument("--fpr-budget", type=float, default=0.01,
                   help="max false-positive rate on validation benign (default 0.01 = 1%%)")
    d.add_argument("--max-train-rows", type=int, default=None,
                   help="stratified subsample of train rows (faster runs)")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--config", default="configs/default.yaml")
    d.add_argument("--data", help="override prepared data directory")
    d.add_argument("--out", default="experiments")
    d.set_defaults(func=cmd_detect)

    rc = sub.add_parser("recommend", help="explain the likely attack path and rank defensive actions")
    rc.add_argument("scenario")
    rc.add_argument("--recall", type=float, default=0.6, help="simulated detector recall")
    rc.add_argument("--fpr", type=float, default=0.01, help="simulated detector false-positive rate")
    rc.add_argument("-k", type=int, default=5, help="predicted paths used to find candidate actions")
    rc.add_argument("--max-actions", type=int, default=3)
    rc.add_argument("--lam", type=float, default=0.6, help="weight of service disruption")
    rc.add_argument("--mu", type=float, default=0.1, help="weight of action cost")
    rc.add_argument("--nu", type=float, default=0.5, help="weight of uncertainty")
    rc.add_argument("--protect", help="comma-separated assets that must never be isolated")
    rc.add_argument("--top", type=int, default=8)
    rc.add_argument("--feedback-file", default="experiments/feedback.jsonl")
    rc.add_argument("--out", default="experiments")
    rc.set_defaults(func=cmd_recommend)

    fb = sub.add_parser("feedback", help="approve or reject a recommended action")
    fb.add_argument("scenario")
    fb.add_argument("--action", type=int, required=True, help="number from the recommend list")
    fb.add_argument("--decision", choices=["approve", "reject"], required=True)
    fb.add_argument("--note")
    fb.add_argument("--feedback-file", default="experiments/feedback.jsonl")
    fb.add_argument("--out", default="experiments")
    fb.set_defaults(func=cmd_feedback)

    pb = sub.add_parser("pathbench",
                        help="attack-path prediction with alert evidence (learned vs baselines)")
    pb.add_argument("--count", type=int, default=200, help="test scenarios")
    pb.add_argument("--seed", type=int, default=1, help="first test seed")
    pb.add_argument("--train-count", type=int, default=400)
    pb.add_argument("--train-seed", type=int, default=1000)
    pb.add_argument("--recall", type=float, default=0.6, help="simulated detector recall")
    pb.add_argument("--fpr", type=float, default=0.01, help="simulated detector false-positive rate")
    pb.add_argument("--test-alerts", help="other test settings, e.g. '0.6:0.01,0.3:0.05,0.9:0.001'")
    pb.add_argument("--no-importance", action="store_true", help="skip permutation importance")
    pb.add_argument("--config", default="configs/default.yaml")
    pb.add_argument("--out", default="experiments")
    pb.set_defaults(func=cmd_pathbench)

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
