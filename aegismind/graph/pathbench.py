"""Attack-path benchmark with alert evidence (Sprint 4).

Compares, on held-out synthetic scenarios:

  hop_shortest            fewest hops
  risk_weighted           most probable path by exploitability (Sprint 1 baseline)
  risk_weighted+alerts    heuristic: edges with alerts made cheaper
  graph_context           learned next-hop model, graph features only (ablation)
  graph_context+evidence  learned next-hop model, graph + alert features

Training and test scenarios use disjoint seeds. Alerts come from a simulated
detector whose recall / FPR can be set from measured Sprint 3 results, and
can differ between training and test (detector mismatch).
"""
from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone

from aegismind.twin.generator import GeneratorConfig, generate_scenario

from . import metrics
from .evidence import alert_weighted_graph, simulate_alerts
from .learned import GraphContextPathModel
from .paths import hop_shortest_paths, risk_weighted_paths

METHODS = ("hop_shortest", "risk_weighted", "risk_weighted+alerts", "graph_context",
           "graph_context+evidence")
METRIC_KEYS = ("top1", "top3", "top5", "mrr", "edge_f1@1")


def _load(seeds, cfg: GeneratorConfig, recall: float, fpr: float):
    out = []
    for s in seeds:
        sc = generate_scenario(s, cfg)
        g = sc.defender_view().to_networkx()
        ev = simulate_alerts(sc.events, recall, fpr, seed=s + 7919)
        out.append((g, sc.campaign["path"], ev))
    return out


def train_models(cfg: GeneratorConfig, train_seeds, recall: float, fpr: float, seed: int = 42):
    train = _load(train_seeds, cfg, recall, fpr)
    models = [GraphContextPathModel(use_evidence=ue, seed=seed, max_hops=cfg.max_hops).fit(train)
              for ue in (False, True)]
    return models, train


def evaluate(models, test, cfg: GeneratorConfig, k: int = 10) -> dict:
    scores = {m: {key: [] for key in METRIC_KEYS} for m in METHODS}
    learned = {m.name: m for m in models}
    for g, truth, ev in test:
        s, t = truth[0], truth[-1]
        preds = {
            "hop_shortest": [p.path for p in hop_shortest_paths(g, s, t, k, cfg.max_hops)],
            "risk_weighted": [p.path for p in risk_weighted_paths(g, s, t, k, cfg.max_hops)],
            "risk_weighted+alerts": [p.path for p in risk_weighted_paths(
                alert_weighted_graph(g, ev), s, t, k, cfg.max_hops)],
        }
        for name, m in learned.items():
            preds[name] = [p for p, _ in m.predict_paths(g, s, t, ev, k=k)]
        for name, ranked in preds.items():
            sc = scores[name]
            sc["top1"].append(metrics.topk_hit(ranked, truth, 1))
            sc["top3"].append(metrics.topk_hit(ranked, truth, 3))
            sc["top5"].append(metrics.topk_hit(ranked, truth, 5))
            sc["mrr"].append(metrics.reciprocal_rank(ranked, truth))
            sc["edge_f1@1"].append(metrics.edge_precision_recall(ranked[0], truth)[2]
                                   if ranked else 0.0)
    return {m: {k2: round(statistics.mean(v), 4) for k2, v in sc.items()}
            for m, sc in scores.items()}


def run_pathbench(cfg: GeneratorConfig, train_seeds, test_seeds, recall: float, fpr: float,
                  test_settings: list[tuple[float, float]] | None = None, seed: int = 42,
                  importance: bool = True, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    models, train = train_models(cfg, train_seeds, recall, fpr, seed)
    if verbose:
        print(f"  trained on {len(train)} scenarios "
              f"({models[0].n_train_rows_} next-hop decisions) in {time.perf_counter() - t0:.1f}s")
    settings = test_settings or [(recall, fpr)]
    runs = []
    for (r, f) in settings:
        test = _load(test_seeds, cfg, r, f)
        res = evaluate(models, test, cfg)
        runs.append({"test_recall": r, "test_fpr": f, "results": res})
        if verbose:
            print(f"  evaluated test alerts recall={r} fpr={f}")
    report = {
        "experiment": "attack_path_with_evidence",
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": "synthetic (aegismind.twin.generator)",
        "train_seeds": [min(train_seeds), max(train_seeds)],
        "test_seeds": [min(test_seeds), max(test_seeds)],
        "train_alerts": {"recall": recall, "fpr": fpr},
        "seed": seed,
        "generator_config": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in vars(cfg).items()},
        "runs": runs,
    }
    if importance:
        sample = train[: min(120, len(train))]
        report["feature_importance"] = {m.name: m.feature_importance(sample) for m in models}
    return report


def results_table(run: dict) -> str:
    lines = [f"{'method':<24}" + "".join(f"{k:>11}" for k in METRIC_KEYS)]
    for m in METHODS:
        row = run["results"][m]
        lines.append(f"{m:<24}" + "".join(f"{row[k]:>11.4f}" for k in METRIC_KEYS))
    return "\n".join(lines)
