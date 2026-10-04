"""Baseline intrusion detectors (report section 6.1, Sprint 3).

Supervised (trained on labelled train rows):
  logreg         Logistic Regression (class-balanced)
  decision_tree  Decision Tree (depth-limited, class-balanced)
  random_forest  Random Forest (class-balanced subsamples)
  hist_gb        Histogram Gradient Boosting (class-balanced)

Anomaly detection (trained on BENIGN train rows only, never sees an attack):
  iforest        Isolation Forest; score = how unusual a flow looks

Threshold strategies (both chosen on VAL, frozen for TEST):
  val_f1      threshold that maximises F1 on validation
  fpr_budget  highest-recall threshold whose false-positive rate on validation
              benign traffic stays within a budget (default 1%), which is how
              security teams usually tune alerting

Protocol (report section 11): fit on train only, choose thresholds on val,
report test once, and give per-attack-family detection rates with a flag for
families never seen in training.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

from aegismind.data.preprocess import PreparedData

SUPERVISED = ("logreg", "decision_tree", "random_forest", "hist_gb")
ANOMALY = ("iforest",)
MODEL_NAMES = SUPERVISED + ANOMALY
STRATEGIES = ("val_f1", "fpr_budget")


def make_model(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=20, min_samples_leaf=5,
                                      class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_leaf=2,
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=seed)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              class_weight="balanced", random_state=seed)
    if name == "iforest":
        return IsolationForest(n_estimators=200, max_samples=256, n_jobs=-1,
                               random_state=seed)
    raise ValueError(f"unknown model: {name} (choose from {MODEL_NAMES})")


def scores_of(model, X: np.ndarray) -> np.ndarray:
    """Higher score = more likely an attack, for every model type."""
    if isinstance(model, IsolationForest):
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- thresholds
def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def fpr_budget_threshold(benign_scores: np.ndarray, budget: float) -> float:
    """Smallest threshold that flags at most ``budget`` of benign rows."""
    if len(benign_scores) == 0:
        raise ValueError("need benign rows to set an FPR-budget threshold")
    s = np.sort(benign_scores)
    k = int(np.floor(budget * len(s)))  # benign rows we may flag
    if k <= 0:
        return float(np.nextafter(s[-1], np.inf))
    # flag everything strictly above the (n-k)-th smallest benign score
    return float(np.nextafter(s[len(s) - k - 1], np.inf))


def choose_thresholds(y_val: np.ndarray, s_val: np.ndarray, budget: float,
                      fallback_benign: np.ndarray | None = None) -> dict[str, float]:
    benign = s_val[y_val == 0]
    if len(benign) == 0 and fallback_benign is not None:
        benign = fallback_benign
    return {"val_f1": best_f1_threshold(y_val, s_val),
            "fpr_budget": fpr_budget_threshold(benign, budget)}


# ------------------------------------------------------------------ metrics
def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4) if both else None,
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4) if both else None,
        "false_positive_rate": round(float(fp / (fp + tn)), 6) if (fp + tn) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"benign": int(tn + fp), "attack": int(tp + fn)},
    }


def per_family(df: pd.DataFrame, scores: np.ndarray, threshold: float,
               train_families: set[str]) -> dict:
    """Detection rate per attack family (false-positive rate for Benign)."""
    pred = scores >= threshold
    out = {}
    for fam, idx in df.groupby("label_family").indices.items():
        out[fam] = {
            "rows": int(len(idx)),
            "flagged_rate": round(float(pred[idx].mean()), 4),
            "meaning": "false positive rate" if fam == "Benign" else "detection rate",
            "seen_in_train": fam in train_families,
        }
    return dict(sorted(out.items(), key=lambda kv: (kv[0] != "Benign", kv[0])))


def top_features(model, features: list[str], k: int = 15) -> list[dict]:
    if hasattr(model, "feature_importances_") and not isinstance(model, IsolationForest):
        imp = np.asarray(model.feature_importances_)
        kind = "impurity_importance"
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_).ravel())
        kind = "abs_coefficient (standardised features)"
    else:
        return []
    order = np.argsort(imp)[::-1][:k]
    return [{"feature": features[i], "importance": round(float(imp[i]), 6), "kind": kind}
            for i in order]


# --------------------------------------------------------------------- run
def _subsample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = [g.sample(frac=frac, random_state=seed) if len(g) * frac >= 1 else g
             for _, g in df.groupby("label_family")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def run_detectors(data: PreparedData, models: list[str], seed: int = 42,
                  max_train_rows: int | None = None, fpr_budget: float = 0.01,
                  iforest_max_benign: int = 200_000, explain_rows: int = 0,
                  verbose: bool = True) -> dict:
    if not 0 < fpr_budget < 1:
        raise ValueError("fpr_budget must be between 0 and 1")
    train = _subsample(data.train, max_train_rows, seed)
    X_tr = train[data.features].to_numpy(np.float32)
    y_tr = train["is_attack"].to_numpy()
    benign_tr = train[train["is_attack"] == 0]
    if len(benign_tr) > iforest_max_benign:
        benign_tr = benign_tr.sample(n=iforest_max_benign, random_state=seed)
    X_benign = benign_tr[data.features].to_numpy(np.float32)
    X_va, y_va = data.X("val"), data.y("val")
    X_te, y_te = data.X("test"), data.y("test")
    train_fams = set(data.train["label_family"])

    results = []
    for name in models:
        m = make_model(name, seed)
        t0 = time.perf_counter()
        if name in ANOMALY:
            m.set_params(max_samples=min(256, len(X_benign)))
            m.fit(X_benign)
            rows_used = len(X_benign)
        else:
            m.fit(X_tr, y_tr)
            rows_used = len(X_tr)
        fit_s = time.perf_counter() - t0

        s_va, s_te = scores_of(m, X_va), scores_of(m, X_te)
        fallback = scores_of(m, X_benign[:50_000]) if (y_va == 0).sum() == 0 else None
        thr = choose_thresholds(y_va, s_va, fpr_budget, fallback)
        r = {
            "model": name,
            "kind": "anomaly (benign-only)" if name in ANOMALY else "supervised",
            "train_rows_used": int(rows_used),
            "fit_seconds": round(fit_s, 2),
            "val": {k: binary_metrics(y_va, s_va, t) for k, t in thr.items()},
            "test": {k: binary_metrics(y_te, s_te, t) for k, t in thr.items()},
            "test_per_family": {k: per_family(data.test, s_te, t, train_fams)
                                for k, t in thr.items()},
            "top_features": top_features(m, data.features),
        }
        if explain_rows and name in SUPERVISED:
            from aegismind.explain import explain_detector
            rng = np.random.default_rng(seed)
            bg = X_tr[rng.choice(len(X_tr), size=min(500, len(X_tr)), replace=False)]
            top = np.argsort(s_te)[::-1][:explain_rows]  # most attack-like test flows
            exp = explain_detector(m, bg, X_te[top], data.features)
            exp["explained_rows"] = [
                {"test_row": int(i), "score": round(float(s_te[i]), 5),
                 "true_family": str(data.test["label_family"].iloc[i]), "top": exp["rows"][k]}
                for k, i in enumerate(top)]
            del exp["rows"]
            r["explanation"] = exp
        results.append(r)
        if verbose:
            a, b = r["test"]["val_f1"], r["test"]["fpr_budget"]
            print(f"  {name:<14} fit {fit_s:7.1f}s | PR-AUC {a['pr_auc']} | "
                  f"val_f1: F1 {a['f1']:.4f} FPR {a['false_positive_rate']} | "
                  f"fpr_budget: F1 {b['f1']:.4f} FPR {b['false_positive_rate']}")
    return {
        "experiment": "baseline_detectors",
        "dataset": data.metadata.get("config", {}).get("dataset"),
        "split": data.metadata.get("config", {}).get("split"),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "max_train_rows": max_train_rows,
        "fpr_budget": fpr_budget,
        "iforest_max_benign": iforest_max_benign,
        "n_features": len(data.features),
        "data_fingerprint": data.metadata.get("fingerprint"),
        "threshold_strategies": {
            "val_f1": "max F1 on validation; frozen for test",
            "fpr_budget": f"<= {fpr_budget:.2%} false positives on validation benign; frozen for test",
        },
        "results": results,
    }


def save_report(report: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    f = out / f"detectors_{report['dataset']}_{stamp}.json"
    f.write_text(json.dumps(report, indent=2))
    return f


# ------------------------------------------------------------------ tables
def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "n/a"


def summary_table(report: dict, strategy: str) -> str:
    lines = [f"{'model':<14}{'kind':<12}{'P':>8}{'R':>8}{'F1':>8}{'PR-AUC':>9}{'FPR':>9}"]
    for r in report["results"]:
        t = r["test"][strategy]
        kind = "anomaly" if r["kind"].startswith("anomaly") else "supervised"
        lines.append(f"{r['model']:<14}{kind:<12}{t['precision']:>8.4f}{t['recall']:>8.4f}"
                     f"{t['f1']:>8.4f}{_fmt(t['pr_auc']):>9}{_fmt(t['false_positive_rate']):>9}")
    return "\n".join(lines)


def family_table(report: dict, strategy: str) -> str:
    first = report["results"][0]["test_per_family"][strategy]
    head = f"{'family':<16}{'rows':>8}{'seen':>6}" + "".join(
        f"{r['model'][:12]:>14}" for r in report["results"])
    lines = [head]
    for fam, info in first.items():
        row = f"{fam:<16}{info['rows']:>8}{('yes' if info['seen_in_train'] else 'NO'):>6}"
        for r in report["results"]:
            row += f"{r['test_per_family'][strategy][fam]['flagged_rate']:>14.4f}"
        lines.append(row)
    return "\n".join(lines)


STRATEGY_TITLES = {
    "val_f1": "Threshold = best F1 on validation",
    "fpr_budget": "Threshold = false-positive budget on validation benign",
}
