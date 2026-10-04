"""Baseline intrusion detectors (report section 6.1, Sprint 3).

Transparent, classical models first:
  logreg         Logistic Regression (class-balanced)
  decision_tree  Decision Tree (depth-limited, class-balanced)
  random_forest  Random Forest (class-balanced subsamples)
  hist_gb        Histogram Gradient Boosting (class-balanced)

Protocol (report section 11):
  * fit on TRAIN only
  * choose the decision threshold on VAL (max F1)
  * report TEST once, with the threshold frozen
  * report per-attack-family detection rate, flagging families never seen
    in training (with the CICIDS2017 day split these measure generalisation
    to unseen attack types)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

from aegismind.data.preprocess import PreparedData

MODEL_NAMES = ("logreg", "decision_tree", "random_forest", "hist_gb")


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
    raise ValueError(f"unknown model: {name} (choose from {MODEL_NAMES})")


# ------------------------------------------------------------------ metrics
def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": round(threshold, 6),
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
        rate = float(pred[idx].mean())
        out[fam] = {
            "rows": int(len(idx)),
            "flagged_rate": round(rate, 4),
            "meaning": "false positive rate" if fam == "Benign" else "detection rate",
            "seen_in_train": fam in train_families,
        }
    return dict(sorted(out.items(), key=lambda kv: (kv[0] != "Benign", kv[0])))


def top_features(model, features: list[str], k: int = 15) -> list[dict]:
    if hasattr(model, "feature_importances_"):
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
@dataclass
class DetectorResult:
    model: str
    fit_seconds: float
    val: dict
    test: dict
    test_per_family: dict
    top_features: list

    def to_dict(self) -> dict:
        return self.__dict__


def _subsample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = [g.sample(frac=frac, random_state=seed) if len(g) * frac >= 1 else g
             for _, g in df.groupby("label_family")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def run_detectors(data: PreparedData, models: list[str], seed: int = 42,
                  max_train_rows: int | None = None, verbose: bool = True) -> dict:
    train = _subsample(data.train, max_train_rows, seed)
    X_tr = train[data.features].to_numpy(np.float32)
    y_tr = train["is_attack"].to_numpy()
    X_va, y_va = data.X("val"), data.y("val")
    X_te, y_te = data.X("test"), data.y("test")
    train_fams = set(data.train["label_family"])

    results = []
    for name in models:
        m = make_model(name, seed)
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        s_va = m.predict_proba(X_va)[:, 1]
        thr = best_f1_threshold(y_va, s_va)
        s_te = m.predict_proba(X_te)[:, 1]
        r = DetectorResult(
            model=name, fit_seconds=round(fit_s, 2),
            val=binary_metrics(y_va, s_va, thr),
            test=binary_metrics(y_te, s_te, thr),
            test_per_family=per_family(data.test, s_te, thr, train_fams),
            top_features=top_features(m, data.features),
        )
        results.append(r)
        if verbose:
            t = r.test
            print(f"  {name:<14} fit {fit_s:7.1f}s | val F1 {r.val['f1']:.4f} | test "
                  f"P {t['precision']:.4f} R {t['recall']:.4f} F1 {t['f1']:.4f} "
                  f"PR-AUC {t['pr_auc']} FPR {t['false_positive_rate']}")
    return {
        "experiment": "baseline_detectors",
        "dataset": data.metadata.get("config", {}).get("dataset"),
        "split": data.metadata.get("config", {}).get("split"),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "train_rows_used": int(len(train)),
        "max_train_rows": max_train_rows,
        "n_features": len(data.features),
        "data_fingerprint": data.metadata.get("fingerprint"),
        "threshold_selection": "max F1 on validation split; frozen for test",
        "results": [r.to_dict() for r in results],
    }


def save_report(report: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    f = out / f"detectors_{report['dataset']}_{stamp}.json"
    f.write_text(json.dumps(report, indent=2))
    return f


def summary_table(report: dict) -> str:
    lines = [f"{'model':<14}{'P':>8}{'R':>8}{'F1':>8}{'PR-AUC':>9}{'FPR':>10}"]
    for r in report["results"]:
        t = r["test"]
        pr = f"{t['pr_auc']:.4f}" if t["pr_auc"] is not None else "n/a"
        fpr = f"{t['false_positive_rate']:.4f}" if t["false_positive_rate"] is not None else "n/a"
        lines.append(f"{r['model']:<14}{t['precision']:>8.4f}{t['recall']:>8.4f}"
                     f"{t['f1']:>8.4f}{pr:>9}{fpr:>10}")
    return "\n".join(lines)


def family_table(report: dict) -> str:
    fams = list(report["results"][0]["test_per_family"].keys())
    head = f"{'family':<16}{'rows':>8}{'seen':>6}" + "".join(
        f"{r['model'][:12]:>14}" for r in report["results"])
    lines = [head]
    for fam in fams:
        info = report["results"][0]["test_per_family"][fam]
        row = f"{fam:<16}{info['rows']:>8}{('yes' if info['seen_in_train'] else 'NO'):>6}"
        for r in report["results"]:
            row += f"{r['test_per_family'][fam]['flagged_rate']:>14.4f}"
        lines.append(row)
    return "\n".join(lines)
