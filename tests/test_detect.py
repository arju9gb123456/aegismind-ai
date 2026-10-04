import json

import numpy as np
import pytest

from aegismind.cli import main
from aegismind.data import preprocess
from aegismind.data.cicids import load_cicids2017
from aegismind.data.unsw import load_unsw_nb15
from aegismind.detect import baseline

from .fixtures import write_cicids, write_unsw


@pytest.fixture(scope="module")
def cicids_data(tmp_path_factory):
    raw = write_cicids(tmp_path_factory.mktemp("raw"), rows_per_label=60)
    return preprocess.prepare(load_cicids2017(raw), preprocess.PrepConfig(dataset="cicids2017"))


# ---------------------------------------------------------------- thresholds
def test_f1_threshold_and_metrics():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.9])
    thr = baseline.best_f1_threshold(y, s)
    m = baseline.binary_metrics(y, s, thr)
    assert 0 <= thr <= 1
    assert sum(m["confusion_matrix"].values()) == 4
    assert m["support"] == {"benign": 2, "attack": 2}
    assert m["roc_auc"] == 0.75


def test_fpr_budget_threshold_respects_budget():
    rng = np.random.default_rng(0)
    benign = rng.normal(size=10_000)
    for budget in (0.001, 0.01, 0.05, 0.2):
        thr = baseline.fpr_budget_threshold(benign, budget)
        flagged = (benign >= thr).mean()
        assert flagged <= budget + 1e-12
        assert flagged >= budget - 0.001  # uses (almost) the whole budget


def test_fpr_budget_with_ties_never_exceeds_budget():
    benign = np.array([0.0] * 95 + [1.0] * 5)
    thr = baseline.fpr_budget_threshold(benign, 0.01)
    assert (benign >= thr).mean() == 0.0  # the 5% tie block can't fit in 1%
    thr = baseline.fpr_budget_threshold(benign, 0.05)
    assert (benign >= thr).mean() == 0.05


def test_fpr_budget_needs_benign():
    with pytest.raises(ValueError):
        baseline.fpr_budget_threshold(np.array([]), 0.01)


def test_single_class_metrics_do_not_crash():
    m = baseline.binary_metrics(np.zeros(5, int), np.linspace(0, 1, 5), 0.5)
    assert m["pr_auc"] is None and m["roc_auc"] is None


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("name", baseline.MODEL_NAMES)
def test_each_model_runs_both_strategies(cicids_data, name):
    rep = baseline.run_detectors(cicids_data, [name], verbose=False)
    r = rep["results"][0]
    assert r["model"] == name
    for strat in baseline.STRATEGIES:
        t = r["test"][strat]
        assert 0.0 <= t["f1"] <= 1.0
        assert sum(t["confusion_matrix"].values()) == len(cicids_data.test)
        fam = r["test_per_family"][strat]
        assert list(fam)[0] == "Benign"
        assert fam["PortScan"]["seen_in_train"] is False
    # the FPR budget is honoured on validation by construction
    assert r["val"]["fpr_budget"]["false_positive_rate"] <= rep["fpr_budget"] + 1e-9


def test_iforest_trains_on_benign_only(cicids_data):
    rep = baseline.run_detectors(cicids_data, ["iforest"], verbose=False)
    r = rep["results"][0]
    assert r["kind"] == "anomaly (benign-only)"
    assert r["train_rows_used"] == int((cicids_data.train["is_attack"] == 0).sum())
    assert r["top_features"] == []


def test_iforest_benign_cap(cicids_data):
    rep = baseline.run_detectors(cicids_data, ["iforest"], iforest_max_benign=30,
                                 verbose=False)
    assert rep["results"][0]["train_rows_used"] == 30


def test_scores_higher_means_more_attack_like():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 4)).astype(np.float32)
    m = baseline.make_model("iforest", 0).fit(X)
    normal = baseline.scores_of(m, np.zeros((1, 4), np.float32))[0]
    weird = baseline.scores_of(m, np.full((1, 4), 8, np.float32))[0]
    assert weird > normal


def test_feature_importance_reported(cicids_data):
    rep = baseline.run_detectors(cicids_data, ["random_forest", "logreg", "hist_gb"],
                                 verbose=False)
    by = {r["model"]: r for r in rep["results"]}
    assert by["random_forest"]["top_features"][0]["feature"] in cicids_data.features
    assert by["logreg"]["top_features"]
    assert by["hist_gb"]["top_features"] == []


def test_subsample(cicids_data):
    rep = baseline.run_detectors(cicids_data, ["decision_tree"], max_train_rows=50,
                                 verbose=False)
    assert rep["results"][0]["train_rows_used"] <= 60


def test_invalid_budget(cicids_data):
    with pytest.raises(ValueError):
        baseline.run_detectors(cicids_data, ["logreg"], fpr_budget=1.5, verbose=False)


def test_deterministic(cicids_data):
    a = baseline.run_detectors(cicids_data, ["random_forest", "iforest"], seed=1, verbose=False)
    b = baseline.run_detectors(cicids_data, ["random_forest", "iforest"], seed=1, verbose=False)
    assert [r["test"] for r in a["results"]] == [r["test"] for r in b["results"]]


def test_unsw_detectors(tmp_path):
    df = load_unsw_nb15(write_unsw(tmp_path / "raw"))
    data = preprocess.prepare(df, preprocess.PrepConfig(
        dataset="unsw-nb15", split="official", categorical=["proto", "service", "state"]))
    rep = baseline.run_detectors(data, ["logreg", "iforest"], verbose=False)
    assert rep["results"][0]["test"]["val_f1"]["support"]["attack"] > 0
    assert "Benign" in baseline.family_table(rep, "fpr_budget")


# --------------------------------------------------------------------- cli
def test_cli_detect(tmp_path, capsys):
    raw = write_cicids(tmp_path / "raw")
    proc = tmp_path / "proc"
    main(["prepare", "cicids2017", "--raw", str(raw), "--out", str(proc)])
    main(["detect", "cicids2017", "--data", str(proc), "--models", "logreg,iforest",
          "--fpr-budget", "0.02", "--out", str(tmp_path / "exp")])
    out = capsys.readouterr().out
    assert "best F1 on validation" in out
    assert "false-positive budget on validation benign (2.0%)" in out
    assert "anomaly" in out and "NO" in out
    files = list((tmp_path / "exp").glob("detectors_cicids2017_*.json"))
    assert len(files) == 1
    rep = json.loads(files[0].read_text())
    assert [r["model"] for r in rep["results"]] == ["logreg", "iforest"]
    assert rep["fpr_budget"] == 0.02


def test_cli_detect_without_prepared_data(tmp_path):
    with pytest.raises(SystemExit):
        main(["detect", "cicids2017", "--data", str(tmp_path / "nothing")])
