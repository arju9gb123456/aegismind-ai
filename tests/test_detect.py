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


def test_threshold_and_metrics():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.9])
    thr = baseline.best_f1_threshold(y, s)
    m = baseline.binary_metrics(y, s, thr)
    assert 0 <= thr <= 1
    cm = m["confusion_matrix"]
    assert sum(cm.values()) == 4
    assert m["support"] == {"benign": 2, "attack": 2}
    assert m["roc_auc"] == 0.75


def test_single_class_metrics_do_not_crash():
    m = baseline.binary_metrics(np.zeros(5, int), np.linspace(0, 1, 5), 0.5)
    assert m["pr_auc"] is None and m["roc_auc"] is None


@pytest.mark.parametrize("name", baseline.MODEL_NAMES)
def test_each_model_runs(cicids_data, name):
    rep = baseline.run_detectors(cicids_data, [name], verbose=False)
    r = rep["results"][0]
    assert r["model"] == name
    assert 0.0 <= r["test"]["f1"] <= 1.0
    fam = r["test_per_family"]
    assert list(fam)[0] == "Benign"
    assert fam["PortScan"]["seen_in_train"] is False
    assert fam["Benign"]["meaning"] == "false positive rate"
    tc = r["test"]["confusion_matrix"]
    assert sum(tc.values()) == len(cicids_data.test)


def test_feature_importance_reported(cicids_data):
    rep = baseline.run_detectors(cicids_data, ["random_forest", "logreg", "hist_gb"],
                                 verbose=False)
    by = {r["model"]: r for r in rep["results"]}
    assert by["random_forest"]["top_features"][0]["feature"] in cicids_data.features
    assert by["logreg"]["top_features"]
    assert by["hist_gb"]["top_features"] == []


def test_subsample_keeps_all_families(cicids_data):
    rep = baseline.run_detectors(cicids_data, ["decision_tree"], max_train_rows=50,
                                 verbose=False)
    assert rep["train_rows_used"] <= 60


def test_deterministic(cicids_data):
    a = baseline.run_detectors(cicids_data, ["random_forest"], seed=1, verbose=False)
    b = baseline.run_detectors(cicids_data, ["random_forest"], seed=1, verbose=False)
    assert a["results"][0]["test"] == b["results"][0]["test"]


def test_unsw_detectors(tmp_path):
    df = load_unsw_nb15(write_unsw(tmp_path / "raw"))
    data = preprocess.prepare(df, preprocess.PrepConfig(
        dataset="unsw-nb15", split="official", categorical=["proto", "service", "state"]))
    rep = baseline.run_detectors(data, ["logreg"], verbose=False)
    assert rep["results"][0]["test"]["support"]["attack"] > 0
    assert "Benign" in baseline.family_table(rep)


def test_cli_detect(tmp_path, capsys):
    raw = write_cicids(tmp_path / "raw")
    proc = tmp_path / "proc"
    main(["prepare", "cicids2017", "--raw", str(raw), "--out", str(proc)])
    main(["detect", "cicids2017", "--data", str(proc), "--models", "logreg,decision_tree",
          "--out", str(tmp_path / "exp")])
    out = capsys.readouterr().out
    assert "Test detection rate by family" in out and "NO" in out
    files = list((tmp_path / "exp").glob("detectors_cicids2017_*.json"))
    assert len(files) == 1
    rep = json.loads(files[0].read_text())
    assert [r["model"] for r in rep["results"]] == ["logreg", "decision_tree"]


def test_cli_detect_without_prepared_data(tmp_path):
    with pytest.raises(SystemExit):
        main(["detect", "cicids2017", "--data", str(tmp_path / "nothing")])
