import json

import numpy as np
import pandas as pd
import pytest

from aegismind.cli import main
from aegismind.data import preprocess
from aegismind.data.cicids import day_from_filename, load_cicids2017, normalise_label
from aegismind.data.unsw import load_unsw_nb15

from .fixtures import write_cicids, write_unsw


# ---------------------------------------------------------------- cicids
def test_label_normalisation():
    assert normalise_label(" Web Attack \x96 XSS") == "Web Attack - XSS"
    assert normalise_label("Web Attack � Sql Injection") == "Web Attack - Sql Injection"
    assert normalise_label("BENIGN ") == "BENIGN"
    assert day_from_filename("Wednesday-workingHours.pcap_ISCX.csv") == "Wednesday"


def test_load_cicids_handles_quirks(tmp_path):
    df = load_cicids2017(write_cicids(tmp_path / "raw"))
    assert "Destination Port" in df.columns  # leading space stripped
    assert "Fwd Header Length.1" in df.columns  # duplicate header kept distinct
    assert set(df["day"]) == {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    assert "WebAttack" in set(df["label_family"])
    assert df.loc[df["label"] == "BENIGN", "is_attack"].eq(0).all()
    assert np.isinf(df["Flow Bytes/s"]).any()  # cleaning happens in prepare()


def test_unknown_label_rejected(tmp_path):
    raw = write_cicids(tmp_path / "raw")
    f = raw / "Monday-WorkingHours.pcap_ISCX.csv"
    text = f.read_bytes().decode("latin-1").splitlines()
    text[1] = text[1].rsplit(",", 1)[0] + ",MYSTERY"
    f.write_bytes("\n".join(text).encode("latin-1"))
    with pytest.raises(ValueError, match="unmapped"):
        load_cicids2017(raw)


def test_prepare_cicids_day_split_is_leakage_safe(tmp_path):
    df = load_cicids2017(write_cicids(tmp_path / "raw"))
    data = preprocess.prepare(df, preprocess.PrepConfig(dataset="cicids2017"))
    m = data.metadata

    assert m["cleaning"]["rows_dropped_nan_or_inf"] > 0
    assert m["cleaning"]["rows_dropped_duplicates"] > 0
    assert "Bwd PSH Flags" in m["cleaning"]["constant_columns_dropped"]
    assert m["splits"]["train"]["label_family"].keys() >= {"Benign", "BruteForce", "DoS"}
    assert set(m["splits"]["test"]["label_family"]) == {"Benign", "PortScan", "DDoS"}
    assert set(m["families_unseen_in_train"]["test"]) == {"PortScan", "DDoS"}

    # scaler fitted on train only: train features ~N(0,1), test generally not
    tr = data.train[data.features]
    assert np.allclose(tr.mean(), 0, atol=1e-4)
    assert np.allclose(tr.std(ddof=0), 1, atol=1e-3)
    assert m["scaler"]["mean"].keys() == set(data.features)

    # no exact feature-vector overlap between train and test
    tk = set(pd.util.hash_pandas_object(data.train[data.features], index=False))
    assert not pd.util.hash_pandas_object(data.test[data.features], index=False).isin(tk).any()
    assert data.X("train").dtype == np.float32 and len(data.y("test")) == len(data.test)


def test_random_split_warns(tmp_path):
    df = load_cicids2017(write_cicids(tmp_path / "raw"))
    data = preprocess.prepare(df, preprocess.PrepConfig(dataset="cicids2017", split="random"))
    assert "optimistic" in data.metadata["cleaning"]["warning"]
    fams_train = set(data.metadata["splits"]["train"]["label_family"])
    assert fams_train == set(data.metadata["splits"]["test"]["label_family"])


def test_drop_columns_ablation(tmp_path):
    df = load_cicids2017(write_cicids(tmp_path / "raw"))
    cfg = preprocess.PrepConfig(dataset="cicids2017", drop_columns=["Destination Port"])
    assert "Destination Port" not in preprocess.prepare(df, cfg).features


def test_prepare_is_deterministic(tmp_path):
    df = load_cicids2017(write_cicids(tmp_path / "raw"))
    a = preprocess.prepare(df, preprocess.PrepConfig(dataset="cicids2017", split="random"))
    b = preprocess.prepare(df, preprocess.PrepConfig(dataset="cicids2017", split="random"))
    assert a.metadata["fingerprint"] == b.metadata["fingerprint"]


def test_save_and_load_roundtrip(tmp_path):
    df = load_cicids2017(write_cicids(tmp_path / "raw"))
    data = preprocess.prepare(df, preprocess.PrepConfig(dataset="cicids2017"))
    out = preprocess.save(data, tmp_path / "out")
    back = preprocess.load(out)
    assert back.features == data.features
    assert len(back.test) == len(data.test)
    np.testing.assert_allclose(back.X("train"), data.X("train"), rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------ unsw
def test_unsw_official_split_and_categoricals(tmp_path):
    df = load_unsw_nb15(write_unsw(tmp_path / "raw"))
    assert set(df["partition"]) == {"train", "test"}
    assert df.attrs["label_mismatch_rows"] == 0
    cfg = preprocess.PrepConfig(dataset="unsw-nb15", split="official",
                                categorical=["proto", "service", "state"])
    data = preprocess.prepare(df, cfg)
    assert any(f.startswith("proto=") for f in data.features)
    assert "service=none" in data.features
    assert not any(f in ("proto", "service", "state") for f in data.features)
    s = data.metadata["splits"]
    assert s["val"]["rows"] > 0 and s["test"]["rows"] > 0


# ------------------------------------------------------------------- cli
def test_cli_prepare(tmp_path, capsys):
    raw = write_cicids(tmp_path / "raw")
    main(["prepare", "cicids2017", "--raw", str(raw), "--out", str(tmp_path / "proc")])
    out = capsys.readouterr().out
    assert "families not seen in train" in out
    meta = json.loads((tmp_path / "proc" / "metadata.json").read_text())
    assert meta["config"]["split"] == "day"
