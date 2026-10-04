"""Leakage-aware preprocessing: clean -> split -> fit on train only -> save.

Safeguards (report section 11):
* split BEFORE fitting anything; scaler, constant-column removal and
  categorical vocabularies are fitted on the training split only
* default split is by capture day / official partition, not random rows
* rows in val/test that exactly duplicate a training row are removed and counted
* every run writes ``metadata.json`` with counts, dropped rows, feature list,
  scaler statistics, config and a data fingerprint
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLUMNS = ["label", "label_family", "is_attack"]
META_COLUMNS = ["day", "source_file", "partition"]

DEFAULT_DAY_SPLIT = {
    "train": ["Monday", "Tuesday", "Wednesday"],
    "val": ["Thursday"],
    "test": ["Friday"],
}


@dataclass
class PrepConfig:
    dataset: str
    split: str = "day"  # day | official | random
    day_split: dict = field(default_factory=lambda: dict(DEFAULT_DAY_SPLIT))
    val_frac: float = 0.15  # used by random / official (carved from train)
    test_frac: float = 0.15  # used by random
    drop_columns: list = field(default_factory=list)
    categorical: list = field(default_factory=list)
    max_categories: int = 20
    scale: bool = True
    drop_cross_split_duplicates: bool = True
    seed: int = 42


@dataclass
class PreparedData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    metadata: dict

    def X(self, split: str) -> np.ndarray:
        return getattr(self, split)[self.features].to_numpy(dtype=np.float32)

    def y(self, split: str, target: str = "is_attack") -> np.ndarray:
        return getattr(self, split)[target].to_numpy()


# ----------------------------------------------------------------- helpers
def _feature_columns(df: pd.DataFrame, cfg: PrepConfig) -> list[str]:
    skip = set(LABEL_COLUMNS + META_COLUMNS + cfg.drop_columns)
    return [c for c in df.columns if c not in skip]


def _stratified_split(df: pd.DataFrame, frac: float, seed: int, by: str = "label_family"):
    rng = np.random.default_rng(seed)
    held = []
    for _, idx in df.groupby(by).groups.items():
        idx = np.array(list(idx))
        n = int(round(len(idx) * frac))
        if len(idx) > 1 and n == 0:
            n = 1
        held.extend(rng.choice(idx, size=n, replace=False).tolist())
    mask = df.index.isin(held)
    return df[~mask], df[mask]


def _fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


# -------------------------------------------------------------------- main
def prepare(df: pd.DataFrame, cfg: PrepConfig) -> PreparedData:
    report: dict = {"rows_loaded": int(len(df))}
    df = df.copy()

    # 1. clean (row-wise only: nothing is learned from the data here)
    feats = _feature_columns(df, cfg)
    numeric = [c for c in feats if c not in cfg.categorical]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    bad = df[numeric].isna().any(axis=1)
    report["rows_dropped_nan_or_inf"] = int(bad.sum())
    df = df[~bad]
    before = len(df)
    df = df.drop_duplicates(subset=feats + ["label"])
    report["rows_dropped_duplicates"] = int(before - len(df))
    df = df.reset_index(drop=True)

    # 2. split
    if cfg.split == "day":
        parts = {k: df[df["day"].isin(v)] for k, v in cfg.day_split.items()}
    elif cfg.split == "official":
        rest, test = df[df["partition"] == "train"], df[df["partition"] == "test"]
        train, val = _stratified_split(rest, cfg.val_frac, cfg.seed)
        parts = {"train": train, "val": val, "test": test}
    elif cfg.split == "random":
        report["warning"] = ("random row split is optimistic for flow data; "
                             "use only as a comparison, never as the headline result")
        rest, test = _stratified_split(df, cfg.test_frac, cfg.seed)
        train, val = _stratified_split(rest, cfg.val_frac / (1 - cfg.test_frac), cfg.seed + 1)
        parts = {"train": train, "val": val, "test": test}
    else:
        raise ValueError(f"unknown split: {cfg.split}")
    for k in ("train", "val", "test"):
        if parts.get(k) is None or parts[k].empty:
            raise ValueError(f"split '{k}' is empty; check split configuration")
    train, val, test = (parts[k].copy() for k in ("train", "val", "test"))

    # 3. remove val/test rows identical to a training row (leakage)
    if cfg.drop_cross_split_duplicates:
        train_keys = set(pd.util.hash_pandas_object(train[feats], index=False))
        for name in ("val", "test"):
            part = val if name == "val" else test
            keys = pd.util.hash_pandas_object(part[feats], index=False)
            dup = keys.isin(train_keys).to_numpy()
            report[f"{name}_rows_dropped_overlap_with_train"] = int(dup.sum())
            if name == "val":
                val = part[~dup]
            else:
                test = part[~dup]

    # 4. fit on TRAIN only: categorical vocab, constant columns, scaler
    vocab: dict[str, list[str]] = {}
    for c in cfg.categorical:
        top = train[c].value_counts().index[: cfg.max_categories].tolist()
        vocab[c] = top
        for part in (train, val, test):
            vals = part[c].where(part[c].isin(top), "__other__")
            for v in top + ["__other__"]:
                part[f"{c}={v}"] = (vals == v).astype(np.float32)
            part.drop(columns=[c], inplace=True)
    feats = _feature_columns(train, cfg)
    constant = [c for c in feats if train[c].nunique(dropna=False) <= 1]
    feats = [c for c in feats if c not in constant]
    report["constant_columns_dropped"] = constant

    scaler = {}
    if cfg.scale:
        mean = train[feats].mean()
        std = train[feats].std(ddof=0).replace(0, 1.0)
        for part in (train, val, test):
            part[feats] = ((part[feats] - mean) / std).astype(np.float32)
        scaler = {"mean": mean.round(8).to_dict(), "std": std.round(8).to_dict()}

    def counts(p: pd.DataFrame) -> dict:
        return {"rows": int(len(p)),
                "label_family": {k: int(v) for k, v in p["label_family"].value_counts().items()},
                "attack_rate": round(float(p["is_attack"].mean()), 4) if len(p) else 0.0}

    fam_train = set(train["label_family"])
    unseen = {n: sorted(set(p["label_family"]) - fam_train) for n, p in (("val", val), ("test", test))}

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "cleaning": report,
        "splits": {"train": counts(train), "val": counts(val), "test": counts(test)},
        "families_unseen_in_train": unseen,
        "n_features": len(feats),
        "features": feats,
        "categorical_vocab": vocab,
        "scaler": scaler,
        "fingerprint": {n: _fingerprint(p[feats + LABEL_COLUMNS])
                        for n, p in (("train", train), ("val", val), ("test", test))},
    }
    if "label_mismatch_rows" in df.attrs:
        metadata["cleaning"]["official_label_mismatch_rows"] = df.attrs["label_mismatch_rows"]
    keep = feats + LABEL_COLUMNS
    return PreparedData(train=train[keep].reset_index(drop=True),
                        val=val[keep].reset_index(drop=True),
                        test=test[keep].reset_index(drop=True),
                        features=feats, metadata=metadata)


def save(data: PreparedData, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        getattr(data, name).to_csv(out / f"{name}.csv.gz", index=False, compression="gzip")
    (out / "metadata.json").write_text(json.dumps(data.metadata, indent=2))
    return out


def load(out_dir: str | Path) -> PreparedData:
    out = Path(out_dir)
    meta = json.loads((out / "metadata.json").read_text())
    parts = {n: pd.read_csv(out / f"{n}.csv.gz") for n in ("train", "val", "test")}
    return PreparedData(**parts, features=meta["features"], metadata=meta)
