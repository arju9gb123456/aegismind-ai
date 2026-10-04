"""UNSW-NB15 loader (official training/testing partition CSVs).

Download ``UNSW_NB15_training-set.csv`` and ``UNSW_NB15_testing-set.csv`` from
https://research.unsw.edu.au/projects/unsw-nb15-dataset and place them in
``data/raw/unsw-nb15/``.

The schema differs from CICIDS2017; features are NOT assumed equivalent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ["proto", "service", "state"]
DROP = ["id"]


def _read(path: Path, sample_frac: float | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    return df


def load_unsw_nb15(raw_dir: str | Path, sample_frac: float | None = None,
                   seed: int = 42) -> pd.DataFrame:
    """Return one frame with a ``partition`` column (official ``train`` / ``test``)."""
    raw = Path(raw_dir)
    parts = []
    for name, part in (("UNSW_NB15_training-set.csv", "train"),
                       ("UNSW_NB15_testing-set.csv", "test")):
        f = raw / name
        if not f.exists():
            raise FileNotFoundError(f"missing {f}")
        d = _read(f, sample_frac, seed)
        d["partition"] = part
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    df["attack_cat"] = df["attack_cat"].replace({"": "Normal", "-": "Normal"})
    # the official files carry a numeric 0/1 'label'; replace it with our schema
    official_binary = df.pop("label") if "label" in df.columns else None
    df["label"] = df.pop("attack_cat")
    df["label_family"] = df["label"].replace({"Normal": "Benign"})
    df["is_attack"] = (df["label"] != "Normal").astype(np.int8)
    if official_binary is not None:
        mismatch = int((official_binary.astype(int).values != df["is_attack"].values).sum())
        df.attrs["label_mismatch_rows"] = mismatch  # reported in metadata
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"-": "none"})
    num = df.select_dtypes(include=[np.number]).columns.difference(["is_attack"])
    df[num] = df[num].astype(np.float32)
    return df
