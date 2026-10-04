"""CICIDS2017 loader (MachineLearningCVE CSV release).

Download the "MachineLearningCSV" files from the Canadian Institute for
Cybersecurity (https://www.unb.ca/cic/datasets/ids-2017.html) and place the
eight ``*.pcap_ISCX.csv`` files in ``data/raw/cicids2017/``.

Known quirks handled here:
* header names have leading spaces (`` Destination Port``)
* ``Fwd Header Length`` appears twice (pandas renames the 2nd to ``.1``)
* ``Flow Bytes/s`` / ``Flow Packets/s`` contain ``Infinity`` and NaN
* web-attack labels contain a mis-encoded dash (``Web Attack \x96 XSS``)
* the "GeneratedLabelledFlows" release adds identifiers (IPs, Flow ID,
  Timestamp) which leak information and are dropped by default
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DAY_PATTERNS = {
    "monday": "Monday",
    "tuesday": "Tuesday",
    "wednesday": "Wednesday",
    "thursday": "Thursday",
    "friday": "Friday",
}

# raw label (normalised) -> family used for multi-class experiments
LABEL_FAMILY = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
}

IDENTIFIER_COLUMNS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]


WEB_ATTACK_RE = re.compile(r"^Web Attack.*?(Brute Force|XSS|Sql Injection)\s*$", re.IGNORECASE)
WEB_ATTACK_CANON = {"brute force": "Brute Force", "xss": "XSS", "sql injection": "Sql Injection"}


def normalise_label(label: str) -> str:
    """Strip spaces and repair the web-attack dash, whatever encoding mangled it.

    Seen in the wild: ``\\x96`` (cp1252 read as latin-1), ``\\ufffd`` (replacement
    char), ``ï¿½`` (UTF-8 replacement char read as latin-1), plain ``-``.
    """
    s = str(label).strip()
    m = WEB_ATTACK_RE.match(s)
    if m:
        return f"Web Attack - {WEB_ATTACK_CANON[m.group(1).lower()]}"
    return s


def day_from_filename(name: str) -> str:
    low = name.lower()
    for key, day in DAY_PATTERNS.items():
        if key in low:
            return day
    return "Unknown"


def _read_one(path: Path, sample_frac: float | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", skipinitialspace=True, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].astype(np.float32)
    df["day"] = day_from_filename(path.name)
    df["source_file"] = path.name
    return df


def load_cicids2017(raw_dir: str | Path, sample_frac: float | None = None, seed: int = 42,
                    keep_identifiers: bool = False) -> pd.DataFrame:
    """Load and concatenate all CICIDS2017 CSVs in ``raw_dir``.

    ``sample_frac`` keeps a random fraction of each file (useful on 8 GB laptops).
    Returns a frame with feature columns plus ``label``, ``label_family``,
    ``is_attack``, ``day`` and ``source_file``.
    """
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found in {raw_dir}")
    df = pd.concat([_read_one(f, sample_frac, seed) for f in files], ignore_index=True)
    if "Label" not in df.columns:
        raise ValueError("expected a 'Label' column (CICIDS2017 MachineLearningCVE format)")
    df["label"] = df.pop("Label").map(normalise_label)
    unknown = sorted(set(df["label"]) - set(LABEL_FAMILY))
    if unknown:
        raise ValueError(f"unmapped CICIDS2017 labels: {unknown}")
    df["label_family"] = df["label"].map(LABEL_FAMILY)
    df["is_attack"] = (df["label_family"] != "Benign").astype(np.int8)
    if not keep_identifiers:
        df = df.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in df.columns])
    return df
