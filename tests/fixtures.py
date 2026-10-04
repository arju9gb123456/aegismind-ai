"""Tiny files that mimic the real dataset formats, including their quirks."""
from __future__ import annotations

from pathlib import Path

import numpy as np

CICIDS_HEADER = [" Destination Port", " Flow Duration", " Total Fwd Packets",
                 " Total Backward Packets", "Total Length of Fwd Packets", "Flow Bytes/s",
                 " Flow Packets/s", " Fwd Header Length", " Fwd Header Length",
                 " Bwd PSH Flags", " Label"]

CICIDS_DAYS = {
    "Monday-WorkingHours.pcap_ISCX.csv": ["BENIGN"],
    "Tuesday-WorkingHours.pcap_ISCX.csv": ["BENIGN", "FTP-Patator", "SSH-Patator"],
    "Wednesday-workingHours.pcap_ISCX.csv": ["BENIGN", "DoS Hulk", "DoS slowloris"],
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv":
        ["BENIGN", "Web Attack \x96 Brute Force", "Web Attack \x96 XSS"],
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": ["BENIGN", "PortScan", "DDoS"],
}


def write_cicids(raw_dir: Path, rows_per_label: int = 40, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname, labels in CICIDS_DAYS.items():
        lines = [",".join(CICIDS_HEADER)]
        for li, label in enumerate(labels):
            for i in range(rows_per_label):
                port = 80 if li == 0 else 21 + li
                dur = int(rng.integers(1, 10_000)) * (li + 1)
                fwd, bwd = int(rng.integers(1, 50)), int(rng.integers(0, 50))
                length = int(rng.integers(0, 5000))
                fbs = "Infinity" if i == 0 else f"{rng.random() * 1e5:.3f}"
                fps = "NaN" if i == 1 else f"{rng.random() * 1e3:.3f}"
                hdr = int(rng.integers(20, 400))
                lines.append(f"{port},{dur},{fwd},{bwd},{length},{fbs},{fps},{hdr},{hdr},0,{label}")
            # one exact duplicate row per label
            lines.append(lines[-1])
        (raw_dir / fname).write_bytes("\n".join(lines).encode("latin-1"))
    return raw_dir


def write_unsw(raw_dir: Path, rows: int = 300, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cats = ["Normal", "Normal", "Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance"]
    for fname, offset in (("UNSW_NB15_training-set.csv", 0), ("UNSW_NB15_testing-set.csv", 1)):
        lines = ["id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,attack_cat,label"]
        for i in range(rows):
            cat = cats[(i + offset) % len(cats)]
            proto = ["tcp", "udp", "arp", "ospf"][i % 4]
            service = ["-", "http", "dns", "ftp"][(i // 3) % 4]
            state = ["FIN", "INT", "CON"][i % 3]
            lab = 0 if cat == "Normal" else 1
            lines.append(f"{i},{rng.random():.6f},{proto},{service},{state},"
                         f"{rng.integers(1, 100)},{rng.integers(0, 100)},"
                         f"{rng.integers(0, 10000)},{rng.integers(0, 10000)},"
                         f"{rng.random() * 1e4:.3f},{cat},{lab}")
        (raw_dir / fname).write_text("\n".join(lines))
    return raw_dir
