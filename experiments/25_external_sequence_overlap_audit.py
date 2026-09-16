#!/usr/bin/env python3
"""Audit exact sequence and identifier overlap for the held-out Staph study."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def sequence_digest(path: Path) -> tuple[str, int]:
    sequence = "".join(
        line.strip().upper()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not line.startswith(">")
    )
    return hashlib.sha256(sequence.encode("ascii")).hexdigest(), len(sequence)


def compare(kind: str, external_dir: Path, training_dir: Path) -> None:
    training = {}
    for path in training_dir.iterdir():
        if path.is_file():
            digest, length = sequence_digest(path)
            training[digest] = (path.name, length)

    external = [path for path in external_dir.iterdir() if path.is_file()]
    hits = []
    for path in external:
        digest, length = sequence_digest(path)
        if digest in training:
            hits.append((path.name, training[digest][0], length))

    print(f"{kind}: external={len(external)}, exact_sequence_overlap={len(hits)}")
    for hit in hits:
        print("  ", hit)


def main() -> None:
    upstream = ROOT / "external" / "upstream_vhip_tool" / "example"
    frozen = ROOT / "external" / "phist_run"
    compare("hosts", upstream / "host_genomes", frozen / "hosts")
    compare("phages", upstream / "virus_genomes", frozen / "phages")

    raw = pd.read_csv(ROOT / "data" / "raw" / "VirusHostInter.csv")
    ncbi = raw.loc[raw["data"] == "NCBI_HR"]
    staph = raw.loc[raw["data"] == "StaphStudy"]
    host_overlap = sorted(set(ncbi["hostname"]) & set(staph["hostname"]))
    phage_overlap = sorted(set(ncbi["phagename"]) & set(staph["phagename"]))
    print(f"host_identifier_overlap={len(host_overlap)}: {host_overlap}")
    print(f"phage_identifier_overlap={len(phage_overlap)}: {phage_overlap}")


if __name__ == "__main__":
    main()
