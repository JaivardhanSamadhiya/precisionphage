#!/usr/bin/env python3
"""Fetch NahantCollection phage genomes (GenBank MG accessions) into data/phages.

The Nahant Vibrio cross-infection matrix (Kauffman et al., Nature 2018) is the
densest real-negative dataset we have; its phages are named by GenBank accession
so they can be fetched directly. Hosts already resolve.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from precisionphage.utils import get_logger, load_config  # noqa: E402

log = get_logger("fetch_nahant")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    from Bio import Entrez
    Entrez.email = cfg.get("fetch", {}).get("ncbi_email", "research@example.com")

    df = pd.read_csv(cfg["paths"]["vhi_csv"])
    nahant = df[df["data"] == "NahantCollection"]["phagename"].astype(str).unique()
    accs = sorted({a.strip() for a in nahant if a and a[0].isalpha()})
    out_dir = Path(cfg["paths"]["phage_fasta_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Nahant phage accessions: %d", len(accs))

    n_new = n_have = n_fail = 0
    for i, acc in enumerate(accs):
        # store under the slug used by the resolver (accession with dot)
        target = out_dir / f"{acc}.fasta"
        if target.exists() and target.stat().st_size > 200:
            n_have += 1
            continue
        try:
            with Entrez.efetch(db="nuccore", id=acc, rettype="fasta",
                               retmode="text") as h:
                seq = h.read()
            if seq and len(seq) > 200:
                target.write_text(seq, encoding="utf-8")
                n_new += 1
            else:
                n_fail += 1
        except Exception as e:
            n_fail += 1
            log.warning("efetch failed for %s: %s", acc, e)
        if (i + 1) % 25 == 0:
            log.info("  %d/%d (new=%d have=%d fail=%d)", i + 1, len(accs),
                     n_new, n_have, n_fail)
        time.sleep(0.34)
    log.info("DONE Nahant fetch: new=%d have=%d fail=%d total_acc=%d",
             n_new, n_have, n_fail, len(accs))


if __name__ == "__main__":
    main()
