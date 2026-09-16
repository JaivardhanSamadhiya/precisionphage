#!/usr/bin/env python3
"""Step 6: precompute leakage-free homology + CRISPR pair features and cache.

These features depend only on each pair's two genomes (never on labels or other
pairs), so they are computed once for every covered (phage, host) pair and cached
to data/interim_v2/seq_pair_features.parquet. The assembly layer then merges them
into the edge-feature block for all downstream experiments.

Run (from /tmp, then cd in):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/06_seq_features.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precisionphage.data import GenomeIndex, load_interactions  # noqa: E402
from precisionphage.features.seqmatch import (  # noqa: E402
    compute_pair_features, pair_feature_cols,
)
from precisionphage.features.cache import pair_feature_cache_metadata  # noqa: E402
from precisionphage.utils import (  # noqa: E402
    ensure_dirs, get_logger, limit_threads, load_config, set_determinism,
)

log = get_logger("seq_features")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_determinism(cfg["seed"])
    ensure_dirs(cfg)
    limit_threads(1)

    ds = load_interactions(cfg)
    df = ds.df.reset_index(drop=True)
    pidx = GenomeIndex([cfg["paths"]["phage_fasta_dir"]])
    hidx = GenomeIndex([cfg["paths"]["host_fasta_dir"]])
    phset = {p for p in df["phage"].unique() if pidx.resolve(p) is not None}
    hset = {h for h in df["host"].unique() if hidx.resolve(h) is not None}
    cov = df[df["phage"].isin(phset) & df["host"].isin(hset)].reset_index(drop=True)
    log.info("Computing pair features for %d covered pairs", len(cov))

    pf = compute_pair_features(cov[["phage", "host"]].drop_duplicates(), cfg,
                               n_jobs=cfg["compute"]["n_jobs"])

    out = cfg["paths"]["interim_dir"] / "seq_pair_features.parquet"
    try:
        pf.to_parquet(out)
        log.info("Wrote %s", out)
    except Exception as e:
        out = cfg["paths"]["interim_dir"] / "seq_pair_features.csv"
        pf.to_csv(out, index=False)
        log.info("Parquet unavailable (%s); wrote %s", e, out)
    meta = pair_feature_cache_metadata(cov, cfg, pidx, hidx)
    meta_path = cfg["paths"]["interim_dir"] / "seq_pair_features.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Wrote content-addressed cache metadata %s", meta_path)

    # quick descriptive: how informative are these features vs label?
    lab = cov.merge(pf, on=["phage", "host"], how="left")
    for c in pair_feature_cols(cfg):
        if c in lab.columns:
            pos = lab.loc[lab["label"] == 1, c].mean()
            neg = lab.loc[lab["label"] == 0, c].mean()
            log.info("  %-18s mean(pos)=%.4f mean(neg)=%.4f", c, pos, neg)


if __name__ == "__main__":
    main()
