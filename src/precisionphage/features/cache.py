"""Provenance metadata for expensive pair-feature caches."""
from __future__ import annotations

import hashlib

from ..data import genome_set_digest


def pair_feature_cache_metadata(cov, cfg, phage_index, host_index) -> dict:
    phages = sorted(cov["phage"].astype(str).unique())
    hosts = sorted(cov["host"].astype(str).unique())
    pair_digest = hashlib.sha256()
    for phage, host in cov[["phage", "host"]].astype(str).itertuples(index=False):
        pair_digest.update(phage.encode("utf-8"))
        pair_digest.update(b"\t")
        pair_digest.update(host.encode("utf-8"))
        pair_digest.update(b"\n")
    features = cfg["features"]
    return {
        "schema": 1,
        "n_pairs": int(len(cov)),
        "pair_order_sha256": pair_digest.hexdigest(),
        "phage_source_sha256": genome_set_digest(phages, phage_index),
        "host_source_sha256": genome_set_digest(hosts, host_index),
        "homology_ks": list(features["homology_ks"]),
        "crispr_repeat_k": int(features["crispr_repeat_k"]),
        "crispr_match_k": int(features["crispr_match_k"]),
        "use_protein_features": bool(features["use_protein_features"]),
        "protein_k": int(features["protein_k"]),
        "protein_min_pep": int(features["protein_min_pep"]),
    }
