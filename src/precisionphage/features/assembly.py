"""Assemble the genome-covered modelling dataset: canonical pairs restricted to
entities with genomes, with node feature matrices, pairwise edge features, and
node-index columns. Shared by the baseline/GNN and the leakage-split experiments
so feature construction never diverges between them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import GenomeIndex, genome_set_digest, load_interactions
from ..utils import get_logger
from .genomic import build_node_features, edge_features_from_spectra, kmer_spectrum
from .cache import pair_feature_cache_metadata

log = get_logger(__name__)

VHI_FEATS = ["k3dist", "k6dist", "GCdiff", "Homology"]


@dataclass
class CoveredDataset:
    df: pd.DataFrame            # covered pairs + pidx/hidx columns
    P_raw: np.ndarray          # [n_phages, D] node features
    H_raw: np.ndarray          # [n_hosts, D] node features
    E_raw: np.ndarray          # [n_pairs, de] pairwise edge features
    phages: list               # node-index-ordered phage names
    hosts: list                # node-index-ordered host names
    phage_index: GenomeIndex
    host_index: GenomeIndex
    edge_cols: list

    def X_flat(self) -> np.ndarray:
        """Per-row [phage_node ‖ host_node ‖ edge] design matrix for trees/MLP."""
        return np.hstack([self.P_raw[self.df["pidx"].to_numpy()],
                          self.H_raw[self.df["hidx"].to_numpy()],
                          self.E_raw]).astype(np.float32)


def build_covered_dataset(cfg: dict) -> CoveredDataset:
    k = cfg["features"]["kmer_k"]
    ds = load_interactions(cfg)
    df = ds.df.reset_index(drop=True)

    pidx_g = GenomeIndex([cfg["paths"]["phage_fasta_dir"]],
                         cache_path=cfg["paths"]["cache_dir"] / "phage_resolution.json")
    hidx_g = GenomeIndex([cfg["paths"]["host_fasta_dir"]],
                         cache_path=cfg["paths"]["cache_dir"] / "host_resolution.json")
    phset = {p for p in df["phage"].unique() if pidx_g.resolve(p) is not None}
    hset = {h for h in df["host"].unique() if hidx_g.resolve(h) is not None}
    cov = df[df["phage"].isin(phset) & df["host"].isin(hset)].reset_index(drop=True)
    log.info("[assembly] covered subset: %d pairs (pos=%d neg=%d) over %d phages,"
             " %d hosts", len(cov), int((cov["label"] == 1).sum()),
             int((cov["label"] == 0).sum()), cov["phage"].nunique(),
             cov["host"].nunique())

    phages = sorted(cov["phage"].unique())
    hosts = sorted(cov["host"].unique())
    p2i = {p: i for i, p in enumerate(phages)}
    h2i = {h: i for i, h in enumerate(hosts)}
    cov["pidx"] = cov["phage"].map(p2i).astype(int)
    cov["hidx"] = cov["host"].map(h2i).astype(int)

    import hashlib
    import json
    names_hash = hashlib.sha256(
        ("\n".join(phages) + "\n--HOSTS--\n" + "\n".join(hosts)).encode("utf-8")
    ).hexdigest()
    node_meta = {
        "schema": 1,
        "names_sha256": names_hash,
        "phage_source_sha256": genome_set_digest(phages, pidx_g),
        "host_source_sha256": genome_set_digest(hosts, hidx_g),
        "kmer_k": int(k),
        "use_codon": bool(cfg["features"]["use_codon"]),
        "use_dinuc": bool(cfg["features"]["use_dinuc"]),
    }
    node_cache = cfg["paths"]["cache_dir"] / "node_features.npz"
    node_meta_path = cfg["paths"]["cache_dir"] / "node_features.meta.json"
    cached_meta = (json.loads(node_meta_path.read_text(encoding="utf-8"))
                   if node_meta_path.exists() else None)
    if node_cache.exists() and cached_meta == node_meta:
        cached = np.load(node_cache)
        P_raw, H_raw = cached["phage"], cached["host"]
        if P_raw.shape[0] != len(phages) or H_raw.shape[0] != len(hosts):
            raise RuntimeError("node feature cache shape does not match entity order")
        log.info("[assembly] loaded content-matched node feature cache")
    else:
        P_raw, _ = build_node_features(phages, pidx_g, k=k,
                                       use_codon=cfg["features"]["use_codon"],
                                       use_dinuc=cfg["features"]["use_dinuc"],
                                       n_workers=cfg["features"]["n_workers"])
        H_raw, _ = build_node_features(hosts, hidx_g, k=k,
                                       use_codon=cfg["features"]["use_codon"],
                                       use_dinuc=cfg["features"]["use_dinuc"],
                                       n_workers=cfg["features"]["n_workers"])
        np.savez_compressed(node_cache, phage=P_raw, host=H_raw)
        node_meta_path.write_text(json.dumps(node_meta, indent=2), encoding="utf-8")
        log.info("[assembly] wrote content-addressed node feature cache")

    kdim = int(kmer_spectrum("ACGT" * k, k).shape[0])
    P_spec, H_spec = P_raw[:, :kdim], H_raw[:, :kdim]
    pi, hi = cov["pidx"].to_numpy(), cov["hidx"].to_numpy()
    recomputed = np.zeros((len(cov), 4), dtype=np.float32)
    for r in range(len(cov)):
        recomputed[r] = edge_features_from_spectra(P_spec[pi[r]], H_spec[hi[r]])
    vhi = [c for c in VHI_FEATS if c in cov.columns]
    E_vhi = cov[vhi].to_numpy(dtype=np.float32) if vhi else np.zeros((len(cov), 0))
    E_raw = np.hstack([recomputed, E_vhi]).astype(np.float32)
    edge_cols = ["cos_dist", "l1", "pearson", "jaccard"] + vhi

    # Optional leakage-free pair features (homology + CRISPR), precomputed and
    # cached by experiments/06_seq_features.py. Merged on (phage, host).
    if cfg["features"].get("use_seq_pair_features", False):
        cache = cfg["paths"]["interim_dir"] / "seq_pair_features.parquet"
        cache_csv = cfg["paths"]["interim_dir"] / "seq_pair_features.csv"
        meta_path = cfg["paths"]["interim_dir"] / "seq_pair_features.meta.json"
        pf = None
        if cache.exists():
            pf = pd.read_parquet(cache)
        elif cache_csv.exists():
            pf = pd.read_csv(cache_csv)
        if pf is not None:
            if not meta_path.exists():
                raise RuntimeError(
                    "pair-feature cache has no provenance metadata; rerun "
                    "experiments/06_seq_features.py")
            observed_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_meta = pair_feature_cache_metadata(cov, cfg, pidx_g, hidx_g)
            if observed_meta != expected_meta:
                raise RuntimeError(
                    "pair-feature cache does not match the frozen FASTAs/config; "
                    "rerun experiments/06_seq_features.py")
            feat_cols = [c for c in pf.columns if c not in ("phage", "host")]
            merged = cov[["phage", "host"]].merge(pf, on=["phage", "host"],
                                                  how="left")
            extra = merged[feat_cols].to_numpy(dtype=np.float32)
            extra = np.nan_to_num(extra, nan=0.0)
            E_raw = np.hstack([E_raw, extra]).astype(np.float32)
            edge_cols = edge_cols + feat_cols
            log.info("[assembly] merged seq pair features (%d cols) from cache",
                     len(feat_cols))
        else:
            log.warning("[assembly] use_seq_pair_features=true but no cache "
                        "found; run experiments/06_seq_features.py first")

    log.info("[assembly] edge dim=%d (%s)", E_raw.shape[1], edge_cols)
    return CoveredDataset(df=cov, P_raw=P_raw, H_raw=H_raw, E_raw=E_raw,
                          phages=phages, hosts=hosts, phage_index=pidx_g,
                          host_index=hidx_g, edge_cols=edge_cols)
