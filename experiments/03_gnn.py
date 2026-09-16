#!/usr/bin/env python3
"""Step 3: inductive GraphSAGE + Edge-MLP vs GBM on the genome-covered subset.

Fair head-to-head: BOTH models receive identical genomic node features and
identical pairwise edge features, evaluated on identical leakage-safe folds
(LOSO / LOGO). The only difference is that the GNN additionally performs
message passing over the training-positive interaction graph. Any AUC gain is
therefore attributable to the relational (graph) signal, not to extra features.

Run:
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python experiments/03_gnn.py
"""
from __future__ import annotations

import os

# CRITICAL: cap BLAS/OpenMP thread pools BEFORE numpy/torch import. These pools
# fix their size at first import; capping later (inside workers) is too late and
# leads to n_jobs * n_cores oversubscription that thrashes the whole node. With
# this set, parallelism is exactly compute.n_jobs processes * 1 thread <= 10.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precisionphage.data import GenomeIndex, load_interactions  # noqa: E402
from precisionphage.eval import bootstrap_ci, calibration_curve_ece  # noqa: E402
from precisionphage.features.genomic import (  # noqa: E402
    build_node_features, edge_features_from_spectra, kmer_spectrum,
)
from precisionphage.models import fit_predict_gbm, run_gnn_cv, run_grouped_cv  # noqa: E402
from precisionphage.splits import leave_one_group_out  # noqa: E402
from precisionphage.utils import (  # noqa: E402
    ensure_dirs, get_logger, limit_threads, load_config, set_determinism,
)

log = get_logger("gnn_exp")

VHI_FEATS = ["k3dist", "k6dist", "GCdiff", "Homology"]


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    seed = cfg["seed"]
    set_determinism(seed)
    ensure_dirs(cfg)
    limit_threads(1)  # parent stays single-threaded; workers do the parallel work
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
    log.info("Genome-covered subset: %d pairs (pos=%d neg=%d) over %d phages, %d hosts",
             len(cov), int((cov["label"] == 1).sum()), int((cov["label"] == 0).sum()),
             cov["phage"].nunique(), cov["host"].nunique())
    log.info("By study:\n%s", cov["study"].value_counts().to_string())

    # node index assignment
    phages = sorted(cov["phage"].unique())
    hosts = sorted(cov["host"].unique())
    p2i = {p: i for i, p in enumerate(phages)}
    h2i = {h: i for i, h in enumerate(hosts)}
    cov["pidx"] = cov["phage"].map(p2i).astype(int)
    cov["hidx"] = cov["host"].map(h2i).astype(int)

    # node feature matrices
    log.info("Building node features ...")
    P_raw, pmask = build_node_features(phages, pidx_g, k=k,
                                       use_codon=cfg["features"]["use_codon"],
                                       use_dinuc=cfg["features"]["use_dinuc"],
                                       n_workers=cfg["features"]["n_workers"])
    H_raw, hmask = build_node_features(hosts, hidx_g, k=k,
                                       use_codon=cfg["features"]["use_codon"],
                                       use_dinuc=cfg["features"]["use_dinuc"],
                                       n_workers=cfg["features"]["n_workers"])
    log.info("Node features: phages %s (cov %d), hosts %s (cov %d)",
             P_raw.shape, int(pmask.sum()), H_raw.shape, int(hmask.sum()))

    # edge features: recomputed-from-spectra (study-invariant) + VHI precomputed.
    # The k-mer spectrum is the first block of the node vector, so reuse it
    # instead of re-reading every genome from disk.
    log.info("Computing edge features ...")
    kdim = int(kmer_spectrum("ACGT" * k, k).shape[0])
    P_spec = P_raw[:, :kdim]
    H_spec = H_raw[:, :kdim]
    pi = cov["pidx"].to_numpy()
    hi = cov["hidx"].to_numpy()
    recomputed = np.zeros((len(cov), 4), dtype=np.float32)
    for r in range(len(cov)):
        recomputed[r] = edge_features_from_spectra(P_spec[pi[r]], H_spec[hi[r]])
    vhi = [c for c in VHI_FEATS if c in cov.columns]
    E_vhi = cov[vhi].to_numpy(dtype=np.float32) if vhi else np.zeros((len(cov), 0))
    E_raw = np.hstack([recomputed, E_vhi]).astype(np.float32)
    log.info("Edge feature dim: %d (4 recomputed + %d VHI %s)",
             E_raw.shape[1], len(vhi), vhi)

    # GBM gets the SAME information as the GNN decoder: node feats + edge feats
    X_gbm = np.hstack([P_raw[cov["pidx"].to_numpy()],
                       H_raw[cov["hidx"].to_numpy()], E_raw]).astype(np.float32)

    regimes = {
        "loso": ("host_species", "loso"),
        "logo": ("host_genus", "logo"),
    }

    results = {}
    for rname, (gcol, reg) in regimes.items():
        folds = list(leave_one_group_out(cov, gcol, reg,
                                         cfg["data"]["min_pos_per_group"],
                                         cfg["data"]["min_neg_per_group"]))
        log.info("[%s] %d evaluable folds", rname, len(folds))
        if not folds:
            continue
        results[rname] = {}

        # ---- GBM baseline ----
        gbm = run_grouped_cv(cov, X_gbm, folds, fit_predict_gbm, seed,
                             cluster_col=gcol, cfg=cfg)
        results[rname]["GBM"] = _summ(gbm, cfg, seed)
        _log_row(rname, "GBM", results[rname]["GBM"])

        # ---- GNN ----
        gnn = run_gnn_cv(cov, P_raw, H_raw, E_raw, folds, cfg, seed,
                         cluster_col=gcol)
        results[rname]["GNN"] = _summ(gnn, cfg, seed)
        _log_row(rname, "GNN", results[rname]["GNN"])

    out = cfg["paths"]["results_dir"] / "gnn_results.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    log.info("Wrote %s", out)

    rows = []
    for rname, mres in results.items():
        for mname, r in mres.items():
            rows.append({"regime": rname, "model": mname,
                         "mean_auc": round(r["mean_roc_auc"], 4),
                         "ci_lo": round(r["ci_mean"]["lo"], 4),
                         "ci_hi": round(r["ci_mean"]["hi"], 4),
                         "pooled_auc": round(r["pooled_roc_auc"], 4),
                         "pooled_pr": round(r["pooled_pr_auc"], 4),
                         "ece": round(r["ece"], 4),
                         "folds_used": r["n_folds_used"]})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(cfg["paths"]["results_dir"] / "gnn_summary.csv", index=False)
    log.info("\n%s", tbl.to_string(index=False))


def _summ(res, cfg, seed):
    aucs = list(res["fold_aucs"].values())
    clusters = [res["fold_clusters"][k] for k in res["fold_aucs"]]
    ci = bootstrap_ci(aucs, cluster_labels=clusters,
                      n_boot=cfg["eval"]["bootstrap_n"],
                      ci=cfg["eval"]["bootstrap_ci"], seed=seed)
    cal = calibration_curve_ece(res["pooled_y"], res["pooled_p"],
                                cfg["eval"]["calibration_bins"])
    return {**res["agg"], "ci_mean": ci, "ece": cal["ece"]}


def _log_row(rname, mname, r):
    log.info("[%s/%s] meanAUC=%.4f (CI %.4f-%.4f) pooledAUC=%.4f pooledPR=%.4f "
             "folds=%d/%d ECE=%.3f", rname, mname, r["mean_roc_auc"],
             r["ci_mean"]["lo"], r["ci_mean"]["hi"], r["pooled_roc_auc"],
             r["pooled_pr_auc"], r["n_folds_used"], r["n_folds_total"], r["ece"])


if __name__ == "__main__":
    main()
