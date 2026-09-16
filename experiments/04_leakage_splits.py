#!/usr/bin/env python3
"""Step 4: homology-aware splits + leakage audit.

Why: taxonomic leave-one-species/genus-out can still place sequence-similar
genomes in both train and test. Here we (1) cluster phage and host genomes by
sequence similarity (MinHash -> Mash distance -> single linkage), (2) QUANTIFY
how much leakage the taxonomic splits hide, (3) re-evaluate the GBM baseline
under strict cluster-holdout and the hardest combined unseen-phage+unseen-host
regime, and (4) ASSERT that no train entity shares a genome cluster with any
test entity in the strict regimes.

Run (start the shell in /tmp to avoid the repo-cwd prompt hang, then cd in):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/04_leakage_splits.py
"""
from __future__ import annotations

import os

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

from precisionphage.eval import bootstrap_ci, calibration_curve_ece  # noqa: E402
from precisionphage.features.assembly import build_covered_dataset  # noqa: E402
from precisionphage.models import fit_predict_gbm, run_grouped_cv  # noqa: E402
from precisionphage.splits import (  # noqa: E402
    audit_taxonomic_leakage, combined_unseen_folds, cross_study_folds,
    leave_one_group_out, load_or_build_clusters,
)
from precisionphage.utils import (  # noqa: E402
    ensure_dirs, get_logger, limit_threads, load_config, set_determinism,
)

log = get_logger("leakage")


def _assert_cluster_disjoint(cov, folds, cols):
    """Hard guarantee: for each fold, train and test share no cluster on `cols`."""
    for fold in folds:
        for col in cols:
            tr = set(cov.iloc[fold.train_idx][col].unique())
            te = set(cov.iloc[fold.test_idx][col].unique())
            overlap = tr & te
            if overlap:
                raise AssertionError(
                    f"LEAKAGE in fold {fold.name}: {col} overlap {len(overlap)} "
                    f"clusters between train and test")
    log.info("  [leak-check] OK: %d folds, no train/test overlap on %s",
             len(folds), cols)


def _summ(res, cfg, seed):
    aucs = list(res["fold_aucs"].values())
    clusters = [res["fold_clusters"][k] for k in res["fold_aucs"]]
    ci = bootstrap_ci(aucs, cluster_labels=clusters,
                      n_boot=cfg["eval"]["bootstrap_n"],
                      ci=cfg["eval"]["bootstrap_ci"], seed=seed)
    cal = calibration_curve_ece(res["pooled_y"], res["pooled_p"],
                                cfg["eval"]["calibration_bins"])
    return {**res["agg"], "ci_mean": ci, "ece": cal["ece"]}


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    seed = cfg["seed"]
    set_determinism(seed)
    ensure_dirs(cfg)
    limit_threads(1)

    data = build_covered_dataset(cfg)
    cov = data.df
    X = data.X_flat()

    # ---- genome-similarity clustering on BOTH axes ----
    sp = cfg["splits"]
    p_clusters, h_clusters = load_or_build_clusters(cfg, data)
    cov["phage_cluster"] = cov["phage"].map(p_clusters).astype(int)
    cov["host_cluster"] = cov["host"].map(h_clusters).astype(int)

    # ---- leakage audit: what taxonomic splits were hiding ----
    audit = {
        "host_species_vs_genome_cluster":
            audit_taxonomic_leakage(cov, "host_species", "host_cluster", "host"),
        "host_genus_vs_genome_cluster":
            audit_taxonomic_leakage(cov, "host_genus", "host_cluster", "host"),
    }
    log.info("LEAKAGE AUDIT (taxonomic splits vs genome clusters):")
    for name, a in audit.items():
        log.info("  %s: %d/%d clusters span >1 group; %d rows (%.1f%%) and "
                 "%d/%d hosts (%.1f%%) are homology-leaky under that split",
                 name, a["clusters_spanning_multiple_groups"], a["n_clusters"],
                 a["leaky_rows"], 100 * a["leaky_row_fraction"],
                 a["leaky_entities"], a["total_entities"],
                 100 * a["leaky_entity_fraction"])

    mpos, mneg = cfg["data"]["min_pos_per_group"], cfg["data"]["min_neg_per_group"]
    regimes = {
        # taxonomic (potentially leaky) — kept as the comparison point
        "loso_species": (list(leave_one_group_out(cov, "host_species", "loso", mpos, mneg)),
                         "host_genus", []),
        "logo_genus": (list(leave_one_group_out(cov, "host_genus", "logo", mpos, mneg)),
                       "host_genus", []),
        # homology-aware (leakage-controlled)
        "host_cluster": (list(leave_one_group_out(cov, "host_cluster", "host_cluster", mpos, mneg)),
                         "host_cluster", ["host_cluster"]),
        "phage_cluster": (list(leave_one_group_out(cov, "phage_cluster", "phage_cluster", mpos, mneg)),
                          "phage_cluster", ["phage_cluster"]),
        "combined_unseen": (list(combined_unseen_folds(cov, "phage_cluster", "host_cluster",
                                                       sp["n_combined_splits"], seed, mpos, mneg)),
                            "host_cluster", ["phage_cluster", "host_cluster"]),
    }

    results = {}
    for rname, (folds, cluster_col, check_cols) in regimes.items():
        log.info("[%s] %d evaluable folds", rname, len(folds))
        if not folds:
            continue
        if check_cols:
            _assert_cluster_disjoint(cov, folds, check_cols)
        res = run_grouped_cv(cov, X, folds, fit_predict_gbm, seed,
                             cluster_col=cluster_col, cfg=cfg)
        results[rname] = _summ(res, cfg, seed)
        r = results[rname]
        log.info("[%s/GBM] meanAUC=%.4f (CI %.4f-%.4f) pooledAUC=%.4f "
                 "pooledPR=%.4f folds=%d/%d ECE=%.3f", rname, r["mean_roc_auc"],
                 r["ci_mean"]["lo"], r["ci_mean"]["hi"], r["pooled_roc_auc"],
                 r["pooled_pr_auc"], r["n_folds_used"], r["n_folds_total"],
                 r["ece"])

    out = {"audit": audit, "results": results,
           "cluster_stats": {"n_phage_clusters": int(cov["phage_cluster"].nunique()),
                             "n_host_clusters": int(cov["host_cluster"].nunique()),
                             "n_phages": int(cov["phage"].nunique()),
                             "n_hosts": int(cov["host"].nunique())}}
    (cfg["paths"]["results_dir"] / "leakage_splits_results.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")

    rows = []
    for rname, r in results.items():
        rows.append({"regime": rname, "model": "GBM",
                     "mean_auc": round(r["mean_roc_auc"], 4),
                     "ci_lo": round(r["ci_mean"]["lo"], 4),
                     "ci_hi": round(r["ci_mean"]["hi"], 4),
                     "pooled_auc": round(r["pooled_roc_auc"], 4),
                     "pooled_pr": round(r["pooled_pr_auc"], 4),
                     "ece": round(r["ece"], 4),
                     "folds_used": r["n_folds_used"]})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(cfg["paths"]["results_dir"] / "leakage_splits_summary.csv", index=False)
    log.info("\n%s", tbl.to_string(index=False))
    log.info("Wrote leakage_splits_results.json / leakage_splits_summary.csv")


if __name__ == "__main__":
    main()
