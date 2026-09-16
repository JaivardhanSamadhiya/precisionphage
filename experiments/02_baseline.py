#!/usr/bin/env python3
"""Step 2: leakage-safe features-only baseline on the full real dataset.

Uses the genome-derived edge features (k3dist, k6dist, GCdiff, Homology) which
are defined for every pair, under three generalization regimes:
  - LOSO   (leave-one-host-species-out)
  - LOGO   (leave-one-host-genus-out)
  - cross-study (train on 2 studies, test on the 3rd; domain shift)

This validates the evaluation harness and gives the first honest numbers. The
GNN must beat this baseline to justify its complexity.

Run:
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python experiments/02_baseline.py
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

from precisionphage.data import load_interactions  # noqa: E402
from precisionphage.eval import bootstrap_ci, calibration_curve_ece  # noqa: E402
from precisionphage.models import fit_predict_gbm, fit_predict_mlp, run_grouped_cv  # noqa: E402
from precisionphage.splits import cross_study_folds, leave_one_group_out  # noqa: E402
from precisionphage.utils import ensure_dirs, get_logger, load_config, set_determinism  # noqa: E402

log = get_logger("baseline")

EDGE_FEATS = ["k3dist", "k6dist", "GCdiff", "Homology"]


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    seed = cfg["seed"]
    set_determinism(seed)
    ensure_dirs(cfg)

    ds = load_interactions(cfg)
    df = ds.df.reset_index(drop=True)
    feats = [c for c in EDGE_FEATS if c in df.columns]
    log.info("Edge features used: %s", feats)
    X = df[feats].to_numpy(dtype=np.float32)

    regimes = {
        "loso": lambda: leave_one_group_out(df, "host_species", "loso",
                                            cfg["data"]["min_pos_per_group"],
                                            cfg["data"]["min_neg_per_group"]),
        "logo": lambda: leave_one_group_out(df, "host_genus", "logo",
                                            cfg["data"]["min_pos_per_group"],
                                            cfg["data"]["min_neg_per_group"]),
        "cross_study": lambda: cross_study_folds(df, "study",
                                                 cfg["data"]["min_pos_per_group"],
                                                 cfg["data"]["min_neg_per_group"]),
    }
    # GBM is the primary baseline (fast, strong). EdgeMLP-on-edge-features is a
    # secondary check and is run only on the cheaper regimes; the production
    # Edge-MLP is the GNN decoder (experiment 03), not this flat baseline.
    models = {
        "GBM": (fit_predict_gbm, {"loso", "logo", "cross_study"}),
        "EdgeMLP": (fit_predict_mlp, {"cross_study"}),
    }

    results = {}
    for rname, fold_fn in regimes.items():
        folds = list(fold_fn())
        log.info("[%s] %d evaluable folds", rname, len(folds))
        results[rname] = {}
        for mname, (mfn, regimes_for) in models.items():
            if rname not in regimes_for:
                continue
            res = run_grouped_cv(df, X, folds, mfn, seed)
            aucs = list(res["fold_aucs"].values())
            clusters = [res["fold_clusters"][k] for k in res["fold_aucs"]]
            ci = bootstrap_ci(aucs, cluster_labels=clusters,
                              n_boot=cfg["eval"]["bootstrap_n"],
                              ci=cfg["eval"]["bootstrap_ci"], seed=seed)
            cal = calibration_curve_ece(res["pooled_y"], res["pooled_p"],
                                        cfg["eval"]["calibration_bins"])
            agg = res["agg"]
            results[rname][mname] = {**agg, "ci_mean": ci, "ece": cal["ece"]}
            log.info("[%s/%s] meanAUC=%.4f (CI %.4f-%.4f) pooledAUC=%.4f "
                     "pooledPR=%.4f folds=%d/%d skipped=%d ECE=%.3f",
                     rname, mname, agg["mean_roc_auc"], ci["lo"], ci["hi"],
                     agg["pooled_roc_auc"], agg["pooled_pr_auc"],
                     agg["n_folds_used"], agg["n_folds_total"],
                     agg["n_folds_skipped"], cal["ece"])

    out = cfg["paths"]["results_dir"] / "baseline_results.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    log.info("Wrote %s", out)

    # flat summary table
    rows = []
    for rname, models_res in results.items():
        for mname, r in models_res.items():
            rows.append({"regime": rname, "model": mname,
                         "mean_auc": round(r["mean_roc_auc"], 4),
                         "ci_lo": round(r["ci_mean"]["lo"], 4),
                         "ci_hi": round(r["ci_mean"]["hi"], 4),
                         "pooled_auc": round(r["pooled_roc_auc"], 4),
                         "pooled_pr": round(r["pooled_pr_auc"], 4),
                         "ece": round(r["ece"], 4),
                         "folds_used": r["n_folds_used"]})
    pd.DataFrame(rows).to_csv(cfg["paths"]["results_dir"] / "baseline_summary.csv",
                              index=False)
    log.info("\n%s", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
