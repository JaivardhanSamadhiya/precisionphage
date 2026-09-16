#!/usr/bin/env python3
"""Step 5: statistical significance on the leakage-controlled regimes.

For each regime (the homology-aware splits from step 4, plus the leaky
loso_species as a contrast) and each model (GBM, GNN) we report:
  * AUC with a DeLong analytic 95% CI,
  * a label-permutation test that AUC > chance,
and per regime we compare GBM vs GNN with:
  * DeLong's paired test (correlated AUCs, identical samples),
  * McNemar's paired test (error profiles at the 0.5 threshold),
  * a paired bootstrap CI for the AUC difference.
All permutation p-values (skill-vs-chance) and all model-comparison p-values are
Benjamini-Hochberg FDR-corrected within their families. These procedures operate
on rows and are exploratory because pairs share phage and host entities.

Run (from /tmp, then cd in, to avoid the repo-cwd prompt hang):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/05_significance.py
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

from precisionphage.eval import (  # noqa: E402
    benjamini_hochberg, bootstrap_auc_diff, calibration_curve_ece, delong_auc_ci,
    delong_test, mcnemar_test, permutation_auc_test,
)
from precisionphage.features.assembly import build_covered_dataset  # noqa: E402
from precisionphage.models import (  # noqa: E402
    fit_predict_gbm, run_gnn_cv, run_grouped_cv,
)
from precisionphage.splits import (  # noqa: E402
    combined_unseen_folds, leave_one_group_out, load_or_build_clusters,
)
from precisionphage.utils import (  # noqa: E402
    ensure_dirs, get_logger, limit_threads, load_config, set_determinism,
)

log = get_logger("significance")


def _assign_clusters(cfg, data):
    return load_or_build_clusters(cfg, data)


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    seed = cfg["seed"]
    set_determinism(seed)
    ensure_dirs(cfg)
    limit_threads(1)
    n_perm = cfg["eval"]["permutation_n"]

    data = build_covered_dataset(cfg)
    cov = data.df
    X = data.X_flat()
    pc, hc = _assign_clusters(cfg, data)
    cov["phage_cluster"] = cov["phage"].map(pc).astype(int)
    cov["host_cluster"] = cov["host"].map(hc).astype(int)

    mpos, mneg = cfg["data"]["min_pos_per_group"], cfg["data"]["min_neg_per_group"]
    sp = cfg["splits"]
    regimes = {
        "loso_species": list(leave_one_group_out(cov, "host_species", "loso", mpos, mneg)),
        "host_cluster": list(leave_one_group_out(cov, "host_cluster", "host_cluster", mpos, mneg)),
        "phage_cluster": list(leave_one_group_out(cov, "phage_cluster", "phage_cluster", mpos, mneg)),
        "combined_unseen": list(combined_unseen_folds(cov, "phage_cluster", "host_cluster",
                                                      sp["n_combined_splits"], seed, mpos, mneg)),
    }

    # NOTE: nested grid tuning + class weighting were evaluated (see
    # fit_predict_gbm_tuned) and did NOT improve AUC over this robust default
    # (they added selection variance and hurt cold-start), so the default GBM is
    # used for the headline. Tuning remains available as an ablation.

    # --- collect pooled predictions per regime/model (aligned samples) ---
    preds = {}
    for rname, folds in regimes.items():
        if not folds:
            continue
        gbm = run_grouped_cv(cov, X, folds, fit_predict_gbm, seed,
                             cluster_col="host_cluster", cfg=cfg)
        gnn = run_gnn_cv(cov, data.P_raw, data.H_raw, data.E_raw, folds, cfg, seed,
                         cluster_col="host_cluster")
        if not np.array_equal(gbm["pooled_y"].astype(int), gnn["pooled_y"].astype(int)):
            raise AssertionError(f"{rname}: GBM/GNN pooled labels misaligned")
        preds[rname] = {"y": gbm["pooled_y"].astype(int),
                        "GBM": gbm["pooled_p"], "GNN": gnn["pooled_p"]}
        log.info("[%s] pooled n=%d (pos=%d) collected", rname,
                 len(gbm["pooled_y"]), int(gbm["pooled_y"].sum()))

    # --- per regime/model: AUC CI + permutation vs chance ---
    skill_rows, skill_pvals, skill_keys = [], [], []
    for rname, d in preds.items():
        for m in ("GBM", "GNN"):
            ci = delong_auc_ci(d["y"], d[m], cfg["eval"]["bootstrap_ci"])
            perm = permutation_auc_test(d["y"], d[m], n_perm, seed)
            ece = calibration_curve_ece(d["y"], d[m], cfg["eval"]["calibration_bins"])["ece"]
            skill_rows.append({"regime": rname, "model": m, "auc": ci["auc"],
                               "auc_lo": ci["lo"], "auc_hi": ci["hi"],
                               "ece": ece, "perm_p": perm["p"]})
            skill_pvals.append(perm["p"])
            skill_keys.append((rname, m))
    skill_q = benjamini_hochberg(skill_pvals)
    for row, q in zip(skill_rows, skill_q):
        row["perm_q_bh"] = float(q)
        row["above_chance_fdr05"] = bool(q < 0.05)

    # --- per regime: GBM vs GNN comparison ---
    cmp_rows, cmp_pvals = [], []
    for rname, d in preds.items():
        dt = delong_test(d["y"], d["GBM"], d["GNN"])
        mc = mcnemar_test(d["y"], (d["GBM"] >= 0.5).astype(int),
                          (d["GNN"] >= 0.5).astype(int))
        bd = bootstrap_auc_diff(d["y"], d["GBM"], d["GNN"],
                                cfg["eval"]["bootstrap_n"], cfg["eval"]["bootstrap_ci"], seed)
        cmp_rows.append({"regime": rname, "auc_gbm": dt["auc1"], "auc_gnn": dt["auc2"],
                         "auc_diff": dt["auc_diff"], "diff_lo": bd["lo"],
                         "diff_hi": bd["hi"], "delong_z": dt["z"], "delong_p": dt["p"],
                         "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "mcnemar_p": mc["p"]})
        cmp_pvals.append(dt["p"])
    cmp_q = benjamini_hochberg(cmp_pvals)
    for row, q in zip(cmp_rows, cmp_q):
        row["delong_q_bh"] = float(q)
        row["gbm_beats_gnn_fdr05"] = bool(q < 0.05 and row["auc_diff"] > 0)

    report = {
        "inference_note": (
            "Exploratory independent-row inference; observations share phage "
            "and host entities, so p-values and intervals are not confirmatory."
        ),
        "skill_vs_chance": skill_rows,
        "model_comparison": cmp_rows,
    }
    (cfg["paths"]["results_dir"] / "significance_results.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    np.savez(cfg["paths"]["results_dir"] / "significance_pooled_preds.npz",
             **{f"{r}__{k}": v for r, d in preds.items() for k, v in d.items()})

    skill_tbl = pd.DataFrame(skill_rows)
    cmp_tbl = pd.DataFrame(cmp_rows)
    skill_tbl.to_csv(cfg["paths"]["results_dir"] / "significance_skill.csv", index=False)
    cmp_tbl.to_csv(cfg["paths"]["results_dir"] / "significance_modelcmp.csv", index=False)
    log.info("SKILL vs CHANCE (AUC CI + permutation, BH-corrected):\n%s",
             skill_tbl.round(4).to_string(index=False))
    log.info("MODEL COMPARISON GBM vs GNN (DeLong/McNemar, BH-corrected):\n%s",
             cmp_tbl.round(4).to_string(index=False))
    log.info("Wrote significance_results.json + significance_*.csv")


if __name__ == "__main__":
    main()
