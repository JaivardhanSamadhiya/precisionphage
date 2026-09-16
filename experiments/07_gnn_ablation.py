#!/usr/bin/env python3
"""Step 7: does graph message-passing add value? (GNN ablation)

For each leakage-controlled regime we evaluate three models on identical folds:
  * GBM            - the feature-based baseline (reference),
  * GNN (graph)    - GraphSAGE message passing over train-positive edges,
  * GNN (no graph) - the SAME architecture/params with an empty edge set, so
                     SAGEConv keeps only its self-transform (a pure MLP on
                     node+edge features).

The decisive test is GNN(graph) vs GNN(no graph): a DeLong paired test on the
pooled, aligned predictions tells us whether message passing contributes signal
beyond the features. We also report GBM vs GNN(graph). All p-values are
Benjamini-Hochberg FDR-corrected. GNN predictions are isotonic-calibrated
(AUC-preserving), so ECE is comparable across models.

Run (from /tmp, then cd in):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/07_gnn_ablation.py
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
    benjamini_hochberg, calibration_curve_ece, delong_auc_ci, delong_test,
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

log = get_logger("gnn_ablation")


def _assign_clusters(cfg, data):
    return load_or_build_clusters(cfg, data)


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    seed = cfg["seed"]
    set_determinism(seed)
    ensure_dirs(cfg)
    limit_threads(1)

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

    preds = {}
    for rname, folds in regimes.items():
        if not folds:
            continue
        gbm = run_grouped_cv(cov, X, folds, fit_predict_gbm, seed,
                             cluster_col="host_cluster", cfg=cfg)
        g_on = run_gnn_cv(cov, data.P_raw, data.H_raw, data.E_raw, folds, cfg,
                          seed, cluster_col="host_cluster", use_graph=True)
        g_off = run_gnn_cv(cov, data.P_raw, data.H_raw, data.E_raw, folds, cfg,
                           seed, cluster_col="host_cluster", use_graph=False)
        y = gbm["pooled_y"].astype(int)
        if not (np.array_equal(y, g_on["pooled_y"].astype(int))
                and np.array_equal(y, g_off["pooled_y"].astype(int))):
            raise AssertionError(f"{rname}: pooled labels misaligned")
        preds[rname] = {"y": y, "GBM": gbm["pooled_p"],
                        "GNN_graph": g_on["pooled_p"],
                        "GNN_nograph": g_off["pooled_p"]}
        log.info("[%s] pooled n=%d (pos=%d) collected", rname, len(y), int(y.sum()))

    # per regime/model: AUC + CI + ECE
    rows = []
    for rname, d in preds.items():
        for m in ("GBM", "GNN_graph", "GNN_nograph"):
            ci = delong_auc_ci(d["y"], d[m], cfg["eval"]["bootstrap_ci"])
            ece = calibration_curve_ece(d["y"], d[m], cfg["eval"]["calibration_bins"])["ece"]
            rows.append({"regime": rname, "model": m, "auc": ci["auc"],
                         "auc_lo": ci["lo"], "auc_hi": ci["hi"], "ece": ece})

    # decisive comparisons: graph contribution + GBM vs GNN(graph)
    cmp_rows, graph_pvals, gbm_pvals = [], [], []
    for rname, d in preds.items():
        gc = delong_test(d["y"], d["GNN_graph"], d["GNN_nograph"])
        gb = delong_test(d["y"], d["GBM"], d["GNN_graph"])
        cmp_rows.append({"regime": rname,
                         "auc_graph": gc["auc1"], "auc_nograph": gc["auc2"],
                         "graph_gain": gc["auc_diff"], "graph_delong_p": gc["p"],
                         "auc_gbm": gb["auc1"], "gbm_minus_graph": gb["auc_diff"],
                         "gbm_delong_p": gb["p"]})
        graph_pvals.append(gc["p"])
        gbm_pvals.append(gb["p"])
    for row, q in zip(cmp_rows, benjamini_hochberg(graph_pvals)):
        row["graph_q_bh"] = float(q)
        row["graph_helps_fdr05"] = bool(q < 0.05 and row["graph_gain"] > 0)
    for row, q in zip(cmp_rows, benjamini_hochberg(gbm_pvals)):
        row["gbm_q_bh"] = float(q)

    auc_tbl = pd.DataFrame(rows)
    cmp_tbl = pd.DataFrame(cmp_rows)
    rd = cfg["paths"]["results_dir"]
    auc_tbl.to_csv(rd / "gnn_ablation_auc.csv", index=False)
    cmp_tbl.to_csv(rd / "gnn_ablation_cmp.csv", index=False)
    (rd / "gnn_ablation.json").write_text(
        json.dumps({"auc": rows, "comparison": cmp_rows}, indent=2, default=float))
    np.savez(rd / "gnn_ablation_preds.npz",
             **{f"{r}__{k}": v for r, d in preds.items() for k, v in d.items()})
    log.info("AUC + ECE by model:\n%s", auc_tbl.round(4).to_string(index=False))
    log.info("GRAPH CONTRIBUTION (GNN graph vs no-graph) + GBM vs GNN, "
             "BH-corrected:\n%s", cmp_tbl.round(4).to_string(index=False))
    log.info("Wrote gnn_ablation_*.csv + gnn_ablation.json")


if __name__ == "__main__":
    main()
