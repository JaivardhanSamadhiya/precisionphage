#!/usr/bin/env python3
"""Step 11: comprehensive supplementary figures for the manuscript.

Reads saved artifacts (pooled predictions, significance/cocktail tables) and the
covered dataset, and emits visualizations covering every result:
  * fig_roc.png            - ROC curves per leakage regime (GBM vs GNN)
  * fig_calibration.png    - reliability diagrams + ECE per regime
  * fig_significance.png   - forest plot of GBM-GNN AUC gap (DeLong CI + q)
  * fig_cocktail_detail.png- budget sweep + k-robustness
  * fig_feature_importance.png - grouped + per-edge GBM gain importances

Run (from /tmp, then cd in):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/11_figures.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import auc, roc_curve  # noqa: E402

from precisionphage.utils import get_logger, load_config  # noqa: E402

log = get_logger("figures")

REGIMES = ["loso_species", "host_cluster", "phage_cluster", "combined_unseen"]
RLAB = {"loso_species": "Unseen species (LOSO)",
        "host_cluster": "Unseen host cluster",
        "phage_cluster": "Unseen phage cluster",
        "combined_unseen": "Both unseen (cold start)"}


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return e


def fig_roc(npz, rd):
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for a, reg in zip(ax.ravel(), REGIMES):
        y = npz[f"{reg}__y"]
        for mdl, col in (("GBM", "#1f77b4"), ("GNN", "#ff7f0e")):
            p = npz[f"{reg}__{mdl}"]
            fpr, tpr, _ = roc_curve(y, p)
            a.plot(fpr, tpr, color=col, lw=2, label=f"{mdl} (AUC={auc(fpr, tpr):.3f})")
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.set_title(RLAB[reg]); a.set_xlabel("False positive rate")
        a.set_ylabel("True positive rate"); a.legend(loc="lower right")
        a.grid(alpha=0.3)
    fig.suptitle("ROC curves across leakage-controlled regimes", fontsize=14)
    fig.tight_layout(); fig.savefig(rd / "fig_roc.png", dpi=180); plt.close(fig)
    log.info("wrote fig_roc.png")


def fig_calibration(npz, rd):
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for a, reg in zip(ax.ravel(), REGIMES):
        y = npz[f"{reg}__y"]
        a.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for mdl, col in (("GBM", "#1f77b4"), ("GNN", "#ff7f0e")):
            p = npz[f"{reg}__{mdl}"]
            edges = np.linspace(0, 1, 11)
            xs, ys = [], []
            for i in range(10):
                m = (p >= edges[i]) & (p <= edges[i + 1] if i == 9 else p < edges[i + 1])
                if m.sum() >= 5:
                    xs.append(p[m].mean()); ys.append(y[m].mean())
            a.plot(xs, ys, "o-", color=col, lw=2,
                   label=f"{mdl} (ECE={_ece(y, p):.3f})")
        a.set_title(RLAB[reg]); a.set_xlabel("Mean predicted probability")
        a.set_ylabel("Observed frequency"); a.legend(loc="upper left")
        a.grid(alpha=0.3); a.set_xlim(0, 1); a.set_ylim(0, 1)
    fig.suptitle("Calibration (reliability) diagrams", fontsize=14)
    fig.tight_layout(); fig.savefig(rd / "fig_calibration.png", dpi=180); plt.close(fig)
    log.info("wrote fig_calibration.png")


def fig_significance(cmp, rd):
    cmp = cmp.set_index("regime").reindex(REGIMES)
    y = np.arange(len(REGIMES))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(cmp.auc_diff, y,
                xerr=[cmp.auc_diff - cmp.diff_lo, cmp.diff_hi - cmp.auc_diff],
                fmt="o", color="#1f77b4", capsize=5, ms=9, lw=2)
    ax.axvline(0, color="k", ls="--", lw=1)
    for yi, reg in zip(y, REGIMES):
        q = cmp.loc[reg, "delong_q_bh"]
        ax.text(cmp.loc[reg, "diff_hi"] + 0.005, yi,
                f"q={q:.1e}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([RLAB[r] for r in REGIMES])
    ax.set_xlabel("AUROC gap (GBM \u2212 GNN), 95% DeLong CI")
    ax.set_title("Row-pooled AUROC difference: GBM minus GNN")
    ax.set_xlim(-0.02, 0.40); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(rd / "fig_significance.png", dpi=180); plt.close(fig)
    log.info("wrote fig_significance.png")


def fig_cocktail(budget, rob, rd):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(budget)); w = 0.38
    ax[0].bar(x - w / 2, budget.ilp_true_coverage, w, color="#d62728", label="ILP (exact)")
    ax[0].bar(x + w / 2, budget.greedy_true_coverage, w, color="#1f77b4", label="greedy")
    ax[0].set_xticks(x); ax[0].set_xticklabels(budget.budget)
    ax[0].set_xlabel("Budget (max phages)"); ax[0].set_ylabel("True host coverage")
    ax[0].set_title("a  Max-coverage under budget"); ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)

    xr = np.arange(len(rob))
    ax[1].bar(xr, rob.cocktail_size, color="#7f7f7f", alpha=0.6, label="cocktail size")
    ax[1].set_xlabel("redundancy k"); ax[1].set_xticks(xr); ax[1].set_xticklabels(rob.k)
    ax[1].set_ylabel("cocktail size (phages)")
    a2 = ax[1].twinx()
    a2.plot(xr, rob["true_cover_>=1"], "o-", color="#2ca02c", label="true cover \u22651")
    a2.plot(xr, rob["true_cover_>=k"], "s--", color="#d62728", label="true cover \u2265k")
    a2.set_ylabel("coverage fraction"); a2.set_ylim(0, 1.02)
    ax[1].set_title("b  Robustness vs cocktail size")
    lines = ax[1].get_legend_handles_labels()[0] + a2.get_legend_handles_labels()[0]
    labs = ax[1].get_legend_handles_labels()[1] + a2.get_legend_handles_labels()[1]
    ax[1].legend(lines, labs, loc="center right")
    fig.suptitle("Cocktail optimisation detail", fontsize=14)
    fig.tight_layout(); fig.savefig(rd / "fig_cocktail_detail.png", dpi=180); plt.close(fig)
    log.info("wrote fig_cocktail_detail.png")


def fig_importance(cfg, rd):
    from precisionphage.features.assembly import build_covered_dataset
    data = build_covered_dataset(cfg)
    X = np.nan_to_num(data.X_flat())
    y = data.df["label"].to_numpy().astype(int)
    D = data.P_raw.shape[1]
    edge_cols = data.edge_cols
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            importance_type="gain", n_jobs=4, eval_metric="logloss",
                            verbosity=0, random_state=cfg["seed"])
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=400, max_depth=5,
                                             learning_rate=0.05, random_state=cfg["seed"])
    clf.fit(X, y)
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        log.warning("no feature_importances_; skipping importance figure")
        return
    imp = np.asarray(imp, dtype=float)
    imp = imp / imp.sum()
    phage_imp = imp[:D].sum()
    host_imp = imp[D:2 * D].sum()
    edge_imp = imp[2 * D:]

    def fam(name):
        if name in ("cos_dist", "l1", "pearson", "jaccard", "k3dist", "k6dist", "GCdiff"):
            return "Composition distance"
        if name == "Homology" or name.startswith("hom"):
            return "Nucleotide homology"
        if name.startswith("crispr"):
            return "CRISPR matching"
        if name.startswith("prot"):
            return "Protein homology"
        return "other"
    fam_tot = {"Phage genome composition": phage_imp,
               "Host genome composition": host_imp,
               "Composition distance": 0.0, "Nucleotide homology": 0.0,
               "CRISPR matching": 0.0, "Protein homology": 0.0}
    for nm, v in zip(edge_cols, edge_imp):
        fam_tot[fam(nm)] += v

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    fams = sorted(fam_tot, key=fam_tot.get)
    ax[0].barh(fams, [fam_tot[f] for f in fams], color="#1f77b4")
    ax[0].set_xlabel("Total GBM gain importance (normalised)")
    ax[0].set_title("a  Importance by feature family")
    ax[0].grid(axis="x", alpha=0.3)

    order = np.argsort(edge_imp)
    ax[1].barh(np.array(edge_cols)[order], edge_imp[order], color="#2ca02c")
    ax[1].set_xlabel("GBM gain importance (normalised)")
    ax[1].set_title("b  Individual edge (pairwise) features")
    ax[1].grid(axis="x", alpha=0.3)
    fig.suptitle("What the model uses", fontsize=14)
    fig.tight_layout(); fig.savefig(rd / "fig_feature_importance.png", dpi=180)
    plt.close(fig)
    pd.DataFrame({"feature": list(fam_tot) + list(edge_cols),
                  "importance": list(fam_tot.values()) + list(edge_imp)}
                 ).to_csv(rd / "feature_importance.csv", index=False)
    log.info("wrote fig_feature_importance.png + feature_importance.csv "
             "(phage=%.3f host=%.3f edge=%.3f)", phage_imp, host_imp, edge_imp.sum())


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    rd = cfg["paths"]["results_dir"]
    npz = np.load(rd / "significance_pooled_preds.npz", allow_pickle=True)
    cmp = pd.read_csv(rd / "significance_modelcmp.csv")
    budget = pd.read_csv(rd / "cocktail_budget.csv")
    rob = pd.read_csv(rd / "cocktail_robustness.csv")
    fig_roc(npz, rd)
    fig_calibration(npz, rd)
    fig_significance(cmp, rd)
    fig_cocktail(budget, rob, rd)
    fig_importance(cfg, rd)
    log.info("all supplementary figures written to %s", rd)


if __name__ == "__main__":
    main()
