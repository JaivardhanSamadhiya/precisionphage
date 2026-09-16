#!/usr/bin/env python3
"""Regenerate the five primary result figures into data/results_v2/.

Uses saved result artifacts (no model retraining) except when optional inputs
exist (phist_pooled_preds.npz for ROC overlays). Safe to run after any pipeline
step that produced the CSV/JSON tables.

Outputs:
  figure_main.png
  fig_feature_importance.png
  fig_phist_compare.png
  cocktail_coverage.png
  temporal_dynamics.png
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RD = ROOT / "data" / "results_v2"

REGIME_LABEL = {
    "loso_species": "Unseen species\n(LOSO)",
    "host_cluster": "Unseen host\ncluster",
    "phage_cluster": "Unseen phage\ncluster",
    "combined_unseen": "Both unseen\n(cold start)",
}
REGIME_ORDER = ["loso_species", "host_cluster", "phage_cluster", "combined_unseen"]
RLAB = {k: v.replace("\n", " ") for k, v in REGIME_LABEL.items()}


def _parse_auc(s: str) -> float:
    return float(str(s).split()[0])


def figure_main() -> None:
    subprocess.run([sys.executable, str(ROOT / "experiments" / "10_report.py")],
                   check=True, cwd=str(ROOT))


def figure_feature_importance() -> None:
    fi = pd.read_csv(RD / "feature_importance.csv")
    fam_cols = {"Phage genome composition", "Host genome composition",
                "Composition distance", "Nucleotide homology",
                "CRISPR matching", "Protein homology"}
    fam = fi[fi.feature.isin(fam_cols)].sort_values("importance")
    edge = fi[~fi.feature.isin(fam_cols)].sort_values("importance")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].barh(fam.feature, fam.importance, color="#1f77b4")
    ax[0].set_xlabel("Total GBM gain importance (normalised)")
    ax[0].set_title("a  Importance by feature family")
    ax[0].grid(axis="x", alpha=0.3)

    ax[1].barh(edge.feature, edge.importance, color="#2ca02c")
    ax[1].set_xlabel("GBM gain importance (normalised)")
    ax[1].set_title("b  Pairwise edge features")
    ax[1].grid(axis="x", alpha=0.3)
    fig.suptitle("GBM feature importances (leakage-safe design matrix)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RD / "fig_feature_importance.png", dpi=300)
    plt.close(fig)
    print("wrote fig_feature_importance.png")


def figure_phist_compare() -> None:
    """Three-way external baseline figure (GBM vs PHIST vs RaFAH-style)."""
    tbl = pd.read_csv(RD / "table_external_baselines.csv")
    regime_map = {
        "Unseen species (LOSO)": "loso_species",
        "Unseen host cluster": "host_cluster",
        "Unseen phage (RaFAH's task)": "phage_cluster",
        "Both unseen (cold start)": "combined_unseen",
    }
    tbl["regime"] = tbl.Regime.map(regime_map)
    tbl = tbl.dropna(subset=["regime"]).set_index("regime").reindex(REGIME_ORDER)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    xx = np.arange(len(REGIME_ORDER))
    w = 0.25
    gb = [_parse_auc(tbl.loc[r, "Our AUROC"]) for r in REGIME_ORDER]
    ph = [_parse_auc(tbl.loc[r, "PHIST AUROC"]) for r in REGIME_ORDER]
    rf = [_parse_auc(tbl.loc[r, "RaFAH-style AUROC"]) for r in REGIME_ORDER]
    ax[0].bar(xx - w, gb, w, color="#1f77b4", label="PrecisionPhage (GBM)")
    ax[0].bar(xx, ph, w, color="#9467bd", label="PHIST (published)")
    ax[0].bar(xx + w, rf, w, color="#ff7f0e", label="RaFAH-style (in-house)")
    ax[0].axhline(0.5, color="k", ls="--", lw=1)
    ax[0].set_xticks(xx)
    ax[0].set_xticklabels([REGIME_LABEL[r] for r in REGIME_ORDER], fontsize=9)
    ax[0].set_ylabel("AUROC")
    ax[0].set_ylim(0.35, 1.02)
    ax[0].set_title("a  External baselines on identical test pairs")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[0].grid(axis="y", alpha=0.3)

    npz_path = RD / "phist_pooled_preds.npz"
    if npz_path.exists():
        from sklearn.metrics import auc as sk_auc
        from sklearn.metrics import roc_curve
        npz = np.load(npz_path)
        for r in REGIME_ORDER:
            y = npz[f"{r}__y"]
            fpr, tpr, _ = roc_curve(y, npz[f"{r}__GBM"])
            ax[1].plot(fpr, tpr, lw=1.8,
                       label=f"GBM {RLAB[r][:12]} ({sk_auc(fpr, tpr):.2f})")
        for r in REGIME_ORDER:
            y = npz[f"{r}__y"]
            fpr, tpr, _ = roc_curve(y, npz[f"{r}__PHIST"])
            ax[1].plot(fpr, tpr, lw=1.4, ls="--",
                       label=f"PHIST {RLAB[r][:12]} ({sk_auc(fpr, tpr):.2f})")
        ax[1].plot([0, 1], [0, 1], "k:", lw=1)
        ax[1].set_title("b  ROC: GBM (solid) vs PHIST (dashed)")
    else:
        # Bar summary of AUROC gaps vs PHIST when pooled preds unavailable
        diffs = [_parse_auc(tbl.loc[r, "Our AUROC"]) - _parse_auc(tbl.loc[r, "PHIST AUROC"])
                 for r in REGIME_ORDER]
        ax[1].bar(xx, diffs, color="#1f77b4", alpha=0.85)
        ax[1].axhline(0, color="k", ls="--", lw=1)
        ax[1].set_xticks(xx)
        ax[1].set_xticklabels([REGIME_LABEL[r] for r in REGIME_ORDER], fontsize=9)
        ax[1].set_ylabel("ΔAUROC (GBM − PHIST)")
        ax[1].set_title("b  GBM advantage over PHIST")

    ax[1].set_xlabel("Leakage regime")
    ax[1].grid(alpha=0.3)
    fig.suptitle("External baseline comparison (leakage-controlled evaluation)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(RD / "fig_phist_compare.png", dpi=300)
    plt.close(fig)
    print("wrote fig_phist_compare.png")


def figure_cocktail_coverage() -> None:
    curve = pd.read_csv(RD / "cocktail_curve.csv")
    summary = json.loads((RD / "cocktail_summary.json").read_text())
    ilp_n = summary.get("ilp_min_size")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["size"], curve["oracle"], label="Oracle (truth-greedy)", lw=2, color="#2ca02c")
    ax.plot(curve["size"], curve["greedy"], label="Model-driven greedy", lw=2, color="#1f77b4")
    ax.plot(curve["size"], curve["random_mean"], label="Random (mean of 50)", lw=2,
            ls="--", color="#888888")
    if ilp_n:
        ax.axvline(ilp_n, color="#d62728", ls=":",
                   label=f"ILP min cocktail (n={ilp_n})")
    ax.set_xlabel("Cocktail size (number of phages)")
    ax.set_ylabel("Fraction of target hosts truly covered")
    ax.set_title("Phage cocktail coverage vs size")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RD / "cocktail_coverage.png", dpi=300)
    plt.close(fig)
    print("wrote cocktail_coverage.png")


def figure_temporal_dynamics() -> None:
    traj_path = RD / "temporal_trajectory.npz"
    if not traj_path.exists():
        print("SKIP temporal_dynamics.png (run experiments/09_temporal.py first)")
        return
    traj = np.load(traj_path, allow_pickle=True)
    meta = json.loads((RD / "temporal_summary.json").read_text())
    sizes = {r["strategy"]: r["n_phages"] for r in meta["outcomes"]}
    colors = {"control": "#7f7f7f", "monophage": "#d62728",
              "cocktail_k1": "#1f77b4", "robust_k2": "#2ca02c"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name in sizes:
        d = traj[name].item()
        lab = f"{name} (n={sizes[name]})"
        ax[0].plot(d["t"], np.maximum(d["total"], 1), label=lab,
                   color=colors.get(name, "#333"), lw=2)
        ax[1].plot(d["t"], np.maximum(d["R"], 1), label=lab,
                   color=colors.get(name, "#333"), lw=2)
    for a, ttl, yl in ((ax[0], "Total bacterial load", "CFU/mL (total)"),
                       (ax[1], "Resistant subpopulation", "CFU/mL (resistant)")):
        a.set_yscale("log")
        a.set_xlabel("time (h)")
        a.set_ylabel(yl)
        a.set_title(ttl)
        a.grid(alpha=0.3)
        a.legend(loc="lower right", fontsize=8)
    fig.suptitle("Illustrative unfitted-parameter trajectories (not biological prediction)")
    fig.tight_layout()
    fig.savefig(RD / "temporal_dynamics.png", dpi=300)
    plt.close(fig)
    print("wrote temporal_dynamics.png")


def main() -> None:
    RD.mkdir(parents=True, exist_ok=True)
    figure_main()
    figure_feature_importance()
    figure_phist_compare()
    figure_cocktail_coverage()
    figure_temporal_dynamics()
    pngs = sorted(RD.glob("*.png"))
    print(f"figures in {RD}: {[p.name for p in pngs]}")


if __name__ == "__main__":
    main()
