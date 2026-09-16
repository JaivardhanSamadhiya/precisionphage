#!/usr/bin/env python3
"""Step 8: phage cocktail optimisation on the predicted susceptibility matrix.

Pipeline:
  1. Host-cluster-grouped outer-OOF GBM predictions for every covered pair.
     A fold-specific F1 threshold is selected using only group-aware inner-OOF
     predictions from that outer fold's training partition.
  2. Build predicted coverage A (pred >= operating threshold) and TRUE coverage
     T (observed label == 1) over tested pairs only.
  3. Optimise cocktails (greedy + exact ILP) and score them on TRUE coverage:
       - coverage-vs-size curve (greedy vs random vs truth-oracle),
       - minimum cocktail to cover all coverable hosts (greedy vs ILP vs oracle),
       - robustness: k-redundant cocktails (k = 1,2,3).

Run (from /tmp, then cd in):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/08_cocktail.py
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

from precisionphage.cocktail import (  # noqa: E402
    greedy_cover, ilp_max_cover, ilp_min_cover, true_coverage,
    true_coverage_curve,
)
from precisionphage.eval import nested_group_oof_decisions  # noqa: E402
from precisionphage.features.assembly import build_covered_dataset  # noqa: E402
from precisionphage.models import fit_predict_gbm  # noqa: E402
from precisionphage.splits import load_or_build_clusters  # noqa: E402
from precisionphage.utils import (  # noqa: E402
    ensure_dirs, get_logger, limit_threads, load_config, set_determinism,
)

log = get_logger("cocktail")


def _assign_host_clusters(cfg, data):
    return load_or_build_clusters(cfg, data)[1]


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    seed = cfg["seed"]
    set_determinism(seed)
    ensure_dirs(cfg)
    limit_threads(1)
    rd = cfg["paths"]["results_dir"]

    data = build_covered_dataset(cfg)
    cov = data.df.reset_index(drop=True)
    X = data.X_flat()
    y = cov["label"].to_numpy().astype(int)
    hc = _assign_host_clusters(cfg, data)
    groups = cov["host"].map(hc).to_numpy()

    log.info("Generating nested-threshold, host-cluster-grouped OOF predictions")
    oof, oof_decision, fold_thresholds = nested_group_oof_decisions(
        X, y, groups, fit_predict_gbm, seed)
    log.info("Fold-specific inner-OOF F1 thresholds: %s", fold_thresholds)
    np.savez(
        rd / "cocktail_oof_predictions.npz",
        probabilities=oof.astype(np.float32),
        decisions=oof_decision.astype(np.uint8),
        thresholds=np.asarray(fold_thresholds, dtype=np.float32),
        pair_keys=(cov["phage"].astype(str) + "\t" + cov["host"].astype(str)).to_numpy(
            dtype=str),
    )

    # build phage x host coverage matrices over TESTED pairs only
    phages = sorted(cov["phage"].unique())
    hosts = sorted(cov["host"].unique())
    pix = {p: i for i, p in enumerate(phages)}
    hix = {h: i for i, h in enumerate(hosts)}
    n_p, n_h = len(phages), len(hosts)
    A = np.zeros((n_p, n_h), dtype=bool)            # predicted coverage
    T = np.zeros((n_p, n_h), dtype=bool)            # true coverage
    for r in range(len(cov)):
        pi = pix[cov.at[r, "phage"]]
        hi = hix[cov.at[r, "host"]]
        if oof_decision[r]:
            A[pi, hi] = True
        if y[r] == 1:
            T[pi, hi] = True

    targets = np.where(T.sum(0) >= 1)[0]            # coverable target hosts
    predicted_targets = targets[A[:, targets].any(axis=0)]
    log.info("panel: %d phages x %d hosts; coverable targets=%d", n_p, n_h,
             len(targets))
    log.info("targets with >=1 predicted covering phage=%d", len(predicted_targets))

    # --- coverage-vs-size curves ---
    g_order = greedy_cover(A, targets, k=1)
    g_curve = true_coverage_curve(g_order, T, targets, k=1)
    oracle_order = greedy_cover(T, targets, k=1)
    o_curve = true_coverage_curve(oracle_order, T, targets, k=1)
    rng = np.random.default_rng(seed)
    rand_curves = []
    cand = np.where(A.any(1))[0]
    for _ in range(50):
        perm = rng.permutation(cand)
        rand_curves.append(true_coverage_curve(perm[:len(g_order) + 5], T, targets, 1))
    L = min(len(c) for c in rand_curves)
    rand_curve = np.mean([c[:L] for c in rand_curves], axis=0)

    def size_to_reach(curve, frac):
        idx = np.where(curve >= frac)[0]
        return int(idx[0] + 1) if len(idx) else None

    log.info("coverage-vs-size (TRUE coverage of %d targets):", len(targets))
    for frac in (0.5, 0.8, 0.9, 0.95):
        log.info("  reach %.0f%%: greedy=%s  oracle=%s  random(mean)=%s",
                 frac * 100, size_to_reach(g_curve, frac),
                 size_to_reach(o_curve, frac), size_to_reach(rand_curve, frac))

    # --- minimum full-cover cocktail (greedy vs ILP vs oracle) ---
    g_full = greedy_cover(A, targets, k=1)
    ilp_full = ilp_min_cover(A, targets, k=1)
    ilp_oracle = ilp_min_cover(T, targets, k=1)
    log.info("MIN COCKTAIL for full predicted-cover of coverable targets:")
    log.info("  greedy : size=%d  true-coverage=%.3f", len(g_full),
             true_coverage(g_full, T, targets, 1))
    if ilp_full is not None:
        log.info("  ILP    : size=%d  true-coverage=%.3f", len(ilp_full),
                 true_coverage(ilp_full, T, targets, 1))
    if ilp_oracle is not None:
        log.info("  ILP(oracle on truth): size=%d (theoretical minimum)",
                 len(ilp_oracle))

    # --- robustness: k-redundant cocktails ---
    rob_rows = []
    for k in (1, 2, 3):
        sel = greedy_cover(A, targets, k=k)
        tc1 = true_coverage(sel, T, targets, 1)
        tck = true_coverage(sel, T, targets, k)
        rob_rows.append({"k": k, "cocktail_size": len(sel),
                         "true_cover_>=1": round(tc1, 3),
                         "true_cover_>=k": round(tck, 3)})
    rob = pd.DataFrame(rob_rows)
    log.info("ROBUSTNESS (k-redundant greedy cocktails):\n%s",
             rob.to_string(index=False))

    # --- max-coverage under budget (exact ILP) ---
    budget_rows = []
    for B in (5, 10, 20, 30):
        sel, _ = ilp_max_cover(A, targets, budget=B, k=1)
        budget_rows.append({"budget": B,
                            "ilp_true_coverage": round(true_coverage(sel, T, targets, 1), 3),
                            "greedy_true_coverage": round(
                                float(g_curve[min(B, len(g_curve)) - 1]), 3)})
    budget = pd.DataFrame(budget_rows)
    log.info("MAX-COVERAGE under budget (ILP vs greedy, TRUE coverage):\n%s",
             budget.to_string(index=False))

    # --- persist ---
    curve_df = pd.DataFrame({
        "size": np.arange(1, len(g_curve) + 1),
        "greedy": g_curve,
        "oracle": np.pad(o_curve, (0, max(0, len(g_curve) - len(o_curve))),
                         constant_values=o_curve[-1] if len(o_curve) else 0)[:len(g_curve)],
        "random_mean": np.pad(rand_curve, (0, max(0, len(g_curve) - len(rand_curve))),
                              constant_values=rand_curve[-1] if len(rand_curve) else 0)[:len(g_curve)],
    })
    curve_df.to_csv(rd / "cocktail_curve.csv", index=False)
    rob.to_csv(rd / "cocktail_robustness.csv", index=False)
    budget.to_csv(rd / "cocktail_budget.csv", index=False)
    (rd / "cocktail_summary.json").write_text(json.dumps({
        "threshold_method": "fold-specific F1 on group-aware inner-OOF training predictions",
        "fold_thresholds": fold_thresholds,
        "threshold_median": float(np.median(fold_thresholds)),
        "n_phages": n_p, "n_hosts": n_h,
        "n_targets": int(len(targets)),
        "n_predicted_targets": int(len(predicted_targets)),
        "greedy_min_size": len(g_full),
        "ilp_min_size": int(len(ilp_full)) if ilp_full is not None else None,
        "ilp_oracle_min_size": int(len(ilp_oracle)) if ilp_oracle is not None else None,
        "robustness": rob_rows, "budget": budget_rows,
    }, indent=2, default=float))

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        s = curve_df["size"]
        ax.plot(s, curve_df["oracle"], label="Oracle (truth-greedy)", lw=2, color="#2ca02c")
        ax.plot(s, curve_df["greedy"], label="Model-driven greedy", lw=2, color="#1f77b4")
        ax.plot(s, curve_df["random_mean"], label="Random (mean of 50)", lw=2,
                ls="--", color="#888888")
        if ilp_full is not None:
            ax.axvline(len(ilp_full), color="#d62728", ls=":",
                       label=f"ILP min cocktail (n={len(ilp_full)})")
        ax.set_xlabel("Cocktail size (number of phages)")
        ax.set_ylabel("Fraction of target hosts truly covered")
        ax.set_title("Phage cocktail coverage vs size")
        ax.set_ylim(0, 1.02); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(rd / "cocktail_coverage.png", dpi=150)
        log.info("Wrote figure %s", rd / "cocktail_coverage.png")
    except Exception as e:
        log.warning("figure skipped (%s)", e)

    log.info("Wrote cocktail OOF predictions, curves, summaries, and figure")


if __name__ == "__main__":
    main()
