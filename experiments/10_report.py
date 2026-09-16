#!/usr/bin/env python3
"""Step 10: assemble manuscript-ready figures and reporting tables.

Reads the result artifacts produced by steps 04-09 (no model recomputation) and
emits:
  * figure_main.png      - 3-panel: generalisation, leakage hierarchy, GNN ablation
  * table1_main_results.csv / .tex   - headline GBM vs GNN per leakage regime
  * table2_leakage_hierarchy.csv     - AUC as leakage is progressively removed
  * table3_cocktail.csv              - cocktail sizes + k-robustness
  * table4_temporal.csv              - therapy outcomes
  * REPORT.md            - consolidated manuscript-ready summary

Run (from /tmp, then cd in):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/10_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from precisionphage.utils import get_logger, load_config  # noqa: E402

log = get_logger("report")

REGIME_LABEL = {
    "loso_species": "Unseen species\n(LOSO)",
    "logo_genus": "Unseen genus\n(LOGO)",
    "host_cluster": "Unseen host\ncluster",
    "phage_cluster": "Unseen phage\ncluster",
    "combined_unseen": "Both unseen\n(cold start)",
}
REGIME_ORDER = ["loso_species", "host_cluster", "phage_cluster", "combined_unseen"]
LEAK_ORDER = ["loso_species", "logo_genus", "host_cluster", "phage_cluster",
              "combined_unseen"]


def _ci(lo, hi):
    return f"({lo:.3f}\u2013{hi:.3f})"


def _md(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (avoids the tabulate dep)."""
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    rd = cfg["paths"]["results_dir"]

    skill = pd.read_csv(rd / "significance_skill.csv")
    cmp = pd.read_csv(rd / "significance_modelcmp.csv")
    leak = pd.read_csv(rd / "leakage_splits_summary.csv")
    abl = pd.read_csv(rd / "gnn_ablation_auc.csv")
    rob = pd.read_csv(rd / "cocktail_robustness.csv")
    ck = json.loads((rd / "cocktail_summary.json").read_text())
    temp = pd.read_csv(rd / "temporal_outcomes.csv")

    gbm = skill[skill.model == "GBM"].set_index("regime")
    gnn = skill[skill.model == "GNN"].set_index("regime")
    cmpi = cmp.set_index("regime")

    # ---------------- Table 1: main results ----------------
    rows = []
    for r in REGIME_ORDER:
        if r not in gbm.index:
            continue
        g, n, c = gbm.loc[r], gnn.loc[r], cmpi.loc[r]
        rows.append({
            "Regime": REGIME_LABEL[r].replace("\n", " "),
            "GBM AUC": f"{g.auc:.3f} {_ci(g.auc_lo, g.auc_hi)}",
            "GBM ECE": f"{g.ece:.3f}",
            "GBM > chance (q)": f"{g.perm_q_bh:.1e}",
            "GNN AUC": f"{n.auc:.3f} {_ci(n.auc_lo, n.auc_hi)}",
            "GBM\u2212GNN \u0394AUC": f"{c.auc_diff:+.3f} {_ci(c.diff_lo, c.diff_hi)}",
            "DeLong q": f"{c.delong_q_bh:.1e}",
        })
    t1 = pd.DataFrame(rows)
    t1.to_csv(rd / "table1_main_results.csv", index=False)
    (rd / "table1_main_results.tex").write_text(
        t1.to_latex(index=False, escape=True,
                    caption="Phage\u2013host interaction prediction across "
                            "leakage-controlled cross-validation regimes.",
                    label="tab:main"),
        encoding="utf-8")

    # ---------------- Table 2: leakage hierarchy ----------------
    lg = leak[leak.model == "GBM"].set_index("regime").reindex(LEAK_ORDER).dropna(how="all")
    t2 = lg.reset_index()[["regime", "mean_auc", "ci_lo", "ci_hi", "pooled_auc",
                           "ece", "folds_used"]]
    t2.to_csv(rd / "table2_leakage_hierarchy.csv", index=False)

    # ---------------- Table 3 & 4 ----------------
    rob.to_csv(rd / "table3_cocktail.csv", index=False)
    temp_disp = temp.copy()
    for c in ("end_load", "nadir"):
        temp_disp[c] = temp_disp[c].map(lambda v: f"{v:.2e}")
    temp_disp.to_csv(rd / "table4_temporal.csv", index=False)

    # ---------------- Figure: 3 panels ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))

    # Panel A: GBM vs GNN AUC by regime with CI
    regs = [r for r in REGIME_ORDER if r in gbm.index]
    x = np.arange(len(regs)); w = 0.38
    g_auc = [gbm.loc[r].auc for r in regs]
    g_err = [[gbm.loc[r].auc - gbm.loc[r].auc_lo for r in regs],
             [gbm.loc[r].auc_hi - gbm.loc[r].auc for r in regs]]
    n_auc = [gnn.loc[r].auc for r in regs]
    n_err = [[gnn.loc[r].auc - gnn.loc[r].auc_lo for r in regs],
             [gnn.loc[r].auc_hi - gnn.loc[r].auc for r in regs]]
    ax[0].bar(x - w / 2, g_auc, w, yerr=g_err, capsize=4, color="#1f77b4",
              label="GBM (features)")
    ax[0].bar(x + w / 2, n_auc, w, yerr=n_err, capsize=4, color="#ff7f0e",
              label="GNN")
    ax[0].axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax[0].set_xticks(x); ax[0].set_xticklabels([REGIME_LABEL[r] for r in regs], fontsize=9)
    ax[0].set_ylabel("AUROC"); ax[0].set_ylim(0.4, 1.0)
    ax[0].set_title("a  Generalisation under leakage control")
    ax[0].legend(loc="lower left", fontsize=9); ax[0].grid(axis="y", alpha=0.3)

    # Panel B: leakage hierarchy (pooled AUC drop)
    lh = leak[leak.model == "GBM"].set_index("regime").reindex(LEAK_ORDER).dropna(how="all")
    xb = np.arange(len(lh))
    ax[1].plot(xb, lh.pooled_auc.values, "o-", color="#1f77b4", lw=2, ms=8)
    for xi, (mu, lo, hi) in enumerate(zip(lh.mean_auc, lh.ci_lo, lh.ci_hi)):
        ax[1].errorbar(xi, mu, yerr=[[mu - lo], [hi - mu]], fmt="s",
                       color="#888", capsize=4, alpha=0.7)
    ax[1].axhline(0.5, color="k", ls="--", lw=1)
    ax[1].set_xticks(xb)
    ax[1].set_xticklabels([REGIME_LABEL[r] for r in lh.index], fontsize=9)
    ax[1].set_ylabel("AUROC"); ax[1].set_ylim(0.4, 1.0)
    ax[1].set_title("b  Performance vs leakage removed")
    ax[1].grid(axis="y", alpha=0.3)
    ax[1].plot([], [], "o-", color="#1f77b4", label="pooled AUC")
    ax[1].plot([], [], "s", color="#888", label="mean fold AUC \u00b195% CI")
    ax[1].legend(loc="lower left", fontsize=9)

    # Panel C: GNN ablation
    ag = abl[abl.model == "GNN_graph"].set_index("regime")
    an = abl[abl.model == "GNN_nograph"].set_index("regime")
    ab = abl[abl.model == "GBM"].set_index("regime")
    regs2 = [r for r in REGIME_ORDER if r in ag.index]
    xc = np.arange(len(regs2)); w2 = 0.27
    ax[2].bar(xc - w2, [ab.loc[r].auc for r in regs2], w2, color="#1f77b4",
              label="GBM")
    ax[2].bar(xc, [ag.loc[r].auc for r in regs2], w2, color="#2ca02c",
              label="GNN + graph")
    ax[2].bar(xc + w2, [an.loc[r].auc for r in regs2], w2, color="#9edae5",
              label="GNN, no graph (MLP)")
    ax[2].axhline(0.5, color="k", ls="--", lw=1)
    ax[2].set_xticks(xc); ax[2].set_xticklabels([REGIME_LABEL[r] for r in regs2], fontsize=9)
    ax[2].set_ylabel("AUROC"); ax[2].set_ylim(0.4, 1.0)
    ax[2].set_title("c  Graph message-passing ablation")
    ax[2].legend(loc="lower left", fontsize=9); ax[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(rd / "figure_main.png", dpi=300)
    log.info("Wrote %s", rd / "figure_main.png")

    # ---------------- REPORT.md ----------------
    cu = gbm.loc["combined_unseen"]
    ls = gbm.loc["loso_species"]
    md = []
    md.append("# PrecisionPhage: results summary\n")
    md.append("Leakage-controlled phage\u2013host interaction prediction, cocktail "
              "design, and eco-evolutionary therapy simulation. All numbers below "
              "are read directly from the pipeline's result artifacts.\n")
    md.append("## 1. Headline performance\n")
    md.append(f"- Easiest realistic regime (unseen species, LOSO): **AUROC "
              f"{ls.auc:.3f}** {_ci(ls.auc_lo, ls.auc_hi)}, ECE {ls.ece:.3f}.")
    md.append(f"- Hardest regime (both phage and host clusters unseen \u2014 true "
              f"cold start): **AUROC {cu.auc:.3f}** {_ci(cu.auc_lo, cu.auc_hi)}.")
    md.append("- The feature-based GBM has higher row-pooled AUROC than the GNN "
              "in every saved regime. Independent-row tests are exploratory "
              "because pairs share phage and host entities.\n")
    md.append("### Table 1. Main results (GBM vs GNN per leakage regime)\n")
    md.append(_md(t1))
    md.append("\n## 2. Leakage hierarchy\n")
    md.append("AUC generally declines under stricter sequence-cluster holdouts. "
              "Taxonomic and sequence-cluster regimes are different axes and "
              "should not be treated as a strictly ordered scale.\n")
    md.append(_md(t2.round(3)))
    md.append("\n## 3. Graph message-passing ablation\n")
    graph_stats = pd.read_csv(rd / "gnn_ablation_cmp.csv").set_index("regime")
    cold_gain = graph_stats.loc["combined_unseen", "graph_gain"]
    cold_q = graph_stats.loc["combined_unseen", "graph_q_bh"]
    md.append("Message passing produced positive row-pooled AUROC differences in all "
              "four regimes; only the dual cold-start gain remained below the "
              f"exploratory BH threshold (Δ={cold_gain:+.3f}, q={cold_q:.3g}). "
              "The GBM nevertheless remained the strongest headline model. "
              "These row-wise tests are not entity-independent. See "
              "`figure_main.png` panel c.\n")
    md.append("## 4. Cocktail optimisation\n")
    md.append(f"- Minimum cocktail (exact ILP) over targets with predicted coverage: "
              f"**{ck['ilp_min_size']} phages** (greedy {ck['greedy_min_size']}, "
              f"oracle minimum {ck['ilp_oracle_min_size']}).")
    md.append("- Model-driven greedy is compared with a truth-informed greedy reference and "
              "random selection (see `cocktail_coverage.png`).\n")
    md.append("### Table 3. k-robust cocktails\n")
    md.append(_md(rob))
    md.append("\n## 5. Eco-evolutionary therapy simulation\n")
    robust = temp[temp["strategy"] == "robust_k2"].iloc[0]
    md.append("In the assumption-driven sensitivity model, the redundant "
              "strategy also rebounds and does not prevent resistant takeover; "
              f"its end resistant fraction is {robust['resistant_frac_end']:.3f}. "
              "This is not independent biological validation "
              "(see `temporal_dynamics.png`).\n")
    md.append("### Table 4. Therapy outcomes\n")
    md.append(_md(temp_disp))
    md.append("\n## Figures\n")
    md.append("- `figure_main.png` \u2014 generalisation, leakage hierarchy, GNN ablation")
    md.append("- `cocktail_coverage.png` \u2014 cocktail coverage vs size")
    md.append("- `temporal_dynamics.png` \u2014 therapy dynamics with resistance\n")
    (rd / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote %s", rd / "REPORT.md")
    log.info("Wrote table1-4 (csv) + table1 (tex)")


if __name__ == "__main__":
    main()
