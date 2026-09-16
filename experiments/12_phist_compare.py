#!/usr/bin/env python3
"""Step 12: external baseline comparison vs PHIST (real tool).

PHIST (Zielezinski et al. 2021) is an alignment-free phage-host predictor that
scores a (phage, host) pair by the number of exact common k-mers (k=25). It is
unsupervised (no training), so its scores are fixed; we evaluate it on EXACTLY
the same leakage-controlled test pairs as our GBM, per regime, and compare with
DeLong's paired test (BH-FDR corrected).

Prereqs: run experiments/stage_phist.py then PHIST itself, producing
  external/phist_run/out/common_kmers.csv  (full pairwise common-kmer matrix).

Run (from /tmp, then cd in):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/12_phist_compare.py
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

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from precisionphage.eval import (  # noqa: E402
    benjamini_hochberg, bootstrap_auc_diff, delong_auc_ci, delong_test,
    permutation_auc_test,
)
from precisionphage.features.assembly import build_covered_dataset  # noqa: E402
from precisionphage.models import fit_predict_gbm, run_grouped_cv  # noqa: E402
from precisionphage.splits import (  # noqa: E402
    combined_unseen_folds, leave_one_group_out, load_or_build_clusters,
)
from precisionphage.utils import (  # noqa: E402
    ensure_dirs, get_logger, limit_threads, load_config, set_determinism,
)

log = get_logger("phist_cmp")

RLAB = {"loso_species": "Unseen species (LOSO)",
        "host_cluster": "Unseen host cluster",
        "phage_cluster": "Unseen phage cluster",
        "combined_unseen": "Both unseen (cold start)"}


def parse_phist_common(path: Path) -> dict:
    """Parse PHIST common_kmers.csv into {(phage_file, host_file): n_common}."""
    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    phage_cols = [c.strip() for c in header[2:]]
    scores = {}
    for line in lines[2:]:                       # skip header + total-kmers row
        if not line.strip():
            continue
        toks = line.split(",")
        host = toks[0].strip()
        if not host:
            continue
        for t in toks[2:]:
            t = t.strip()
            if not t or ":" not in t:
                continue
            idx, cnt = t.split(":")
            ph = phage_cols[int(idx) - 1]
            scores[(ph, host)] = float(cnt)
    return scores


def _assign_clusters(cfg, data):
    return load_or_build_clusters(cfg, data)


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
    pc, hc = _assign_clusters(cfg, data)
    cov["phage_cluster"] = cov["phage"].map(pc).astype(int)
    cov["host_cluster"] = cov["host"].map(hc).astype(int)

    # --- PHIST per-pair score (common 25-mers; 0 if no shared k-mers) ---
    common = parse_phist_common(ROOT / "external" / "phist_run" / "out"
                                / "common_kmers.csv")
    pfile = {n: (data.phage_index.resolve(n).name
                 if data.phage_index.resolve(n) else None) for n in cov["phage"].unique()}
    hfile = {n: (data.host_index.resolve(n).name
                 if data.host_index.resolve(n) else None) for n in cov["host"].unique()}
    phist = np.zeros(len(cov), dtype=np.float64)
    for r in range(len(cov)):
        key = (pfile.get(cov.at[r, "phage"]), hfile.get(cov.at[r, "host"]))
        phist[r] = common.get(key, 0.0)
    nz = int((phist > 0).sum())
    log.info("PHIST scored: %d/%d covered pairs have >0 common 25-mers (%.1f%%)",
             nz, len(cov), 100 * nz / len(cov))

    mpos, mneg = cfg["data"]["min_pos_per_group"], cfg["data"]["min_neg_per_group"]
    sp = cfg["splits"]
    regimes = {
        "loso_species": list(leave_one_group_out(cov, "host_species", "loso", mpos, mneg)),
        "host_cluster": list(leave_one_group_out(cov, "host_cluster", "host_cluster", mpos, mneg)),
        "phage_cluster": list(leave_one_group_out(cov, "phage_cluster", "phage_cluster", mpos, mneg)),
        "combined_unseen": list(combined_unseen_folds(cov, "phage_cluster", "host_cluster",
                                                      sp["n_combined_splits"], seed, mpos, mneg)),
    }

    rows, cmp_rows, cmp_pvals = [], [], []
    preds_out = {}
    for rname, folds in regimes.items():
        if not folds:
            continue
        gbm = run_grouped_cv(cov, X, folds, fit_predict_gbm, seed,
                             cluster_col="host_cluster", cfg=cfg)
        pooled_idx = np.concatenate([f.test_idx for f in folds])
        y = gbm["pooled_y"].astype(int)
        if not np.array_equal(y, cov["label"].to_numpy()[pooled_idx].astype(int)):
            raise AssertionError(f"{rname}: pooled index reconstruction misaligned")
        g = gbm["pooled_p"].astype(float)
        p = phist[pooled_idx]
        preds_out[f"{rname}__y"] = y
        preds_out[f"{rname}__GBM"] = g
        preds_out[f"{rname}__PHIST"] = p

        for mdl, sc in (("GBM", g), ("PHIST", p)):
            ci = delong_auc_ci(y, sc, cfg["eval"]["bootstrap_ci"])
            perm = permutation_auc_test(y, sc, cfg["eval"]["permutation_n"], seed)
            rows.append({"regime": rname, "model": mdl, "n": len(y),
                         "auc": ci["auc"], "auc_lo": ci["lo"], "auc_hi": ci["hi"],
                         "auprc": average_precision_score(y, sc),
                         "perm_p": perm["p"]})
        dt = delong_test(y, g, p)
        bd = bootstrap_auc_diff(y, g, p, cfg["eval"]["bootstrap_n"],
                                cfg["eval"]["bootstrap_ci"], seed)
        cmp_rows.append({"regime": rname, "auc_gbm": dt["auc1"],
                         "auc_phist": dt["auc2"], "auc_diff": dt["auc_diff"],
                         "diff_lo": bd["lo"], "diff_hi": bd["hi"],
                         "delong_z": dt["z"], "delong_p": dt["p"]})
        cmp_pvals.append(dt["p"])
        log.info("[%s] n=%d  GBM AUC=%.3f  PHIST AUC=%.3f  diff=%.3f",
                 rname, len(y), dt["auc1"], dt["auc2"], dt["auc_diff"])

    for row, q in zip(rows, benjamini_hochberg([r["perm_p"] for r in rows])):
        row["perm_q_bh"] = float(q)
    for row, q in zip(cmp_rows, benjamini_hochberg(cmp_pvals)):
        row["delong_q_bh"] = float(q)
        row["gbm_beats_phist_fdr05"] = bool(q < 0.05 and row["auc_diff"] > 0)

    skill = pd.DataFrame(rows)
    cmp = pd.DataFrame(cmp_rows)
    skill.to_csv(rd / "phist_skill.csv", index=False)
    cmp.to_csv(rd / "phist_modelcmp.csv", index=False)
    np.savez(rd / "phist_pooled_preds.npz", **preds_out)
    (rd / "phist_compare.json").write_text(json.dumps(
        {"phist_nonzero_frac": nz / len(cov), "skill": rows, "comparison": cmp_rows},
        indent=2, default=float))
    log.info("SKILL (GBM vs PHIST):\n%s", skill.round(4).to_string(index=False))
    log.info("COMPARISON (DeLong, BH):\n%s", cmp.round(4).to_string(index=False))

    # --- figure: AUC bars GBM vs PHIST + ROC overlays ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc as _auc
        from sklearn.metrics import roc_curve
        order = [r for r in RLAB if r in regimes and regimes[r]]
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        xx = np.arange(len(order)); w = 0.38
        gb = [cmp.set_index("regime").loc[r, "auc_gbm"] for r in order]
        ph = [cmp.set_index("regime").loc[r, "auc_phist"] for r in order]
        ax[0].bar(xx - w / 2, gb, w, color="#1f77b4", label="Our model (GBM)")
        ax[0].bar(xx + w / 2, ph, w, color="#9467bd", label="PHIST (k=25)")
        ax[0].axhline(0.5, color="k", ls="--", lw=1)
        for i, r in enumerate(order):
            q = cmp.set_index("regime").loc[r, "delong_q_bh"]
            ax[0].text(i, max(gb[i], ph[i]) + 0.01, f"q={q:.1e}", ha="center", fontsize=8)
        ax[0].set_xticks(xx); ax[0].set_xticklabels([RLAB[r] for r in order], fontsize=9)
        ax[0].set_ylabel("AUROC"); ax[0].set_ylim(0.4, 1.0)
        ax[0].set_title("a  Our model vs PHIST across leakage regimes")
        ax[0].legend(loc="lower left"); ax[0].grid(axis="y", alpha=0.3)

        for r in order:
            y = preds_out[f"{r}__y"]
            fpr, tpr, _ = roc_curve(y, preds_out[f"{r}__GBM"])
            ax[1].plot(fpr, tpr, lw=1.8, label=f"GBM {RLAB[r][:14]} ({_auc(fpr, tpr):.2f})")
        for r in order:
            y = preds_out[f"{r}__y"]
            fpr, tpr, _ = roc_curve(y, preds_out[f"{r}__PHIST"])
            ax[1].plot(fpr, tpr, lw=1.4, ls="--",
                       label=f"PHIST {RLAB[r][:14]} ({_auc(fpr, tpr):.2f})")
        ax[1].plot([0, 1], [0, 1], "k:", lw=1)
        ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
        ax[1].set_title("b  ROC: GBM (solid) vs PHIST (dashed)")
        ax[1].legend(fontsize=6.5, loc="lower right", ncol=2); ax[1].grid(alpha=0.3)
        fig.suptitle("External baseline comparison: PrecisionPhage GBM vs PHIST", fontsize=13)
        fig.tight_layout(); fig.savefig(rd / "fig_phist_compare.png", dpi=180)
        log.info("wrote fig_phist_compare.png")
    except Exception as e:
        log.warning("figure skipped (%s)", e)


if __name__ == "__main__":
    main()
