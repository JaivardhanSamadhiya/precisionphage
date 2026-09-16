#!/usr/bin/env python3
"""Step 9: deterministic resistance scenario analysis.

Using host-cluster-grouped OOF GBM susceptibility predictions, we pick a
multi-species illustrative panel and
simulate four strategies forward in time with resistance evolution:
  * no phage (control),
  * best single phage (monophage),
  * model-designed cocktail (greedy, k=1),
  * redundancy-constrained cocktail (greedy, k=2).

The primary scenario assumes independent resistance to every targeting phage.
A factorial sensitivity grid also spans complete cross-resistance, an
intermediate dependence assumption, mutation rate, and fitness cost. This is
an assumption stress test, not a fitted mechanistic or efficacy model.

Run (from /tmp, then cd in):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/09_temporal.py
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

from precisionphage.cocktail import greedy_cover  # noqa: E402
from precisionphage.eval import nested_group_oof_decisions  # noqa: E402
from precisionphage.features.assembly import build_covered_dataset  # noqa: E402
from precisionphage.models import fit_predict_gbm  # noqa: E402
from precisionphage.splits import load_or_build_clusters  # noqa: E402
from precisionphage.temporal import TherapyParams, simulate, therapy_metrics  # noqa: E402
from precisionphage.utils import (  # noqa: E402
    ensure_dirs, get_logger, limit_threads, load_config, set_determinism,
)

log = get_logger("temporal")


def _host_clusters(cfg, data):
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
    hc = _host_clusters(cfg, data)
    groups = cov["host"].map(hc).to_numpy()
    oof_cache = rd / "cocktail_oof_predictions.npz"
    pair_keys = (cov["phage"].astype(str) + "\t" + cov["host"].astype(str)).to_numpy(
        dtype=str)
    if oof_cache.exists():
        cached = np.load(oof_cache)
        cached_keys = cached["pair_keys"].astype(str)
        if np.array_equal(cached_keys, pair_keys):
            oof = cached["probabilities"].astype(np.float32)
            oof_decision = cached["decisions"].astype(bool)
            fold_thresholds = cached["thresholds"].astype(float).tolist()
            log.info("Loaded nested-threshold OOF predictions from %s", oof_cache.name)
        else:
            raise AssertionError("cocktail OOF cache pair order does not match dataset")
    else:
        oof, oof_decision, fold_thresholds = nested_group_oof_decisions(
            X, y, groups, fit_predict_gbm, seed)

    phages = sorted(cov["phage"].unique())
    hosts = sorted(cov["host"].unique())
    pix = {p: i for i, p in enumerate(phages)}
    hix = {h: i for i, h in enumerate(hosts)}
    n_p, n_h = len(phages), len(hosts)
    Aprob = np.zeros((n_p, n_h), dtype=float)
    T = np.zeros((n_p, n_h), dtype=bool)
    for r in range(len(cov)):
        pi, hi = pix[cov.at[r, "phage"]], hix[cov.at[r, "host"]]
        Aprob[pi, hi] = max(Aprob[pi, hi], float(oof[r]))
        if y[r] == 1:
            T[pi, hi] = True
    A = np.zeros((n_p, n_h), dtype=bool)
    for r in range(len(cov)):
        pi, hi = pix[cov.at[r, "phage"]], hix[cov.at[r, "host"]]
        A[pi, hi] = A[pi, hi] or bool(oof_decision[r])

    # infection panel = the best-covered host taxa (each with multiple
    # candidate phages) so cocktails can provide per-host redundancy and the
    # resistance-prevention benefit of k>1 cocktails is observable.
    coverable = T.sum(0) >= 1
    cand_count = A.sum(0)                              # predicted phages per host
    elig = np.where(coverable & (cand_count >= 3))[0]
    if len(elig) == 0:                                 # fallback: most-covered
        elig = np.where(coverable)[0]
    panel = elig[np.argsort(cand_count[elig])[::-1][:5]]
    log.info("infection panel: %d taxa (candidate phages/taxon=%s): %s",
             len(panel), cand_count[panel].tolist(),
             [hosts[i][:24] for i in panel])

    # candidate phages with >=1 predicted edge to the panel
    cand = np.where(A[:, panel].any(1))[0]
    Apanel = A[np.ix_(cand, panel)]
    Pprob = Aprob[np.ix_(cand, panel)]
    panel_local = np.arange(len(panel))

    # strategies -> selected phage rows (indices into cand) and susceptibility
    mono = [int(np.argmax(Apanel.sum(1)))]
    greedy1 = greedy_cover(Apanel, panel_local, k=1)
    greedy2 = greedy_cover(Apanel, panel_local, k=2)
    strategies = {
        "control": np.zeros((1, len(panel))),
        "monophage": Pprob[mono, :],
        "cocktail_k1": Pprob[greedy1, :] if greedy1 else Pprob[mono, :],
        "robust_k2": Pprob[greedy2, :] if greedy2 else Pprob[mono, :],
    }
    sizes = {"control": 0, "monophage": 1, "cocktail_k1": len(greedy1),
             "robust_k2": len(greedy2)}

    S0 = np.full(len(panel), 1e6)
    rows, traj = [], {}
    temporal_cfg = cfg["temporal"]
    horizon = float(temporal_cfg["horizon_hours"])
    dt = float(temporal_cfg["dt_hours"])
    n_steps = int(round(horizon / dt)) + 1
    for name, A_sel in strategies.items():
        pp = TherapyParams(
            dose=0.0 if name == "control" else 1e8,
            mu=float(temporal_cfg["mutation_rate"]),
            cost=float(temporal_cfg["resistance_cost"]),
            burst=float(temporal_cfg["burst_size"]),
            beta=float(temporal_cfg["adsorption_rate"]),
            t_max=horizon,
            n_steps=n_steps,
            resistance_independence=1.0,
        )
        out = simulate(A_sel, S0, pp)
        m = therapy_metrics(out, S0.sum())
        traj[name] = {"t": out["t"], "total": out["total"],
                      "R": out["R"].sum(0)}
        rows.append({"strategy": name, "n_phages": sizes[name],
                     "end_load": m["end_load"], "nadir": m["nadir"],
                     "log10_drop": round(m["log10_drop"], 2),
                     "resistant_frac_end": round(m["resistant_fraction_end"], 3),
                     "rebound": m["rebound"]})
    tbl = pd.DataFrame(rows)
    log.info("THERAPY OUTCOMES (panel of %d strains):\n%s", len(panel),
             tbl.to_string(index=False))

    tbl.to_csv(rd / "temporal_outcomes.csv", index=False)

    # Stress-test the assumption that resistance to multiple targeting phages
    # must be acquired independently. alpha=0 gives complete cross-resistance
    # (mu_eff=mu), alpha=1 gives independence (mu_eff=mu**n), and alpha=0.5 is
    # an explicitly phenomenological intermediate. This grid does not estimate
    # any of these quantities from experimental data.
    alpha_values = [0.0, 0.5, 1.0]
    mutation_values = [1e-8, 1e-7, 1e-6]
    cost_values = [0.0, 0.05, 0.10]
    alpha_labels = {
        0.0: "complete_cross_resistance",
        0.5: "partial_dependence",
        1.0: "independent_resistance",
    }
    sensitivity_rows = []
    for name in ("cocktail_k1", "robust_k2"):
        for alpha in alpha_values:
            for mutation_rate in mutation_values:
                for resistance_cost in cost_values:
                    pp = TherapyParams(
                        dose=1e8,
                        mu=mutation_rate,
                        cost=resistance_cost,
                        resistance_independence=alpha,
                        burst=float(temporal_cfg["burst_size"]),
                        beta=float(temporal_cfg["adsorption_rate"]),
                        t_max=horizon,
                        n_steps=n_steps,
                    )
                    out = simulate(strategies[name], S0, pp)
                    m = therapy_metrics(out, S0.sum())
                    sensitivity_rows.append({
                        "strategy": name,
                        "n_phages": sizes[name],
                        "resistance_independence": alpha,
                        "resistance_structure": alpha_labels[alpha],
                        "mutation_rate": mutation_rate,
                        "resistance_cost": resistance_cost,
                        "end_load": m["end_load"],
                        "nadir": m["nadir"],
                        "log10_drop": m["log10_drop"],
                        "resistant_fraction_end": m["resistant_fraction_end"],
                        "resistant_takeover": bool(m["resistant_fraction_end"] >= 0.5),
                        "rebound": m["rebound"],
                        "time_to_99pct_suppression_h": m["time_to_99pct_suppression_h"],
                    })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(rd / "temporal_resistance_sensitivity.csv", index=False)
    sensitivity_summary = []
    for (name, alpha, label), group in sensitivity.groupby(
            ["strategy", "resistance_independence", "resistance_structure"], sort=True):
        sensitivity_summary.append({
            "strategy": name,
            "n_phages": int(group["n_phages"].iloc[0]),
            "resistance_independence": float(alpha),
            "resistance_structure": label,
            "n_scenarios": int(len(group)),
            "rebound_scenarios": int(group["rebound"].sum()),
            "resistant_takeover_scenarios": int(group["resistant_takeover"].sum()),
            "nadir_min": float(group["nadir"].min()),
            "nadir_max": float(group["nadir"].max()),
            "end_load_min": float(group["end_load"].min()),
            "end_load_max": float(group["end_load"].max()),
            "log10_drop_min": float(group["log10_drop"].min()),
            "log10_drop_max": float(group["log10_drop"].max()),
        })
    (rd / "temporal_summary.json").write_text(json.dumps(
        {"threshold_method": "fold-specific F1 on group-aware inner-OOF training predictions",
         "fold_thresholds": fold_thresholds,
         "threshold_median": float(np.median(fold_thresholds)),
         "panel_taxa": [hosts[i] for i in panel],
         "panel_size": int(len(panel)),
         "parameters": {**temporal_cfg, "n_steps": n_steps,
                        "dose_per_phage": 1e8,
                        "resistance_independence": 1.0,
                        "resistance_assumption": "independent across targeting phages"},
         "panel_selection": "five eligible taxa with the most predicted candidate phages",
         "outcomes": rows,
         "resistance_sensitivity": {
             "interpretation": "assumption stress test; not fitted mechanistic or efficacy validation",
             "formula": "mu_eff = mu ** (1 + alpha * (n_targeting - 1))",
             "alpha_values": alpha_values,
             "mutation_rates": mutation_values,
             "resistance_costs": cost_values,
             "n_runs": int(len(sensitivity)),
             "summary": sensitivity_summary,
         }}, indent=2, default=float))

    # Persist trajectories so figures can be regenerated without rerunning simulation
    traj_save = {k: v for k, v in traj.items()}
    np.savez(rd / "temporal_trajectory.npz", **traj_save)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"control": "#7f7f7f", "monophage": "#d62728",
                  "cocktail_k1": "#1f77b4", "robust_k2": "#2ca02c"}
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for name, d in traj.items():
            lab = f"{name} (n={sizes[name]})"
            ax[0].plot(d["t"], np.maximum(d["total"], 1), label=lab,
                       color=colors[name], lw=2)
            ax[1].plot(d["t"], np.maximum(d["R"], 1), label=lab,
                       color=colors[name], lw=2)
        for a, ttl, yl in ((ax[0], "Total bacterial load", "CFU/mL (total)"),
                           (ax[1], "Resistant subpopulation", "CFU/mL (resistant)")):
            a.set_yscale("log"); a.set_xlabel("time (h)"); a.set_ylabel(yl)
            a.set_title(ttl); a.grid(alpha=0.3); a.legend(loc="lower right")
        fig.suptitle("Illustrative unfitted-parameter trajectories (not biological prediction)")
        fig.tight_layout()
        fig.savefig(rd / "temporal_dynamics.png", dpi=300)
        log.info("Wrote figure %s", rd / "temporal_dynamics.png")
    except Exception as e:
        log.warning("figure skipped (%s)", e)

    log.info("Wrote temporal outcomes, resistance-sensitivity grid, and summary")


if __name__ == "__main__":
    main()
