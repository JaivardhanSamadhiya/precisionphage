"""Deterministic resistance scenario model for predicted phage selections.

A deterministic ODE system couples bacterial host populations (each split into a
phage-sensitive and a phage-resistant subpopulation) with a phage cocktail.
Infection rates are seeded by GBM-predicted susceptibility probabilities. The
system is an assumption stress test: its parameters are not fitted to biological
time courses, and its output is not mechanistic or efficacy validation.

State (per host species h):
    S_h  - phage-sensitive bacteria
    R_h  - phage-resistant bacteria (cross-resistant to the cocktail)
Per phage p:
    V_p  - free phage particles

Dynamics (logistic growth with a shared carrying capacity, mass-action infection,
phage decay + burst, and mutation S->R):

  dS_h/dt = r_h S_h (1 - N/K) - S_h * sum_p beta * A[p,h] V_p - mu_h r_h S_h
  dR_h/dt = r_h (1 - cost) R_h (1 - N/K) + mu_h r_h S_h
  dV_p/dt = burst * S_h-weighted infections - delta V_p - adsorption loss

where A[p,h] in [0,1] is the predicted infection probability (susceptibility),
N = sum_h (S_h + R_h) is total bacterial load. For n targeting phages,
mu_h = mu ** (1 + alpha * (n - 1)); alpha spans complete cross-resistance
(0) to independent resistance (1). `cost` is the assumed fitness cost.

Pure NumPy + scipy.integrate; no external services.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TherapyParams:
    r: float = 1.0            # bacterial max growth rate (1/h)
    K: float = 1e9            # carrying capacity (CFU/mL)
    beta: float = 1e-9        # adsorption/infection rate constant
    burst: float = 50.0       # phage burst size
    delta: float = 0.1        # phage decay rate (1/h)
    mu: float = 1e-7          # per-division resistance rate PER phage
    cost: float = 0.1         # fitness cost of resistance (0..1)
    resistance_independence: float = 1.0
    # 0 = complete cross-resistance; 1 = independent resistance; intermediate
    # values interpolate the exponent 1 + alpha * (n_targeting - 1).
    eff_thr: float = 0.5      # susceptibility above which a phage "targets" a host
    floor: float = 1.0        # extinction threshold (cells below this can't regrow)
    t_max: float = 72.0       # hours
    n_steps: int = 721
    dose: float = 1e8         # phage dose per selected phage at t=0


def simulate(A_sel: np.ndarray, S0: np.ndarray, p: TherapyParams,
             field_seed: int = 0):
    """Integrate the eco-evolutionary system.

    A_sel : [n_phage_selected, n_host] predicted susceptibility (0..1) of the
            cocktail's phages against each host species.
    S0    : [n_host] initial sensitive bacterial densities.
    Returns dict with time grid and trajectories (S, R, V, total).
    """
    from scipy.integrate import solve_ivp
    A_sel = np.atleast_2d(np.asarray(A_sel, dtype=float))
    n_v, n_h = A_sel.shape
    S0 = np.asarray(S0, dtype=float)
    R0 = np.zeros(n_h)
    V0 = np.full(n_v, p.dose)
    y0 = np.concatenate([S0, R0, V0])

    # Per-host effective resistance rate. alpha=1 recovers the original
    # independent-resistance assumption (mu ** n_targeting); alpha=0 represents
    # complete cross-resistance (mu for any targeted host).
    if not 0.0 <= p.resistance_independence <= 1.0:
        raise ValueError("resistance_independence must be between 0 and 1")
    n_targeting = (A_sel >= p.eff_thr).sum(0)          # [n_host]
    resistance_exponent = 1.0 + p.resistance_independence * np.maximum(
        n_targeting - 1, 0)
    mu_vec = np.where(n_targeting >= 1,
                      p.mu ** resistance_exponent, 0.0)

    def rhs(t, y):
        S = np.clip(y[:n_h], 0, None)
        R = np.clip(y[n_h:2 * n_h], 0, None)
        V = np.clip(y[2 * n_h:], 0, None)
        N = S.sum() + R.sum()
        logistic = 1.0 - N / p.K
        # extinction floor: subpopulations below ~1 cell cannot regrow
        S_g = np.where(S >= p.floor, S, 0.0)
        R_g = np.where(R >= p.floor, R, 0.0)
        infect_h = p.beta * (A_sel.T @ V)              # [n_host] predation
        growthS = p.r * S_g * logistic
        mut = mu_vec * growthS                          # S -> R flux
        dS = growthS - infect_h * S - mut
        dR = p.r * (1 - p.cost) * R_g * logistic + mut
        new_phage = p.burst * p.beta * (A_sel @ S_g) * V  # burst per phage
        dV = new_phage - p.delta * V
        return np.concatenate([dS, dR, dV])

    t_eval = np.linspace(0, p.t_max, p.n_steps)
    sol = solve_ivp(rhs, (0, p.t_max), y0, t_eval=t_eval, method="LSODA",
                    rtol=1e-6, atol=1.0, max_step=1.0)
    S = np.clip(sol.y[:n_h], 0, None)
    R = np.clip(sol.y[n_h:2 * n_h], 0, None)
    V = np.clip(sol.y[2 * n_h:], 0, None)
    total = S.sum(0) + R.sum(0)
    return {"t": sol.t, "S": S, "R": R, "V": V, "total": total,
            "success": sol.success}


def therapy_metrics(out: dict, S0_total: float) -> dict:
    """Summarise a trajectory: suppression, resistance takeover, clearance."""
    total = out["total"]
    R = out["R"].sum(0)
    t = out["t"]
    nadir = float(total.min())
    end = float(total[-1])
    log_drop = float(np.log10(max(S0_total, 1.0)) - np.log10(max(nadir, 1.0)))
    resist_frac_end = float(R[-1] / max(total[-1], 1.0))
    # time to 99% suppression (if reached)
    thr = 0.01 * S0_total
    below = np.where(total <= thr)[0]
    t_supp = float(t[below[0]]) if below.size else None
    # rebound: relapse above 10% of initial after a nadir
    rebound = bool(end > 0.1 * S0_total and nadir < 0.1 * S0_total)
    return {"nadir": nadir, "end_load": end, "log10_drop": log_drop,
            "resistant_fraction_end": resist_frac_end,
            "time_to_99pct_suppression_h": t_supp, "rebound": rebound}
