"""Phage cocktail optimisation over a predicted susceptibility matrix.

Given a boolean coverage matrix A[phage, host] (phage predicted to lyse host)
we solve two classic problems:

  * minimum-cardinality (k-)cover: fewest phages so every (coverable) target
    host is hit by >= k phages  -- greedy and exact ILP;
  * maximum coverage under a budget B: choose <= B phages covering the most
    targets -- greedy (1-1/e guarantee) and exact ILP.

k>1 yields cocktails that stay effective if a host evolves resistance to some
phages (redundancy = robustness).

Evaluation is always done against the TRUE (experimentally observed) coverage
matrix, so cocktails selected from model predictions are scored on real coverage.
"""
from __future__ import annotations

import numpy as np


def greedy_cover(A: np.ndarray, targets, k: int = 1, budget: int | None = None):
    """Greedy (weighted) set cover with k-redundancy. Returns phage indices in
    selection order. A is bool [n_phage, n_host]."""
    A = np.asarray(A, dtype=bool)
    n_p, n_h = A.shape
    targets = np.asarray(targets)
    avail = A[:, targets].sum(0)
    need_t = np.minimum(k, avail)                  # cap by what's coverable
    remaining = np.zeros(n_h, dtype=int)
    remaining[targets] = need_t
    used = np.zeros(n_p, dtype=bool)
    order = []
    while True:
        if budget is not None and len(order) >= budget:
            break
        active = remaining > 0
        if not active.any():
            break
        gains = (A[:, active]).sum(1).astype(int)
        gains[used] = -1
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            break
        used[j] = True
        order.append(j)
        remaining[A[j] & active] -= 1
    return order


def ilp_min_cover(A: np.ndarray, targets, k: int = 1):
    """Exact minimum-cardinality k-cover via MILP. Returns selected phage idx."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    A = np.asarray(A, dtype=bool)
    n_p = A.shape[0]
    targets = np.asarray(targets)
    avail = A[:, targets].sum(0)
    keep = avail > 0
    tcols = targets[keep]
    req = np.minimum(k, avail[keep]).astype(float)
    if tcols.size == 0:
        return np.array([], dtype=int)
    Acon = A[:, tcols].T.astype(float)             # [n_target, n_phage]
    res = milp(c=np.ones(n_p),
               constraints=LinearConstraint(Acon, req, np.inf),
               integrality=np.ones(n_p),
               bounds=Bounds(0, 1))
    if not res.success:
        return None
    return np.where(np.round(res.x).astype(bool))[0]


def ilp_max_cover(A: np.ndarray, targets, budget: int, k: int = 1):
    """Exact maximum k-coverage under a budget of `budget` phages.
    Returns (selected_phage_idx, n_targets_covered)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    A = np.asarray(A, dtype=bool)
    n_p = A.shape[0]
    targets = np.asarray(targets)
    avail = A[:, targets].sum(0)
    keep = avail >= k
    tcols = targets[keep]
    nt = tcols.size
    if nt == 0:
        return np.array([], dtype=int), 0
    # vars: x (n_p phages) then y (nt host indicators)
    nv = n_p + nt
    c = np.concatenate([np.zeros(n_p), -np.ones(nt)])      # maximise sum y
    rows, lb, ub = [], [], []
    # budget: sum x <= budget
    r = np.zeros(nv); r[:n_p] = 1.0
    rows.append(r); lb.append(-np.inf); ub.append(float(budget))
    # link: k*y_i - sum_j A[j,i] x_j <= 0
    Acol = A[:, tcols].astype(float)                        # [n_p, nt]
    for i in range(nt):
        r = np.zeros(nv)
        r[:n_p] = -Acol[:, i]
        r[n_p + i] = float(k)
        rows.append(r); lb.append(-np.inf); ub.append(0.0)
    res = milp(c=c,
               constraints=LinearConstraint(np.array(rows), lb, ub),
               integrality=np.ones(nv),
               bounds=Bounds(0, 1))
    if not res.success:
        return None, 0
    x = np.round(res.x[:n_p]).astype(bool)
    ncov = int(np.round(res.x[n_p:]).sum())
    return np.where(x)[0], ncov


def true_coverage_curve(order, T: np.ndarray, targets, k: int = 1) -> np.ndarray:
    """Fraction of target hosts truly covered by >= k of the first s phages, for
    s = 1..len(order). T is the TRUE bool coverage matrix [n_phage, n_host]."""
    T = np.asarray(T, dtype=bool)
    targets = np.asarray(targets)
    cov = np.zeros(T.shape[1], dtype=int)
    frac = []
    for j in order:
        cov += T[j].astype(int)
        frac.append(float((cov[targets] >= k).mean()))
    return np.asarray(frac)


def true_coverage(selected, T: np.ndarray, targets, k: int = 1) -> float:
    """Fraction of targets truly covered by >= k of the selected phages."""
    T = np.asarray(T, dtype=bool)
    targets = np.asarray(targets)
    if len(selected) == 0:
        return 0.0
    cov = T[np.asarray(selected)].sum(0)
    return float((cov[targets] >= k).mean())
