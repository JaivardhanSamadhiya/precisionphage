"""Statistical significance for model comparison and skill-vs-chance.

Implemented:
  * Fast DeLong (Sun & Xu, 2014) for the variance of one AUC and for the
    covariance of two AUCs estimated on the SAME samples -> analytic CI and a
    paired two-sided test that AUC_1 != AUC_2.
  * Label-permutation test that an AUC is above chance (exact-ish, +1 smoothing).
  * McNemar's paired test for two classifiers' error profiles.
  * Paired bootstrap CI for an AUC difference.
  * Benjamini-Hochberg FDR correction for a family of p-values.

These let us state, with correction for multiple comparisons, whether a model
beats chance under each leakage-controlled regime and whether two models differ.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# --- fast DeLong -------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """preds_sorted: [k, n] with the m positive-label columns first."""
    k, total = preds_sorted.shape
    n = total - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, total])
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def _order_positives_first(y_true: np.ndarray):
    order = np.argsort(-y_true)  # label 1 sorts before label 0
    return order, int(np.sum(y_true == 1))


def delong_auc_variance(y_true, y_score) -> tuple[float, float]:
    """Return (AUC, variance) for a single score vector via DeLong."""
    y = np.asarray(y_true).astype(int)
    order, m = _order_positives_first(y)
    preds = np.asarray(y_score, float)[order][np.newaxis, :]
    aucs, cov = _fast_delong(preds, m)
    return float(aucs[0]), float(np.atleast_2d(cov)[0, 0])


def delong_auc_ci(y_true, y_score, ci: float = 0.95) -> dict:
    auc, var = delong_auc_variance(y_true, y_score)
    se = float(np.sqrt(max(var, 0.0)))
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    lo, hi = auc - z * se, auc + z * se
    return {"auc": auc, "se": se,
            "lo": float(np.clip(lo, 0, 1)), "hi": float(np.clip(hi, 0, 1))}


def delong_test(y_true, score1, score2) -> dict:
    """Paired two-sided test AUC_1 != AUC_2 on identical samples."""
    y = np.asarray(y_true).astype(int)
    order, m = _order_positives_first(y)
    preds = np.vstack([np.asarray(score1, float), np.asarray(score2, float)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    cov = np.atleast_2d(cov)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]),
            "auc_diff": float(aucs[0] - aucs[1]), "z": float(z), "p": p}


# --- permutation, McNemar, bootstrap -----------------------------------------

def permutation_auc_test(y_true, y_score, n_perm: int = 1000, seed: int = 42) -> dict:
    """Test AUC > chance by permuting labels. p = (#perm AUC >= obs + 1)/(n+1)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score, float)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "p": float("nan"), "n_perm": 0}
    obs = roc_auc_score(y, p)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if roc_auc_score(yp, p) >= obs:
            count += 1
    return {"auc": float(obs), "p": float((count + 1) / (n_perm + 1)),
            "n_perm": int(n_perm)}


def mcnemar_test(y_true, pred1, pred2) -> dict:
    """Paired test of two classifiers' discordant errors (exact for small n)."""
    y = np.asarray(y_true).astype(int)
    c1 = np.asarray(pred1).astype(int) == y
    c2 = np.asarray(pred2).astype(int) == y
    b = int(np.sum(c1 & ~c2))   # model1 right, model2 wrong
    c = int(np.sum(~c1 & c2))   # model1 wrong, model2 right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "stat": 0.0, "p": 1.0}
    if n < 25:
        p = float(stats.binomtest(min(b, c), n, 0.5).pvalue)
        stat = float(min(b, c))
    else:
        stat = float((abs(b - c) - 1) ** 2 / n)
        p = float(stats.chi2.sf(stat, 1))
    return {"b": b, "c": c, "stat": stat, "p": p}


def bootstrap_auc_diff(y_true, score1, score2, n_boot: int = 10000,
                       ci: float = 0.95, seed: int = 42) -> dict:
    """Paired percentile-bootstrap CI for AUC_1 - AUC_2 on identical samples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    s1 = np.asarray(score1, float)
    s2 = np.asarray(score2, float)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        diffs.append(roc_auc_score(ys, s1[idx]) - roc_auc_score(ys, s2[idx]))
    diffs = np.asarray(diffs)
    a = (1 - ci) / 2
    obs = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    return {"diff": float(obs),
            "lo": float(np.percentile(diffs, a * 100)) if diffs.size else float("nan"),
            "hi": float(np.percentile(diffs, (1 - a) * 100)) if diffs.size else float("nan")}


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH FDR-adjusted q-values (monotone, clipped to [0,1])."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out
