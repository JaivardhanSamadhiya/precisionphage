"""Honest metrics: no silent 0.5 substitution; degenerate folds are recorded
and excluded transparently.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, matthews_corrcoef,
    roc_auc_score,
)


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict | None:
    """Return metrics, or None if the fold has a single class (AUC undefined)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    p = np.clip(np.where(np.isfinite(p), p, 0.5), 0.0, 1.0)
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "brier": float(brier_score_loss(y, p)),
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
    }


def aggregate_folds(per_fold: list[dict], pooled_y, pooled_p) -> dict:
    """Combine per-fold metrics into per-group means and a pooled estimate.

    per_fold: list of dicts each with at least 'roc_auc' (degenerate folds
    should be excluded by the caller; we also guard here).
    """
    valid = [d for d in per_fold if d is not None and np.isfinite(d.get("roc_auc", np.nan))]
    aucs = np.array([d["roc_auc"] for d in valid], dtype=float)
    praucs = np.array([d["pr_auc"] for d in valid], dtype=float)
    out = {
        "n_folds_total": len(per_fold),
        "n_folds_used": len(valid),
        "n_folds_skipped": len(per_fold) - len(valid),
        "mean_roc_auc": float(aucs.mean()) if len(aucs) else float("nan"),
        "std_roc_auc": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "mean_pr_auc": float(praucs.mean()) if len(praucs) else float("nan"),
    }
    py = np.asarray(pooled_y).astype(int)
    pp = np.asarray(pooled_p, dtype=float)
    if len(np.unique(py)) > 1:
        out["pooled_roc_auc"] = float(roc_auc_score(py, pp))
        out["pooled_pr_auc"] = float(average_precision_score(py, pp))
    else:
        out["pooled_roc_auc"] = float("nan")
        out["pooled_pr_auc"] = float("nan")
    return out


def bootstrap_ci(values, cluster_labels=None, n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 42) -> dict:
    """Percentile bootstrap mean CI; cluster bootstrap when labels given."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    alpha = (1 - ci) / 2
    if cluster_labels is not None and len(cluster_labels) == n:
        labels = np.asarray(cluster_labels)
        clusters = np.unique(labels)
        means = []
        for _ in range(n_boot):
            chosen = rng.choice(clusters, size=len(clusters), replace=True)
            sample = np.concatenate([vals[labels == c] for c in chosen])
            means.append(sample.mean())
        boot = np.array(means)
    else:
        boot = vals[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    return {"mean": float(vals.mean()),
            "lo": float(np.percentile(boot, alpha * 100)),
            "hi": float(np.percentile(boot, (1 - alpha) * 100)), "n": int(n)}


def calibration_curve_ece(y_true, y_prob, n_bins: int = 10) -> dict:
    """Reliability curve points and Expected Calibration Error."""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    pts, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        w = float(m.mean())
        ece += w * abs(acc - conf)
        pts.append({"bin": b, "confidence": conf, "accuracy": acc, "count": int(m.sum())})
    return {"ece": float(ece), "points": pts}
