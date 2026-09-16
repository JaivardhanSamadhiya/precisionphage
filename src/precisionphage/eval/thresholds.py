"""Nested, group-aware out-of-fold probabilities and binary decisions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def _scaled_predict(X, y, train_idx, test_idx, predictor, seed):
    scaler = StandardScaler().fit(X[train_idx])
    x_train = np.nan_to_num(scaler.transform(X[train_idx])).astype(np.float32)
    x_test = np.nan_to_num(scaler.transform(X[test_idx])).astype(np.float32)
    return predictor(x_train, y[train_idx], x_test, seed)


def nested_group_oof_decisions(X, y, groups, predictor, seed: int,
                               n_splits: int = 5, inner_splits: int = 4):
    """Return outer-OOF probabilities and leakage-free binary decisions.

    Each outer fold receives a threshold selected by F1 on group-aware inner-OOF
    predictions from that outer fold's training rows. The outer test labels are
    never used to choose their threshold.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    probabilities = np.full(len(y), np.nan, dtype=np.float32)
    decisions = np.zeros(len(y), dtype=bool)
    thresholds: list[float] = []
    grid = np.round(np.arange(0.10, 0.91, 0.02), 3)
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                 random_state=seed)
    for fold_no, (train, test) in enumerate(outer.split(X, y, groups)):
        inner_prob = np.full(len(train), np.nan, dtype=np.float32)
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True,
                                     random_state=seed + fold_no + 1)
        for inner_train, inner_val in inner.split(
                X[train], y[train], groups[train]):
            fit_rows = train[inner_train]
            val_rows = train[inner_val]
            inner_prob[inner_val] = _scaled_predict(
                X, y, fit_rows, val_rows, predictor, seed + fold_no)
        if np.isnan(inner_prob).any():
            raise AssertionError("inner OOF predictions are incomplete")
        scores = [f1_score(y[train], inner_prob >= t, zero_division=0)
                  for t in grid]
        threshold = float(grid[int(np.argmax(scores))])
        thresholds.append(threshold)
        probabilities[test] = _scaled_predict(
            X, y, train, test, predictor, seed + fold_no)
        decisions[test] = probabilities[test] >= threshold
    if np.isnan(probabilities).any():
        raise AssertionError("outer OOF predictions are incomplete")
    return probabilities, decisions, thresholds
