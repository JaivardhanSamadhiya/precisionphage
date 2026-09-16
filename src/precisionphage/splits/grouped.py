"""Grouped leave-one-group-out splits with validity filtering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    name: str
    regime: str
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self):
        # leakage invariant: train and test row sets are disjoint
        if np.intersect1d(self.train_idx, self.test_idx).size:
            raise ValueError(f"Fold {self.name}: train/test indices overlap")


def leave_one_group_out(df: pd.DataFrame, group_col: str, regime: str,
                        min_pos: int = 3, min_neg: int = 3):
    """Yield a Fold per group that has >= min_pos positives and min_neg negatives.

    Test = rows of the held-out group; train = all other rows. Groups too small
    to evaluate are never used as a test fold but remain available for training.
    """
    idx = np.arange(len(df))
    labels = df["label"].to_numpy()
    groups = df[group_col].to_numpy()
    for g in pd.unique(groups):
        test_mask = groups == g
        n_pos = int(labels[test_mask].sum())
        n_neg = int((labels[test_mask] == 0).sum())
        if n_pos < min_pos or n_neg < min_neg:
            continue
        yield Fold(name=str(g), regime=regime,
                   train_idx=idx[~test_mask], test_idx=idx[test_mask])


def cross_study_folds(df: pd.DataFrame, study_col: str = "study",
                      min_pos: int = 3, min_neg: int = 3):
    """Train on all-but-one study, test on the held-out study (domain shift)."""
    yield from leave_one_group_out(df, study_col, regime="cross_study",
                                   min_pos=min_pos, min_neg=min_neg)
