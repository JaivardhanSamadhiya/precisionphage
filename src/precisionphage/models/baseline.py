"""Feature-based baselines and a generic leakage-safe grouped-CV runner.

All scaling is fit on the training fold only. The same runner is reused for the
GNN so baselines and the graph model are compared under identical folds.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from ..eval.metrics import aggregate_folds, binary_metrics
from ..utils import get_logger, limit_threads, resolve_n_jobs

log = get_logger(__name__)


def fit_predict_gbm(X_tr, y_tr, X_te, seed: int) -> np.ndarray:
    """Gradient-boosted trees; prefers XGBoost, falls back to HistGB.

    Thread count honours PP_THREADS (set per worker) to avoid oversubscription
    when folds run in parallel processes."""
    n_thr = int(os.environ.get("PP_THREADS", "4"))
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            tree_method="hist", n_jobs=n_thr, eval_metric="logloss",
            verbosity=0)
    except Exception:
        clf = HistGradientBoostingClassifier(
            max_iter=400, max_depth=5, learning_rate=0.05, random_state=seed)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1].astype(np.float32)


# --- tuned, class-weighted GBM with leakage-free nested model selection ------
# A small principled grid (over/under-fit controls). For each OUTER fold we pick
# the config by group-aware inner CV on the TRAINING rows only, then refit on the
# full training fold. The test fold is never seen during selection -> no
# optimisation leakage. scale_pos_weight is set from the training labels only.
_GBM_GRID = (
    {"max_depth": 3, "min_child_weight": 5, "reg_lambda": 3.0,
     "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "min_child_weight": 3, "reg_lambda": 1.0,
     "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "min_child_weight": 1, "reg_lambda": 1.0,
     "subsample": 0.9, "colsample_bytree": 0.8},
)


def _inner_folds(y, groups, k, seed):
    """k inner (train, val) splits; group-aware when groups are usable, else
    stratified. Each val is guaranteed both classes when at all possible."""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y)
    n = len(y)
    if groups is not None:
        uniq = np.unique(groups)
        if len(uniq) >= k:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(uniq)
            buckets = np.array_split(perm, k)
            folds = []
            for b in buckets:
                va = np.isin(groups, b)
                tr = ~va
                if (va.sum() == 0 or tr.sum() == 0
                        or len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2):
                    continue
                folds.append((np.where(tr)[0], np.where(va)[0]))
            if len(folds) >= 2:
                return folds
    # fallback: stratified
    if min(np.bincount(y.astype(int))) < k:
        k = max(2, int(min(np.bincount(y.astype(int)))))
    try:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in skf.split(np.zeros(n), y)]
    except Exception:
        return []


def _spw(y) -> float:
    y = np.asarray(y).astype(int)
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    if npos == 0:
        return 1.0
    return float(np.clip(nneg / max(1, npos), 1.0, 50.0))


def fit_predict_gbm_tuned(X_tr, y_tr, X_te, seed: int, groups_tr=None) -> np.ndarray:
    """XGBoost with nested, group-aware model selection (leakage-free).

    The search grid spans over/under-fit controls AND the class-weighting choice
    (scale_pos_weight in {1, balanced}); the winner is chosen by mean inner-fold
    AUC, so weighting is only applied where it actually improves ranking. A final
    isotonic calibration (monotonic -> AUC-preserving) is fit on inner out-of-fold
    predictions to improve probability calibration without changing the ranking.

    Falls back to a fixed config when the training fold is too small or XGBoost is
    unavailable."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    X_tr = np.asarray(X_tr, np.float32)
    y_tr = np.asarray(y_tr).astype(int)
    X_te = np.asarray(X_te, np.float32)
    n_thr = int(os.environ.get("PP_THREADS", "4"))
    spw_bal = _spw(y_tr)

    try:
        from xgboost import XGBClassifier
    except Exception:
        clf = HistGradientBoostingClassifier(
            max_iter=400, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=seed)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1].astype(np.float32)

    def _make(cfg, spw, n_est):
        return XGBClassifier(
            n_estimators=n_est, learning_rate=0.05, random_state=seed,
            tree_method="hist", n_jobs=n_thr, eval_metric="logloss",
            scale_pos_weight=spw, verbosity=0, **cfg)

    spw_opts = [1.0] if spw_bal <= 1.0 else [1.0, spw_bal]
    candidates = [(cfg, spw) for cfg in _GBM_GRID for spw in spw_opts]

    inner = _inner_folds(y_tr, groups_tr, 3, seed) if len(y_tr) >= 40 else []
    best = (_GBM_GRID[1], 1.0, 400)
    if inner:
        best_score = -1.0
        for cfg, spw in candidates:
            aucs, iters = [], []
            for tr, va in inner:
                if len(np.unique(y_tr[tr])) < 2 or len(np.unique(y_tr[va])) < 2:
                    continue
                clf = _make(cfg, spw, 800)
                clf.set_params(early_stopping_rounds=40)
                try:
                    clf.fit(X_tr[tr], y_tr[tr],
                            eval_set=[(X_tr[va], y_tr[va])], verbose=False)
                    p = clf.predict_proba(X_tr[va])[:, 1]
                    aucs.append(roc_auc_score(y_tr[va], p))
                    bi = getattr(clf, "best_iteration", None)
                    iters.append(int(bi) + 1 if bi is not None else 400)
                except Exception:
                    continue
            if aucs:
                s = float(np.mean(aucs))
                if s > best_score:
                    best_score = s
                    best = (cfg, spw, int(np.clip(np.median(iters) * 1.1, 50, 800)))

    best_cfg, best_spw, best_iter = best
    clf = _make(best_cfg, best_spw, best_iter)
    clf.fit(X_tr, y_tr)
    raw_te = clf.predict_proba(X_te)[:, 1].astype(np.float32)

    # monotonic isotonic calibration on inner OOF preds (AUC preserved exactly)
    if inner:
        oof_p, oof_y = [], []
        for tr, va in inner:
            if len(np.unique(y_tr[tr])) < 2:
                continue
            c = _make(best_cfg, best_spw, best_iter)
            c.fit(X_tr[tr], y_tr[tr])
            oof_p.append(c.predict_proba(X_tr[va])[:, 1])
            oof_y.append(y_tr[va])
        if oof_p:
            oof_p = np.concatenate(oof_p)
            oof_y = np.concatenate(oof_y)
            if len(np.unique(oof_y)) == 2:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(oof_p, oof_y)
                cal = iso.predict(raw_te).astype(np.float32)
                # order-preserving epsilon: break isotonic plateaus in the raw
                # ranking so calibration never lowers AUC (monotone in raw_te).
                cal = cal + np.float32(1e-6) * raw_te
                if np.unique(cal).size > 1:
                    return cal
    return raw_te


try:
    import torch
    import torch.nn as nn

    class EdgeMLP(nn.Module):
        """MLP decoder over [phage_emb ‖ host_emb ‖ edge_features].

        For the features-only baseline the embeddings are empty and it is a plain
        MLP over edge features; the GNN reuses it as the decoder.
        """
        def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False
    EdgeMLP = None  # type: ignore


def fit_predict_mlp(X_tr, y_tr, X_te, seed: int, epochs: int = 300,
                    patience: int = 25, lr: float = 1e-3, val_frac: float = 0.15,
                    hidden: int = 128, dropout: float = 0.3) -> np.ndarray:
    assert HAS_TORCH, "torch unavailable"
    import torch
    from copy import deepcopy
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr = np.asarray(X_tr, np.float32)
    ytr = np.asarray(y_tr, np.float32)
    order = rng.permutation(len(Xtr))
    n_val = max(1, int(len(order) * val_frac))
    vi, ti = order[:n_val], order[n_val:]
    dev = torch.device("cpu")
    model = EdgeMLP(Xtr.shape[1], hidden=hidden, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    # class imbalance handling
    pos_w = float((ytr[ti] == 0).sum()) / max(1.0, float((ytr[ti] == 1).sum()))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    Xt = torch.tensor(Xtr[ti]); yt = torch.tensor(ytr[ti])
    Xv = torch.tensor(Xtr[vi]); yv = torch.tensor(ytr[vi])
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xt), yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xv), yv))
        if vloss < best - 1e-4:
            best, best_state, bad = vloss, deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(np.asarray(X_te, np.float32)))).numpy()
    return p.astype(np.float32)


# --- parallel fold execution -------------------------------------------------
# Big arrays are shared into workers once via an initializer (not per task) to
# keep memory and IPC low.
_B: dict = {}


def _gbm_init(X, y, model_fn, seed, threads, groups):
    import inspect
    limit_threads(threads)
    pass_groups = "groups_tr" in inspect.signature(model_fn).parameters
    _B.update(X=X, y=y, model_fn=model_fn, seed=seed, groups=groups,
              pass_groups=pass_groups)


def _gbm_worker(item):
    name, tr, te = item
    X, y = _B["X"], _B["y"]
    scaler = StandardScaler().fit(X[tr])
    Xtr = np.nan_to_num(scaler.transform(X[tr]), nan=0.0).astype(np.float32)
    Xte = np.nan_to_num(scaler.transform(X[te]), nan=0.0).astype(np.float32)
    if _B.get("pass_groups") and _B.get("groups") is not None:
        gtr = _B["groups"][tr]
        probs = _B["model_fn"](Xtr, y[tr], Xte, _B["seed"], groups_tr=gtr)
    else:
        probs = _B["model_fn"](Xtr, y[tr], Xte, _B["seed"])
    return name, te, probs.astype(np.float32)


def run_grouped_cv(df, X, folds, model_fn, seed: int,
                   cluster_col: str = "host_genus", cfg: dict | None = None,
                   inner_group_col: str | None = None) -> dict:
    """Run a model across folds with fold-internal scaling. Returns metrics +
    pooled predictions + per-fold AUC series (indexed by fold name).

    Folds run in up to compute.n_jobs (<=10) worker processes; each worker is
    pinned to threads_per_job BLAS threads to avoid oversubscription.

    `inner_group_col`: when set and the model accepts a `groups_tr` argument,
    the training-row labels of that column are passed in so the model can run
    group-aware nested model selection (leakage-free).
    """
    X = np.asarray(X, dtype=np.float32)
    y = df["label"].to_numpy().astype(np.float32)
    folds = list(folds)
    items = [(f.name, f.train_idx, f.test_idx) for f in folds]
    cfg = cfg or {}
    n_jobs = resolve_n_jobs(cfg, len(items))
    threads = int(cfg.get("compute", {}).get("threads_per_job", 1))
    groups = None
    if inner_group_col is not None and inner_group_col in df.columns:
        groups = df[inner_group_col].to_numpy()

    if n_jobs <= 1:
        _gbm_init(X, y, model_fn, seed, threads, groups)
        results = [_gbm_worker(it) for it in items]
    else:
        # 'spawn' avoids fork-after-threads deadlocks and honours the thread-cap
        # env set at entry-point import time (no BLAS oversubscription).
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx,
                                 initializer=_gbm_init,
                                 initargs=(X, y, model_fn, seed, threads,
                                           groups)) as ex:
            results = list(ex.map(_gbm_worker, items))

    by_name = {name: (te, probs) for name, te, probs in results}
    per_fold, fold_aucs, fold_clusters = [], {}, {}
    pooled_y, pooled_p = [], []
    for fold in folds:
        te, probs = by_name[fold.name]
        m = binary_metrics(y[te], probs)
        per_fold.append(m)
        if m is not None:
            fold_aucs[fold.name] = m["roc_auc"]
            try:
                fold_clusters[fold.name] = df.iloc[te][cluster_col].mode().iloc[0]
            except Exception:
                fold_clusters[fold.name] = "na"
        pooled_y.append(y[te]); pooled_p.append(probs)
    pooled_y = np.concatenate(pooled_y) if pooled_y else np.array([])
    pooled_p = np.concatenate(pooled_p) if pooled_p else np.array([])
    agg = aggregate_folds(per_fold, pooled_y, pooled_p)
    return {
        "agg": agg,
        "fold_aucs": fold_aucs,
        "fold_clusters": fold_clusters,
        "pooled_y": pooled_y,
        "pooled_p": pooled_p,
    }
