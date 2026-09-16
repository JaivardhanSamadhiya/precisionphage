#!/usr/bin/env python3
"""Freeze ColoColi cocktail predictions without reading external outcomes.

This implements EXTERNAL_COCKTAIL_VALIDATION_PROTOCOL.md.  The four held-out
outcome/composition tables named there are deliberately not opened here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "external" / "gaborieau_coli_phage_interactions_2023"
OUT = ROOT / "data" / "results_v2" / "external_cocktail_validation"
PROTOCOL = ROOT / "EXTERNAL_COCKTAIL_VALIDATION_PROTOCOL.md"

INTERACTIONS = UPSTREAM / "data" / "interactions" / "interaction_matrix.csv"
TRAIN_META = UPSTREAM / "data" / "genomics" / "bacteria" / "picard_collection.csv"
TEST_META = UPSTREAM / "dev" / "cocktails" / "data" / "test_collection.csv"
DISTANCES = (UPSTREAM / "dev" / "cocktails" / "data" /
             "picard+test_collection_phylogenetic_distances.tsv")
PHAGE_META = UPSTREAM / "data" / "genomics" / "phages" / "guelin_collection.csv"

FEATURES = ["Clermont_Phylo", "ST_Warwick", "LPS_type", "O-type", "H-type"]
C_GRID = [0.1, 1.0, 10.0]
K_GRID = [5, 10, 20, 40]
SEED = 42
DISTANCE_GROUP_THRESHOLD = 1e-4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in FEATURES:
        out[col] = out[col].astype("string").fillna("Unknown").replace("", "Unknown")
    return out


def build_groups(distance: pd.DataFrame, names: list[str]) -> np.ndarray:
    square = distance.loc[names, names].to_numpy(float)
    adjacency = csr_matrix(square <= DISTANCE_GROUP_THRESHOLD)
    _, labels = connected_components(adjacency, directed=False)
    return labels.astype(int)


def make_logistic(c_value: float) -> Pipeline:
    pre = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=3), FEATURES)
    ], remainder="drop")
    model = LogisticRegression(
        C=c_value, class_weight="balanced", max_iter=2000,
        solver="liblinear", random_state=SEED,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def constant_or_logistic(train_x: pd.DataFrame, train_y: np.ndarray,
                         test_x: pd.DataFrame, c_value: float) -> np.ndarray:
    if np.unique(train_y).size < 2:
        return np.full(len(test_x), float(np.mean(train_y)), dtype=float)
    pipe = make_logistic(c_value)
    pipe.fit(train_x, train_y)
    return pipe.predict_proba(test_x)[:, 1]


def knn_predict(train_names: list[str], train_y: np.ndarray,
                query_names: list[str], distance: pd.DataFrame,
                k: int) -> np.ndarray:
    prior = float(np.mean(train_y))
    values = []
    available_rows = set(distance.index)
    available_cols = set(distance.columns)
    usable_train = [name for name in train_names if name in available_cols]
    y_lookup = dict(zip(train_names, train_y))
    for query in query_names:
        if query not in available_rows or not usable_train:
            values.append(prior)
            continue
        d = distance.loc[query, usable_train].to_numpy(float)
        finite = np.isfinite(d)
        if not finite.any():
            values.append(prior)
            continue
        names = np.asarray(usable_train, dtype=object)[finite]
        d = d[finite]
        order = np.argsort(d, kind="stable")[:min(k, len(d))]
        selected_names = names[order]
        selected_d = d[order]
        selected_y = np.asarray([y_lookup[name] for name in selected_names], float)
        weights = 1.0 / np.maximum(selected_d, 1e-6)
        values.append(float(np.average(selected_y, weights=weights)))
    return np.asarray(values, dtype=float)


def valid_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float] | None:
    if np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / (len(values) + 1.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(INTERACTIONS, sep=";", index_col=0)
    train_meta = clean_features(pd.read_csv(TRAIN_META, sep=";"))
    test_meta = clean_features(pd.read_csv(TEST_META, sep=";"))
    distance = pd.read_csv(DISTANCES, sep="\t", index_col=0)
    phage_meta = pd.read_csv(PHAGE_META, sep=";")

    train_names = list(matrix.index.astype(str))
    test_names = list(test_meta["bacteria"].astype(str))
    train_meta = train_meta.set_index("bacteria").loc[train_names].reset_index()
    test_meta = test_meta.set_index("bacteria").loc[test_names].reset_index()
    labels = (matrix.fillna(0.0).to_numpy(float) > 0).astype(int)
    observed = ~matrix.isna().to_numpy()

    if len(train_names) != 402 or len(test_names) != 100 or matrix.shape[1] != 96:
        raise RuntimeError(f"Unexpected cohort dimensions: {matrix.shape}, test={len(test_names)}")
    if not PROTOCOL.exists():
        raise RuntimeError("Locked protocol is missing")

    groups = build_groups(distance, train_names)
    splitter = GroupKFold(n_splits=10)
    folds = list(splitter.split(train_meta, groups=groups))
    phages = list(matrix.columns.astype(str))

    cv_rows: list[dict] = []
    logistic_scores: dict[float, list[float]] = {c: [] for c in C_GRID}
    logistic_aucs: dict[float, list[float]] = {c: [] for c in C_GRID}
    knn_scores: dict[int, list[float]] = {k: [] for k in K_GRID}
    knn_aucs: dict[int, list[float]] = {k: [] for k in K_GRID}

    for fold_id, (tr, va) in enumerate(folds):
        tr_names = [train_names[i] for i in tr]
        va_names = [train_names[i] for i in va]
        for j, phage in enumerate(phages):
            tr_obs = observed[tr, j]
            va_obs = observed[va, j]
            tr_idx = tr[tr_obs]
            va_idx = va[va_obs]
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            y_tr = labels[tr_idx, j]
            y_va = labels[va_idx, j]
            for c_value in C_GRID:
                prob = constant_or_logistic(
                    train_meta.iloc[tr_idx], y_tr, train_meta.iloc[va_idx], c_value)
                metric = valid_metrics(y_va, prob)
                if metric is not None:
                    auc, ap = metric
                    logistic_aucs[c_value].append(auc)
                    logistic_scores[c_value].append(ap)
                    cv_rows.append({"fold": fold_id, "phage": phage,
                                    "component": "logistic", "parameter": c_value,
                                    "roc_auc": auc, "average_precision": ap})
            for k in K_GRID:
                prob = knn_predict(
                    [train_names[i] for i in tr_idx], y_tr,
                    [train_names[i] for i in va_idx], distance, k)
                metric = valid_metrics(y_va, prob)
                if metric is not None:
                    auc, ap = metric
                    knn_aucs[k].append(auc)
                    knn_scores[k].append(ap)
                    cv_rows.append({"fold": fold_id, "phage": phage,
                                    "component": "knn", "parameter": k,
                                    "roc_auc": auc, "average_precision": ap})

    best_c = max(C_GRID, key=lambda c: float(np.mean(logistic_scores[c])))
    best_k = max(K_GRID, key=lambda k: float(np.mean(knn_scores[k])))

    predictions: list[dict] = []
    for j, phage in enumerate(phages):
        obs = observed[:, j]
        y = labels[obs, j]
        names = [train_names[i] for i in np.where(obs)[0]]
        logit = constant_or_logistic(train_meta.loc[obs], y, test_meta, best_c)
        knn = knn_predict(names, y, test_names, distance, best_k)
        ensemble = 0.5 * percentile(logit) + 0.5 * percentile(knn)
        for i, host in enumerate(test_names):
            predictions.append({
                "bacteria": host,
                "phage": phage,
                "prob_logistic_raw": float(logit[i]),
                "prob_knn_raw": float(knn[i]),
                "score_ensemble": float(ensemble[i]),
                "training_prevalence": float(np.mean(y)),
            })

    pred = pd.DataFrame(predictions)
    pred = pred.merge(phage_meta[["phage", "Genus", "Phage_host"]],
                      on="phage", how="left", validate="many_to_one")
    if pred[["Genus", "Phage_host"]].isna().any().any():
        raise RuntimeError("Missing phage metadata")

    cv = pd.DataFrame(cv_rows)
    pred_path = OUT / "frozen_pair_predictions.csv"
    cv_path = OUT / "development_grouped_cv.csv"
    pred.to_csv(pred_path, index=False)
    cv.to_csv(cv_path, index=False)

    summary = {
        "status": "predictions_frozen_before_external_outcome_read",
        "n_train_hosts": len(train_names),
        "n_test_hosts": len(test_names),
        "n_phages": len(phages),
        "training_positive_fraction": float(labels[observed].mean()),
        "distance_groups": int(np.unique(groups).size),
        "test_hosts_without_distance_row": sorted(set(test_names) - set(distance.index)),
        "selected": {"logistic_C": best_c, "knn_k": best_k},
        "development_cv": {
            "logistic": {
                str(c): {"macro_average_precision": float(np.mean(logistic_scores[c])),
                         "macro_roc_auc": float(np.mean(logistic_aucs[c])),
                         "n_evaluable_fold_phages": len(logistic_scores[c])}
                for c in C_GRID
            },
            "knn": {
                str(k): {"macro_average_precision": float(np.mean(knn_scores[k])),
                         "macro_roc_auc": float(np.mean(knn_aucs[k])),
                         "n_evaluable_fold_phages": len(knn_scores[k])}
                for k in K_GRID
            },
        },
        "checksums": {
            "protocol": sha256(PROTOCOL),
            "script": sha256(Path(__file__)),
            "interactions": sha256(INTERACTIONS),
            "train_metadata": sha256(TRAIN_META),
            "test_metadata": sha256(TEST_META),
            "distances": sha256(DISTANCES),
            "phage_metadata": sha256(PHAGE_META),
            "frozen_pair_predictions": sha256(pred_path),
            "development_grouped_cv": sha256(cv_path),
        },
    }
    manifest = OUT / "frozen_prediction_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
