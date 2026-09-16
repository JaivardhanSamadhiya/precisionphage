#!/usr/bin/env python3
"""Nested, group-held-out validation of de novo three-phage selection.

The Picard interaction matrix is development data.  For every outer fold this
script learns only from the remaining hosts, chooses three phages for each
held-out host, and then evaluates those choices against the held-out matrix.
Hyperparameters are selected again inside each outer training partition.
"""
from __future__ import annotations

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
INTERACTIONS = UPSTREAM / "data" / "interactions" / "interaction_matrix.csv"
HOST_META = UPSTREAM / "data" / "genomics" / "bacteria" / "picard_collection.csv"
PHAGE_META = UPSTREAM / "data" / "genomics" / "phages" / "guelin_collection.csv"
DISTANCES = (UPSTREAM / "dev" / "cocktails" / "data" /
             "picard+test_collection_phylogenetic_distances.tsv")

FEATURES = ["Clermont_Phylo", "ST_Warwick", "LPS_type", "O-type", "H-type"]
C_GRID = [0.1, 1.0, 10.0]
K_GRID = [5, 10, 20, 40]
SEED = 31042
GROUP_THRESHOLD = 1e-4


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in FEATURES:
        out[col] = out[col].astype("string").fillna("Unknown").replace("", "Unknown")
    return out


def groups_from_distance(distance: pd.DataFrame, names: list[str]) -> np.ndarray:
    square = distance.loc[names, names].to_numpy(float)
    _, labels = connected_components(
        csr_matrix(square <= GROUP_THRESHOLD), directed=False)
    return labels.astype(int)


def logistic(c_value: float) -> Pipeline:
    pre = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=3), FEATURES)
    ], remainder="drop")
    # Balanced fitting improves ranking for rare-host-range phages.  Scores are
    # rank-normalized below and are never interpreted as calibrated probability.
    model = LogisticRegression(
        C=c_value, class_weight="balanced", max_iter=2000,
        solver="liblinear", random_state=SEED,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def logistic_predict(x_train: pd.DataFrame, y_train: np.ndarray,
                     x_test: pd.DataFrame, c_value: float) -> np.ndarray:
    if np.unique(y_train).size < 2:
        return np.full(len(x_test), float(np.mean(y_train)))
    model = logistic(c_value)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def knn_predict(train_names: list[str], y_train: np.ndarray,
                query_names: list[str], distance: pd.DataFrame, k: int) -> np.ndarray:
    prior = float(np.mean(y_train))
    lookup = dict(zip(train_names, y_train))
    usable = [name for name in train_names if name in distance.columns]
    result = []
    for query in query_names:
        if query not in distance.index or not usable:
            result.append(prior)
            continue
        d = distance.loc[query, usable].to_numpy(float)
        finite = np.isfinite(d)
        if not finite.any():
            result.append(prior)
            continue
        names = np.asarray(usable, object)[finite]
        d = d[finite]
        order = np.argsort(d, kind="stable")[:min(k, len(d))]
        yy = np.asarray([lookup[name] for name in names[order]], float)
        weights = 1.0 / np.maximum(d[order], 1e-6)
        result.append(float(np.average(yy, weights=weights)))
    return np.asarray(result)


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / (len(values) + 1.0)


def select_three(scores: pd.Series, metadata: pd.DataFrame) -> list[str]:
    """Published diversity rule: forbid same-genus AND same-host pairs."""
    selected: list[str] = []
    for phage in scores.sort_values(ascending=False, kind="stable").index:
        candidate = metadata.loc[phage]
        conflicts = [
            metadata.loc[chosen, "Genus"] == candidate["Genus"]
            and metadata.loc[chosen, "Phage_host"] == candidate["Phage_host"]
            for chosen in selected
        ]
        if not any(conflicts):
            selected.append(phage)
        if len(selected) == 3:
            return selected
    raise RuntimeError("Could not construct a diverse three-phage cocktail")


def tune_inner(train_idx: np.ndarray, matrix: pd.DataFrame, labels: np.ndarray,
               observed: np.ndarray, meta: pd.DataFrame, names: list[str],
               groups: np.ndarray, distance: pd.DataFrame) -> tuple[float, int]:
    group_count = len(np.unique(groups[train_idx]))
    splitter = GroupKFold(n_splits=min(5, group_count))
    log_scores = {c: [] for c in C_GRID}
    knn_scores = {k: [] for k in K_GRID}
    for inner_train_local, inner_val_local in splitter.split(
            train_idx, groups=groups[train_idx]):
        tr = train_idx[inner_train_local]
        va = train_idx[inner_val_local]
        for j in range(matrix.shape[1]):
            tr_idx = tr[observed[tr, j]]
            va_idx = va[observed[va, j]]
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            y_tr, y_va = labels[tr_idx, j], labels[va_idx, j]
            if np.unique(y_va).size < 2:
                continue
            for c_value in C_GRID:
                p = logistic_predict(meta.iloc[tr_idx], y_tr, meta.iloc[va_idx], c_value)
                log_scores[c_value].append(average_precision_score(y_va, p))
            for k in K_GRID:
                p = knn_predict([names[i] for i in tr_idx], y_tr,
                                [names[i] for i in va_idx], distance, k)
                knn_scores[k].append(average_precision_score(y_va, p))
    best_c = max(C_GRID, key=lambda c: float(np.mean(log_scores[c])))
    best_k = max(K_GRID, key=lambda k: float(np.mean(knn_scores[k])))
    return best_c, best_k


def bootstrap_difference(frame: pd.DataFrame, column: str,
                         group_column: str = "group", n: int = 5000) -> list[float]:
    rng = np.random.default_rng(SEED)
    groups = frame[group_column].unique()
    values = []
    for _ in range(n):
        sampled = rng.choice(groups, len(groups), replace=True)
        chunks = [frame.loc[frame[group_column] == group] for group in sampled]
        sample = pd.concat(chunks, ignore_index=True)
        values.append(float(sample[column].mean()))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(INTERACTIONS, sep=";", index_col=0)
    names = list(matrix.index.astype(str))
    phages = list(matrix.columns.astype(str))
    meta = clean_features(pd.read_csv(HOST_META, sep=";"))
    meta = meta.set_index("bacteria").loc[names].reset_index()
    phage_meta = pd.read_csv(PHAGE_META, sep=";").set_index("phage").loc[phages]
    distance = pd.read_csv(DISTANCES, sep="\t", index_col=0)
    labels = (matrix.fillna(0).to_numpy(float) > 0).astype(int)
    mlc = matrix.to_numpy(float)
    observed = ~matrix.isna().to_numpy()
    groups = groups_from_distance(distance, names)
    outer = GroupKFold(n_splits=10)
    rows: list[dict] = []
    fold_settings: list[dict] = []

    for fold, (tr, te) in enumerate(outer.split(meta, groups=groups)):
        best_c, best_k = tune_inner(
            tr, matrix, labels, observed, meta, names, groups, distance)
        fold_settings.append({"fold": fold, "logistic_C": best_c, "knn_k": best_k,
                              "n_train": len(tr), "n_test": len(te)})
        score = np.zeros((len(te), len(phages)), float)
        prevalence = np.zeros(len(phages), float)
        for j in range(len(phages)):
            tr_idx = tr[observed[tr, j]]
            y_tr = labels[tr_idx, j]
            prevalence[j] = float(np.mean(y_tr))
            p_log = logistic_predict(meta.iloc[tr_idx], y_tr, meta.iloc[te], best_c)
            p_knn = knn_predict([names[i] for i in tr_idx], y_tr,
                                [names[i] for i in te], distance, best_k)
            score[:, j] = 0.5 * percentile(p_log) + 0.5 * percentile(p_knn)

        baseline = select_three(pd.Series(prevalence, index=phages), phage_meta)
        for local_i, global_i in enumerate(te):
            designed = select_three(pd.Series(score[local_i], index=phages), phage_meta)
            design_idx = [phages.index(p) for p in designed]
            base_idx = [phages.index(p) for p in baseline]
            if not observed[global_i, design_idx].all() or not observed[global_i, base_idx].all():
                continue
            design_y = labels[global_i, design_idx]
            base_y = labels[global_i, base_idx]
            design_mlc = mlc[global_i, design_idx]
            base_mlc = mlc[global_i, base_idx]
            rows.append({
                "fold": fold, "group": int(groups[global_i]), "bacteria": names[global_i],
                "designed_phages": "|".join(designed),
                "baseline_phages": "|".join(baseline),
                "designed_success": int(design_y.any()),
                "baseline_success": int(base_y.any()),
                "success_difference": int(design_y.any()) - int(base_y.any()),
                "designed_positive_fraction": float(design_y.mean()),
                "baseline_positive_fraction": float(base_y.mean()),
                "positive_fraction_difference": float(design_y.mean() - base_y.mean()),
                "designed_max_mlc": float(design_mlc.max()),
                "baseline_max_mlc": float(base_mlc.max()),
            })

    result = pd.DataFrame(rows)
    summary = {
        "design": "nested 10-fold host-grouped cross-validation",
        "n_hosts_total": len(names),
        "n_distance_components": int(len(np.unique(groups))),
        "n_hosts_with_complete_selected_outcomes": int(len(result)),
        "designed_success": float(result["designed_success"].mean()),
        "baseline_success": float(result["baseline_success"].mean()),
        "success_difference": float(result["success_difference"].mean()),
        "success_difference_cluster_bootstrap_95_ci": bootstrap_difference(
            result, "success_difference"),
        "designed_constituent_positive_fraction": float(
            result["designed_positive_fraction"].mean()),
        "baseline_constituent_positive_fraction": float(
            result["baseline_positive_fraction"].mean()),
        "constituent_positive_fraction_difference": float(
            result["positive_fraction_difference"].mean()),
        "constituent_difference_cluster_bootstrap_95_ci": bootstrap_difference(
            result, "positive_fraction_difference"),
        "designed_mean_max_mlc": float(result["designed_max_mlc"].mean()),
        "baseline_mean_max_mlc": float(result["baseline_max_mlc"].mean()),
        "fold_settings": fold_settings,
        "interpretation": [
            "All selections for a held-out host were made without its interaction labels.",
            "This is internal grouped validation in the Picard matrix, not an external wet-lab test of these newly selected cocktails.",
            "The external ColoColi analysis separately validates locked scoring of experimentally tested cocktails on unseen hosts.",
        ],
    }
    result.to_csv(OUT / "nested_de_novo_cocktail_selections.csv", index=False)
    (OUT / "nested_de_novo_cocktail_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
