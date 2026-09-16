#!/usr/bin/env python3
"""Evaluate the already frozen predictions on held-out ColoColi outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "external" / "gaborieau_coli_phage_interactions_2023"
OUT = ROOT / "data" / "results_v2" / "external_cocktail_validation"
PROTOCOL = ROOT / "EXTERNAL_COCKTAIL_VALIDATION_PROTOCOL.md"
PREDICTIONS = OUT / "frozen_pair_predictions.csv"
MANIFEST = OUT / "frozen_prediction_manifest.json"
OUTCOMES = UPSTREAM / "dev" / "cocktails" / "data" / "cml_cocktails.csv"
COMPOSITION = UPSTREAM / "dev" / "cocktails" / "data" / "cocktails_composition.csv"
DISTANCES = (UPSTREAM / "dev" / "cocktails" / "data" /
             "picard+test_collection_phylogenetic_distances.tsv")
THRESHOLD = 1e-4
SEED = 42030
OUTCOME_TO_TEST_ALIASES = {"BCH857": "BCH857-72"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    result = 0.0
    for b in range(bins):
        keep = ids == b
        if keep.any():
            result += keep.mean() * abs(float(y[keep].mean()) - float(p[keep].mean()))
    return float(result)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }


def host_clusters(names: list[str], distance: pd.DataFrame) -> np.ndarray:
    available = [n for n in names if n in distance.index]
    labels: dict[str, int] = {}
    if available:
        square = distance.loc[available, available].to_numpy(float)
        _, component = connected_components(csr_matrix(square <= THRESHOLD), directed=False)
        labels.update(dict(zip(available, component.astype(int))))
    next_id = max(labels.values(), default=-1) + 1
    for name in names:
        if name not in labels:
            labels[name] = next_id
            next_id += 1
    return np.asarray([labels[n] for n in names], int)


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, clusters: np.ndarray,
                      second: np.ndarray | None = None, n_valid: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    first_unique = np.unique(clusters)
    second_unique = np.unique(second) if second is not None else None
    aucs, aps = [], []
    attempts = 0
    while len(aucs) < n_valid and attempts < n_valid * 50:
        attempts += 1
        sampled_first = rng.choice(first_unique, len(first_unique), replace=True)
        first_count = {g: int((sampled_first == g).sum()) for g in first_unique}
        weight = np.asarray([first_count[g] for g in clusters], float)
        if second_unique is not None:
            sampled_second = rng.choice(second_unique, len(second_unique), replace=True)
            second_count = {g: int((sampled_second == g).sum()) for g in second_unique}
            weight *= np.asarray([second_count[g] for g in second], float)
        keep = weight > 0
        if np.unique(y[keep]).size < 2:
            continue
        aucs.append(roc_auc_score(y, p, sample_weight=weight))
        aps.append(average_precision_score(y, p, sample_weight=weight))
    if len(aucs) != n_valid:
        raise RuntimeError(f"Only {len(aucs)} valid bootstrap replicates")
    return {
        "n_valid": n_valid,
        "roc_auc_95_ci": [float(np.percentile(aucs, 2.5)),
                           float(np.percentile(aucs, 97.5))],
        "average_precision_95_ci": [float(np.percentile(aps, 2.5)),
                                     float(np.percentile(aps, 97.5))],
    }


def main() -> None:
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if frozen["status"] != "predictions_frozen_before_external_outcome_read":
        raise RuntimeError("Prediction manifest is not frozen")
    checks = frozen["checksums"]
    if sha256(PROTOCOL) != checks["protocol"] or sha256(PREDICTIONS) != checks["frozen_pair_predictions"]:
        raise RuntimeError("Protocol or frozen predictions changed after freezing")

    pred = pd.read_csv(PREDICTIONS)
    outcomes = pd.read_csv(OUTCOMES, sep=";")
    composition = pd.read_csv(COMPOSITION, sep=";")
    distance = pd.read_csv(DISTANCES, sep="\t", index_col=0)
    if len(outcomes) != 100 or len(composition) != 100:
        raise RuntimeError("Unexpected held-out cohort size")
    if not outcomes["bacteria"].equals(composition["bacteria"]):
        raise RuntimeError("Outcome and composition host order differs")

    cocktail_rows = []
    pair_rows = []
    for row in outcomes.itertuples(index=False):
        host = str(row.bacteria)
        prediction_host = OUTCOME_TO_TEST_ALIASES.get(host, host)
        phages = [str(row.p1), str(row.p2), str(row.p3)]
        mlc = [float(getattr(row, "p1")), float(getattr(row, "p2")), float(getattr(row, "p3"))]
        # In cml_cocktails.csv p1/p2/p3 are MLC values; phage identities are in composition.
        comp = composition.loc[composition["bacteria"] == host].iloc[0]
        phages = [str(comp["p1"]), str(comp["p2"]), str(comp["p3"])]
        selected = pred[(pred["bacteria"] == prediction_host) &
                        pred["phage"].isin(phages)].copy()
        selected = selected.set_index("phage").loc[phages].reset_index()
        score = selected["score_ensemble"].to_numpy(float)
        prevalence = selected["training_prevalence"].to_numpy(float)
        for phage, outcome, (_, values) in zip(phages, mlc, selected.iterrows()):
            pair_rows.append({
                "bacteria": prediction_host, "phage": phage, "mlc": outcome,
                "productive": int(outcome > 0),
                "score_ensemble": float(values["score_ensemble"]),
                "score_logistic": float(values["prob_logistic_raw"]),
                "score_knn": float(values["prob_knn_raw"]),
                "training_prevalence": float(values["training_prevalence"]),
                "phage_genus": str(values["Genus"]),
            })
        cocktail_mlc = float(getattr(row, "_13")) if hasattr(row, "_13") else float(row[-2])
        # Namedtuple sanitization is brittle for '+' columns; use DataFrame lookup.
        cocktail_mlc = float(outcomes.loc[outcomes["bacteria"] == host, "p1+p2+p3"].iloc[0])
        cocktail_rows.append({
            "bacteria": prediction_host,
            "cocktail_mlc": cocktail_mlc,
            "productive": int(cocktail_mlc > 0),
            "score_independence": float(1.0 - np.prod(1.0 - score)),
            "score_max": float(np.max(score)),
            "score_generalist": float(1.0 - np.prod(1.0 - prevalence)),
            "predicted_constituents": "|".join(phages),
        })

    cocktails = pd.DataFrame(cocktail_rows)
    pairs = pd.DataFrame(pair_rows)
    c_clusters = host_clusters(cocktails["bacteria"].tolist(), distance)
    p_clusters = host_clusters(pairs["bacteria"].tolist(), distance)

    result = {
        "protocol": "EXTERNAL_COCKTAIL_VALIDATION_PROTOCOL.md",
        "frozen_prediction_sha256": sha256(PREDICTIONS),
        "external_outcome_sha256": sha256(OUTCOMES),
        "external_composition_sha256": sha256(COMPOSITION),
        "primary_cocktail": {
            "score_independence": metrics(cocktails["productive"].to_numpy(),
                                          cocktails["score_independence"].to_numpy()),
            "cluster_bootstrap": cluster_bootstrap(
                cocktails["productive"].to_numpy(),
                cocktails["score_independence"].to_numpy(), c_clusters),
            "spearman_with_mlc": {
                "rho": float(spearmanr(cocktails["score_independence"],
                                       cocktails["cocktail_mlc"]).statistic),
                "p_rowwise_descriptive": float(spearmanr(
                    cocktails["score_independence"], cocktails["cocktail_mlc"]).pvalue),
            },
        },
        "secondary_cocktail": {
            "score_max": metrics(cocktails["productive"].to_numpy(),
                                 cocktails["score_max"].to_numpy()),
            "score_generalist": metrics(cocktails["productive"].to_numpy(),
                                        cocktails["score_generalist"].to_numpy()),
        },
        "secondary_selected_constituent_pairs": {
            "ensemble": metrics(pairs["productive"].to_numpy(),
                                pairs["score_ensemble"].to_numpy()),
            "two_way_cluster_bootstrap": cluster_bootstrap(
                pairs["productive"].to_numpy(), pairs["score_ensemble"].to_numpy(),
                p_clusters, pairs["phage_genus"].to_numpy()),
            "logistic": metrics(pairs["productive"].to_numpy(),
                                pairs["score_logistic"].to_numpy()),
            "knn": metrics(pairs["productive"].to_numpy(),
                           pairs["score_knn"].to_numpy()),
            "training_prevalence": metrics(pairs["productive"].to_numpy(),
                                           pairs["training_prevalence"].to_numpy()),
        },
        "notes": [
            "The primary cocktail score was frozen before held-out outcomes were read.",
            "Selected-constituent pair analysis is secondary because pairs were selected by the published recommender.",
            "This validates a fixed Escherichia phage bank on unseen hosts, not unseen-phage or Staphylococcus transfer.",
        ],
    }
    cocktails.to_csv(OUT / "heldout_cocktail_predictions_and_outcomes.csv", index=False)
    pairs.to_csv(OUT / "heldout_selected_pair_predictions_and_outcomes.csv", index=False)
    (OUT / "external_cocktail_validation_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
