#!/usr/bin/env python3
"""Development-only audit of cross-phage-comparable cocktail scores.

This analysis responds to the failed rank-only selector in experiment 31.  It
uses grouped out-of-fold Picard predictions and never reads ColoColi outcomes.
Its purpose is to choose a defensible selector family for a subsequent nested
assessment, not to report a final external estimate.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).with_name("31_nested_de_novo_cocktail_validation.py")
SPEC = importlib.util.spec_from_file_location("nested_selector", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load experiment 31 helpers")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def unbalanced_logistic_predict(x_train, y_train, x_test, c_value):
    if np.unique(y_train).size < 2:
        return np.full(len(x_test), float(np.mean(y_train)))
    model = helper.logistic(c_value)
    model.set_params(model__class_weight=None)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def main() -> None:
    matrix = pd.read_csv(helper.INTERACTIONS, sep=";", index_col=0)
    names = list(matrix.index.astype(str))
    phages = list(matrix.columns.astype(str))
    meta = helper.clean_features(pd.read_csv(helper.HOST_META, sep=";"))
    meta = meta.set_index("bacteria").loc[names].reset_index()
    phage_meta = pd.read_csv(helper.PHAGE_META, sep=";").set_index("phage").loc[phages]
    distance = pd.read_csv(helper.DISTANCES, sep="\t", index_col=0)
    labels = (matrix.fillna(0).to_numpy(float) > 0).astype(int)
    observed = ~matrix.isna().to_numpy()
    groups = helper.groups_from_distance(distance, names)
    rows = []

    for fold, (tr, te) in enumerate(GroupKFold(10).split(meta, groups=groups)):
        raw_log = np.zeros((len(te), len(phages)))
        raw_knn = np.zeros_like(raw_log)
        prevalence = np.zeros(len(phages))
        for j in range(len(phages)):
            tr_idx = tr[observed[tr, j]]
            y_tr = labels[tr_idx, j]
            prevalence[j] = y_tr.mean()
            raw_log[:, j] = unbalanced_logistic_predict(
                meta.iloc[tr_idx], y_tr, meta.iloc[te], 0.1)
            raw_knn[:, j] = helper.knn_predict(
                [names[i] for i in tr_idx], y_tr, [names[i] for i in te],
                distance, 40)

        variants = {
            "prevalence": np.tile(prevalence, (len(te), 1)),
            "unbalanced_logistic": raw_log,
            "knn": raw_knn,
            "raw_mean": 0.5 * raw_log + 0.5 * raw_knn,
        }
        # Shrink the personalized estimate toward empirical breadth.  Alpha is
        # audited here and must be selected inside training data in the final
        # nested assessment.
        raw_mean = variants["raw_mean"]
        for alpha in (0.25, 0.5, 0.75):
            variants[f"raw_mean_alpha_{alpha}"] = (
                alpha * raw_mean + (1.0 - alpha) * prevalence[None, :])

        for local_i, global_i in enumerate(te):
            for variant, score in variants.items():
                chosen = helper.select_three(
                    pd.Series(score[local_i], index=phages), phage_meta)
                idx = [phages.index(p) for p in chosen]
                if not observed[global_i, idx].all():
                    continue
                yy = labels[global_i, idx]
                rows.append({
                    "fold": fold, "group": int(groups[global_i]),
                    "bacteria": names[global_i], "variant": variant,
                    "selected": "|".join(chosen), "success": int(yy.any()),
                    "positive_fraction": float(yy.mean()),
                })

    result = pd.DataFrame(rows)
    summary = result.groupby("variant").agg(
        n=("success", "size"), success=("success", "mean"),
        constituent_positive_fraction=("positive_fraction", "mean"),
    ).sort_values(["success", "constituent_positive_fraction"], ascending=False)
    out = helper.OUT
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "development_selector_audit.csv", index=False)
    (out / "development_selector_audit_summary.json").write_text(
        json.dumps(summary.reset_index().to_dict("records"), indent=2),
        encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
