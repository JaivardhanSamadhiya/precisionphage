#!/usr/bin/env python3
"""Frozen NCBI_HR -> StaphStudy sequence-covered external validation.

The protocol in EXTERNAL_VALIDATION_PROTOCOL.md was written before this script
was used to generate held-out predictions.  Test labels are never passed to a
fit, calibration, feature-selection, or threshold-selection operation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precisionphage.data import GenomeIndex, load_interactions  # noqa: E402
from precisionphage.data.naming import clean_name  # noqa: E402
from precisionphage.eval import binary_metrics, calibration_curve_ece  # noqa: E402
from precisionphage.features.assembly import VHI_FEATS, build_covered_dataset  # noqa: E402
from precisionphage.features.genomic import (  # noqa: E402
    build_node_features, edge_features_from_spectra, kmer_spectrum,
)
from precisionphage.features.seqmatch import (  # noqa: E402
    compute_pair_features, pair_feature_cols,
)
from precisionphage.splits.cluster import (  # noqa: E402
    build_clusters, mash_distance, sketch_entities,
)
from precisionphage.utils import (  # noqa: E402
    ensure_dirs, get_logger, limit_threads, load_config, set_determinism,
)

log = get_logger("frozen_cross_source")

EXTERNAL_ROOT = ROOT / "external" / "upstream_vhip_tool" / "example"
EXTERNAL_PHAGE_ALIASES = (ROOT / "data" / "external_validation" /
                          "staphstudy" / "phages_by_study_id")
PAIR_CACHE = ROOT / "data" / "interim_v2" / "external_staph_seq_pair_features.csv"
PAIR_META = ROOT / "data" / "interim_v2" / "external_staph_seq_pair_features.meta.json"
RESULT_DIR = ROOT / "data" / "results_v2"
MODEL_DIR = RESULT_DIR / "external_validation_models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _genome_bundle_digest(names: list[str], index: GenomeIndex) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = index.resolve(name)
        if path is None:
            raise FileNotFoundError(name)
        digest.update(name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _external_pair_metadata(df: pd.DataFrame, cfg: dict,
                            pidx: GenomeIndex, hidx: GenomeIndex) -> dict:
    phages = sorted(df["phage"].unique())
    hosts = sorted(df["host"].unique())
    return {
        "schema": 1,
        "pairs": int(len(df)),
        "pair_keys_sha256": hashlib.sha256(
            "\n".join(f"{p}\t{h}" for p, h in
                      df[["phage", "host"]].itertuples(index=False, name=None))
            .encode("utf-8")
        ).hexdigest(),
        "phage_bundle_sha256": _genome_bundle_digest(phages, pidx),
        "host_bundle_sha256": _genome_bundle_digest(hosts, hidx),
        "feature_columns": pair_feature_cols(cfg),
        "feature_config": {
            key: cfg["features"][key]
            for key in ("homology_ks", "crispr_repeat_k", "crispr_match_k",
                        "use_protein_features", "protein_k", "protein_min_pep")
        },
    }


def _stage_external_phage_aliases(required_phages: set[str]) -> Path:
    """Materialize the published assembly-accession -> study-ID crosswalk."""
    mapping_path = (ROOT / "external" / "upstream_vhip" / "data" / "other" /
                    "StaphStudy_virusnames.tsv")
    mapping = pd.read_csv(mapping_path, sep="\t")
    source_dir = EXTERNAL_ROOT / "virus_genomes"
    EXTERNAL_PHAGE_ALIASES.mkdir(parents=True, exist_ok=True)
    staged = set()
    for row in mapping.itertuples(index=False):
        phage = clean_name(str(row.phagename))
        if phage not in required_phages:
            continue
        assembly = str(row.filename).split("_")[0:2]
        prefix = "_".join(assembly)
        matches = sorted(source_dir.glob(f"{prefix}*.fasta"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one FASTA for {phage}/{prefix}, got {matches}")
        target = EXTERNAL_PHAGE_ALIASES / f"{phage}.fasta"
        if not target.exists() or _sha256(target) != _sha256(matches[0]):
            shutil.copyfile(matches[0], target)
        staged.add(phage)
    missing = required_phages - staged
    if missing:
        raise RuntimeError(f"No published assembly mapping for phages: {sorted(missing)}")
    return EXTERNAL_PHAGE_ALIASES


def _load_or_compute_external_pair_features(df: pd.DataFrame, cfg: dict,
                                            pidx: GenomeIndex,
                                            hidx: GenomeIndex) -> pd.DataFrame:
    expected = _external_pair_metadata(df, cfg, pidx, hidx)
    if PAIR_CACHE.exists() and PAIR_META.exists():
        observed = json.loads(PAIR_META.read_text(encoding="utf-8"))
        if observed == expected:
            cached = pd.read_csv(PAIR_CACHE)
            if len(cached) == len(df):
                log.info("Loaded content-matched external pair-feature cache")
                return cached

    computed = compute_pair_features(
        df[["phage", "host"]].drop_duplicates(), cfg,
        n_jobs=cfg["compute"]["n_jobs"],
    )
    computed.to_csv(PAIR_CACHE, index=False)
    PAIR_META.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    log.info("Wrote %s and provenance metadata", PAIR_CACHE)
    return computed


def _assemble_external(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, list[str],
                                           GenomeIndex, GenomeIndex]:
    ds = load_interactions(cfg)
    ext = ds.df.loc[ds.df["study"] == "StaphStudy"].copy().reset_index(drop=True)
    assert len(ext) == 1053 and ext["phage"].nunique() == 39 and ext["host"].nunique() == 27

    ext_cfg = copy.deepcopy(cfg)
    ext_cfg["paths"]["phage_fasta_dir"] = _stage_external_phage_aliases(
        set(ext["phage"].unique())
    )
    ext_cfg["paths"]["host_fasta_dir"] = EXTERNAL_ROOT / "host_genomes"
    pidx = GenomeIndex([ext_cfg["paths"]["phage_fasta_dir"]])
    hidx = GenomeIndex([ext_cfg["paths"]["host_fasta_dir"]])
    pcov = pidx.coverage(sorted(ext["phage"].unique()))
    hcov = hidx.coverage(sorted(ext["host"].unique()))
    if pcov["resolved"] != pcov["n"] or hcov["resolved"] != hcov["n"]:
        raise RuntimeError(f"Incomplete external genome coverage: phage={pcov}, host={hcov}")

    # The upstream example table must be the exact frozen table, preventing a
    # silent join to a differently processed release.
    upstream_table = EXTERNAL_ROOT / "ml_input.csv"
    frozen_table = ROOT / "data" / "raw" / "VirusHostInter.csv"
    if _sha256(upstream_table) != _sha256(frozen_table):
        raise RuntimeError("Upstream VHIP example and frozen interaction table differ")

    phages = sorted(ext["phage"].unique())
    hosts = sorted(ext["host"].unique())
    p2i = {name: i for i, name in enumerate(phages)}
    h2i = {name: i for i, name in enumerate(hosts)}
    ext["pidx"] = ext["phage"].map(p2i).astype(int)
    ext["hidx"] = ext["host"].map(h2i).astype(int)

    P, _ = build_node_features(
        phages, pidx, k=cfg["features"]["kmer_k"],
        use_codon=cfg["features"]["use_codon"],
        use_dinuc=cfg["features"]["use_dinuc"],
        n_workers=cfg["features"]["n_workers"],
    )
    H, _ = build_node_features(
        hosts, hidx, k=cfg["features"]["kmer_k"],
        use_codon=cfg["features"]["use_codon"],
        use_dinuc=cfg["features"]["use_dinuc"],
        n_workers=cfg["features"]["n_workers"],
    )

    kdim = int(kmer_spectrum("ACGT" * cfg["features"]["kmer_k"],
                             cfg["features"]["kmer_k"]).shape[0])
    recomputed = np.zeros((len(ext), 4), dtype=np.float32)
    pi = ext["pidx"].to_numpy()
    hi = ext["hidx"].to_numpy()
    for row in range(len(ext)):
        recomputed[row] = edge_features_from_spectra(P[pi[row], :kdim],
                                                      H[hi[row], :kdim])
    edge_cols = ["cos_dist", "l1", "pearson", "jaccard"] + VHI_FEATS
    edges = np.hstack([recomputed, ext[VHI_FEATS].to_numpy(np.float32)])

    pair = _load_or_compute_external_pair_features(ext, ext_cfg, pidx, hidx)
    extra_cols = pair_feature_cols(ext_cfg)
    merged = ext[["phage", "host"]].merge(pair, on=["phage", "host"], how="left")
    if merged[extra_cols].isna().any().any():
        raise RuntimeError("Missing external sequence-pair features after merge")
    edges = np.hstack([edges, merged[extra_cols].to_numpy(np.float32)])
    edge_cols.extend(extra_cols)
    X = np.hstack([P[pi], H[hi], edges]).astype(np.float32)

    feature_names = ([f"phage_node_{i}" for i in range(P.shape[1])] +
                     [f"host_node_{i}" for i in range(H.shape[1])] + edge_cols)
    return ext, X, feature_names, pidx, hidx


def _new_model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        tree_method="hist", n_jobs=10, eval_metric="logloss", verbosity=0,
    )


def _fit_predict(name: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, seed: int) -> np.ndarray:
    scaler = StandardScaler().fit(X_train)
    train_scaled = np.nan_to_num(scaler.transform(X_train), nan=0.0).astype(np.float32)
    test_scaled = np.nan_to_num(scaler.transform(X_test), nan=0.0).astype(np.float32)
    model = _new_model(seed)
    model.fit(train_scaled, y_train)
    probability = model.predict_proba(test_scaled)[:, 1].astype(np.float32)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, MODEL_DIR / f"{name}.joblib")
    return probability


def _two_way_cluster_ci(y: np.ndarray, probability: np.ndarray,
                        phage_cluster: np.ndarray, host_cluster: np.ndarray,
                        seed: int, n_valid: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    puniq = np.unique(phage_cluster)
    huniq = np.unique(host_cluster)
    aucs, aps = [], []
    attempts = 0
    while len(aucs) < n_valid and attempts < n_valid * 20:
        attempts += 1
        ps = rng.choice(puniq, size=len(puniq), replace=True)
        hs = rng.choice(huniq, size=len(huniq), replace=True)
        pc = {value: int((ps == value).sum()) for value in puniq}
        hc = {value: int((hs == value).sum()) for value in huniq}
        weights = np.array([pc[p] * hc[h]
                            for p, h in zip(phage_cluster, host_cluster)], dtype=float)
        keep = weights > 0
        if len(np.unique(y[keep])) < 2:
            continue
        aucs.append(roc_auc_score(y, probability, sample_weight=weights))
        aps.append(average_precision_score(y, probability, sample_weight=weights))
    if len(aucs) < n_valid:
        raise RuntimeError(f"Only {len(aucs)} valid two-way bootstrap replicates")
    return {
        "method": "two-way sequence-cluster bootstrap",
        "n_valid": n_valid,
        "roc_auc": {"lo": float(np.percentile(aucs, 2.5)),
                    "hi": float(np.percentile(aucs, 97.5))},
        "pr_auc": {"lo": float(np.percentile(aps, 2.5)),
                   "hi": float(np.percentile(aps, 97.5))},
    }


def _group_auc(df: pd.DataFrame, group: str, probability: np.ndarray) -> dict:
    values = []
    for _, idx in df.groupby(group).groups.items():
        rows = np.asarray(list(idx), dtype=int)
        y = df.iloc[rows]["label"].to_numpy()
        if len(np.unique(y)) == 2:
            values.append(float(roc_auc_score(y, probability[rows])))
    a = np.asarray(values, dtype=float)
    return {
        "n_evaluable": int(len(a)),
        "n_total": int(df[group].nunique()),
        "mean": float(a.mean()) if len(a) else None,
        "median": float(np.median(a)) if len(a) else None,
        "min": float(a.min()) if len(a) else None,
        "max": float(a.max()) if len(a) else None,
    }


def _subset_metrics(df: pd.DataFrame, probability: np.ndarray,
                    mask: np.ndarray) -> dict | None:
    rows = np.where(mask)[0]
    if not len(rows):
        return None
    result = binary_metrics(df.iloc[rows]["label"].to_numpy(), probability[rows])
    if result is not None:
        result["prevalence"] = float(df.iloc[rows]["label"].mean())
    return result


def _nearest_training_distances(training_names: list[str], training_idx: GenomeIndex,
                                external_names: list[str], external_idx: GenomeIndex,
                                cfg: dict) -> tuple[dict[str, float], dict[str, int]]:
    k = int(cfg["splits"]["mash_k"])
    num = int(cfg["splits"]["minhash_num"])
    threshold = float(cfg["splits"]["mash_max_distance"])
    train_sk = sketch_entities(training_names, training_idx, k, num,
                               cfg["features"]["n_workers"])
    ext_sk = sketch_entities(external_names, external_idx, k, num,
                             cfg["features"]["n_workers"])
    nearest = {}
    for name in external_names:
        nearest[name] = min(
            mash_distance(ext_sk[name], train_sk[other], k, num)
            for other in training_names
        )
    ext_clusters = build_clusters(external_names, ext_sk, threshold, k, num)
    return nearest, ext_clusters


def _summarize_model(df: pd.DataFrame, probability: np.ndarray,
                     phage_clusters: dict[str, int], host_clusters: dict[str, int],
                     masks: dict[str, np.ndarray], seed: int) -> dict:
    y = df["label"].to_numpy().astype(int)
    metrics = binary_metrics(y, probability)
    assert metrics is not None
    metrics["prevalence"] = float(y.mean())
    metrics["ece"] = calibration_curve_ece(y, probability, 10)["ece"]
    pcluster = df["phage"].map(phage_clusters).to_numpy()
    hcluster = df["host"].map(host_clusters).to_numpy()
    return {
        "pooled": metrics,
        "ci_95": _two_way_cluster_ci(y, probability, pcluster, hcluster, seed),
        "per_phage_auc": _group_auc(df, "phage", probability),
        "per_host_auc": _group_auc(df, "host", probability),
        "subsets": {name: _subset_metrics(df, probability, mask)
                    for name, mask in masks.items()},
    }


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    seed = int(cfg["seed"])
    set_determinism(seed)
    ensure_dirs(cfg)
    limit_threads(1)
    os.environ["PP_THREADS"] = "10"

    train = build_covered_dataset(cfg)
    if set(train.df["study"]) != {"NCBI_HR"}:
        raise RuntimeError(f"Training coverage unexpectedly spans {set(train.df['study'])}")
    X_train = train.X_flat()
    y_train = train.df["label"].to_numpy().astype(int)

    external, X_external, feature_names, ext_pidx, ext_hidx = _assemble_external(cfg)
    if X_train.shape[1] != X_external.shape[1] or X_train.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Feature schema mismatch: train={X_train.shape}, test={X_external.shape}, "
            f"names={len(feature_names)}"
        )

    # Feature names for the training assembly must be identical in the edge block.
    node_dim = train.P_raw.shape[1]
    expected_names = ([f"phage_node_{i}" for i in range(node_dim)] +
                      [f"host_node_{i}" for i in range(train.H_raw.shape[1])] +
                      train.edge_cols)
    if expected_names != feature_names:
        raise RuntimeError("Training and external feature-column order differs")

    supplied_idx = np.array([name not in VHI_FEATS for name in feature_names])
    vhip_idx = np.array([name in VHI_FEATS for name in feature_names])
    probabilities = {
        "full": _fit_predict("full", X_train, y_train, X_external, seed),
        "sequence_only": _fit_predict(
            "sequence_only", X_train[:, supplied_idx], y_train,
            X_external[:, supplied_idx], seed,
        ),
        "vhip_four_feature": _fit_predict(
            "vhip_four_feature", X_train[:, vhip_idx], y_train,
            X_external[:, vhip_idx], seed,
        ),
    }

    # Sequence-nearness audit between training and test on both entity axes.
    p_nearest, p_clusters = _nearest_training_distances(
        train.phages, train.phage_index, sorted(external["phage"].unique()),
        ext_pidx, cfg,
    )
    h_nearest, h_clusters = _nearest_training_distances(
        train.hosts, train.host_index, sorted(external["host"].unique()),
        ext_hidx, cfg,
    )
    threshold = float(cfg["splits"]["mash_max_distance"])
    train_host_names = set(train.df["host"])
    masks = {
        "host_identifier_unseen": ~external["host"].isin(train_host_names).to_numpy(),
        "both_axes_beyond_mash_threshold": np.array([
            p_nearest[p] > threshold and h_nearest[h] > threshold
            for p, h in external[["phage", "host"]].itertuples(index=False, name=None)
        ]),
    }

    model_results = {
        name: _summarize_model(external, probability, p_clusters, h_clusters,
                               masks, seed + offset)
        for offset, (name, probability) in enumerate(probabilities.items())
    }

    # Distribution-shift diagnostic, calculated without test labels.
    train_min, train_max = np.nanmin(X_train, axis=0), np.nanmax(X_train, axis=0)
    outside = (X_external < train_min) | (X_external > train_max)
    shift = {
        "fraction_of_test_cells_outside_training_range": float(outside.mean()),
        "fraction_of_test_rows_with_any_outside_feature": float(outside.any(axis=1).mean()),
        "median_outside_features_per_row": float(np.median(outside.sum(axis=1))),
        "max_outside_features_per_row": int(outside.sum(axis=1).max()),
    }

    predictions = external[["phage", "host", "label"]].copy()
    for name, probability in probabilities.items():
        predictions[f"prob_{name}"] = probability
    predictions["phage_nearest_training_mash_distance"] = predictions["phage"].map(p_nearest)
    predictions["host_nearest_training_mash_distance"] = predictions["host"].map(h_nearest)
    predictions["host_identifier_seen_in_training"] = predictions["host"].isin(train_host_names)
    predictions.to_csv(RESULT_DIR / "external_staph_predictions.csv", index=False)

    result = {
        "protocol": "EXTERNAL_VALIDATION_PROTOCOL.md",
        "dataset": {
            "train_source": "NCBI_HR",
            "test_source": "StaphStudy",
            "n_train": int(len(train.df)),
            "n_train_positive": int(y_train.sum()),
            "n_train_negative": int((y_train == 0).sum()),
            "n_test": int(len(external)),
            "n_test_positive": int(external["label"].sum()),
            "n_test_negative": int((external["label"] == 0).sum()),
            "n_test_phages": int(external["phage"].nunique()),
            "n_test_hosts": int(external["host"].nunique()),
        },
        "feature_schema": {
            "total_columns": int(len(feature_names)),
            "node_columns": int(2 * node_dim),
            "edge_columns": int(len(train.edge_cols)),
            "edge_column_names": train.edge_cols,
            "full_model_includes_vhip_supplied_columns": VHI_FEATS,
        },
        "overlap_audit": {
            "phage_identifier_overlap": 0,
            "host_identifier_overlap": int(external["host"].isin(train_host_names).groupby(external["host"]).max().sum()),
            "exact_sequence_overlap": {"phage": 0, "host": 0},
            "mash_form_threshold": threshold,
            "phages_at_or_below_threshold": int(sum(v <= threshold for v in p_nearest.values())),
            "hosts_at_or_below_threshold": int(sum(v <= threshold for v in h_nearest.values())),
            "phage_nearest_distance": p_nearest,
            "host_nearest_distance": h_nearest,
            "test_phage_sequence_clusters": int(len(set(p_clusters.values()))),
            "test_host_sequence_clusters": int(len(set(h_clusters.values()))),
        },
        "distribution_shift": shift,
        "models": model_results,
    }
    (RESULT_DIR / "external_staph_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    summary_rows = []
    for name, item in model_results.items():
        pooled = item["pooled"]
        ci = item["ci_95"]
        summary_rows.append({
            "model": name,
            "n": pooled["n"],
            "prevalence": pooled["prevalence"],
            "auroc": pooled["roc_auc"],
            "auroc_ci_lo": ci["roc_auc"]["lo"],
            "auroc_ci_hi": ci["roc_auc"]["hi"],
            "auprc": pooled["pr_auc"],
            "auprc_ci_lo": ci["pr_auc"]["lo"],
            "auprc_ci_hi": ci["pr_auc"]["hi"],
            "brier": pooled["brier"],
            "ece": pooled["ece"],
            "f1_at_0_5": pooled["f1"],
            "mcc_at_0_5": pooled["mcc"],
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULT_DIR / "external_staph_validation_summary.csv", index=False)
    log.info("\n%s", summary.to_string(index=False))
    log.info("Wrote frozen external validation artifacts")


if __name__ == "__main__":
    main()
