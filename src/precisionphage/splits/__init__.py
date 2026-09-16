"""Leakage-safe cross-validation splits.

Two families:
  * Grouped leave-one-group-out over a categorical column (host_species -> LOSO,
    host_genus -> LOGO, study -> cross-study). No genome needed.
  * Similarity-cluster splits (phage/host genome clusters) for homology-aware
    generalization, added once genomes are available.

The invariant: a fold exposes train/test row indices only; ALL preprocessing
(scaling, PCA, graph construction) must be fit on train indices downstream.
"""
from .grouped import Fold, leave_one_group_out, cross_study_folds
from .cluster import (
    audit_taxonomic_leakage,
    build_clusters,
    combined_unseen_folds,
    mash_distance,
    minhash_sketch,
    sketch_entities,
)
from .cache import load_or_build_clusters

__all__ = [
    "Fold", "leave_one_group_out", "cross_study_folds",
    "audit_taxonomic_leakage", "build_clusters", "combined_unseen_folds",
    "mash_distance", "minhash_sketch", "sketch_entities",
    "load_or_build_clusters",
]
