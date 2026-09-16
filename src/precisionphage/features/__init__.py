"""Genomic feature computation for nodes (phage/host) and edges (pairs)."""
from .genomic import (
    kmer_spectrum, codon_usage, gc_content, build_node_features,
    edge_features_from_spectra,
)

__all__ = ["kmer_spectrum", "codon_usage", "gc_content", "build_node_features",
           "edge_features_from_spectra"]
