"""Homology-aware (sequence-similarity) splits.

Taxonomic names are an unreliable proxy for genomic similarity: two differently
named host "species" can be near-identical genomes, so a leave-one-species-out
split can still place sequence-similar entities in both train and test. To make
generalization claims defensible we cluster genomes by *sequence* similarity
(bottom-k MinHash -> Mash distance -> single-linkage) and hold out whole
clusters. The hardest, most honest regime holds out clusters on BOTH axes
simultaneously (combined unseen phage + unseen host).

All functions are deterministic given a seed and operate on row indices only;
downstream preprocessing must still be fit on the training rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.genomic import _encode
from ..utils import get_logger
from .grouped import Fold

log = get_logger(__name__)


# --- MinHash sketching -------------------------------------------------------

def _mix64(x: np.ndarray) -> np.ndarray:
    """MurmurHash3 64-bit finalizer (vectorized, wraps mod 2**64)."""
    x = x.astype(np.uint64, copy=False)
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xFF51AFD7ED558CCD)
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xC4CEB9FE1A85EC53)
    x = x ^ (x >> np.uint64(33))
    return x


def minhash_sketch(seq: str, k: int = 21, num: int = 256) -> np.ndarray:
    """Bottom-`num` MinHash sketch of a genome's forward-strand k-mers.

    Returns a sorted uint64 array (length <= num). N-containing windows are
    skipped. k <= 31 so the k-mer code fits in int64."""
    if not seq or len(seq) < k:
        return np.empty(0, dtype=np.uint64)
    a = _encode(seq)
    n = a.shape[0]
    valid = a >= 0
    aclip = np.where(valid, a, 0).astype(np.int64)
    m = n - k + 1
    codes = np.zeros(m, dtype=np.int64)
    vw = np.ones(m, dtype=bool)
    for j in range(k):
        codes = codes * 4 + aclip[j:j + m]
        vw &= valid[j:j + m]
    codes = codes[vw]
    if codes.size == 0:
        return np.empty(0, dtype=np.uint64)
    h = np.unique(_mix64(codes.astype(np.uint64)))  # sorted ascending, unique
    return h[:num].copy()


def sketch_entities(names, genome_index, k: int = 21, num: int = 256,
                    n_workers: int = 8) -> dict:
    """Compute MinHash sketches for `names`, streaming one genome at a time.

    Memory-aware like build_node_features: a worker loads a genome, sketches it
    to <=num uint64, and drops the (megabase) sequence immediately."""
    from concurrent.futures import ThreadPoolExecutor
    names = list(names)
    out: dict = {}
    workers = max(1, min(n_workers, 10))

    def _proc(nm):
        s = genome_index.load_sequence(nm)
        return nm, (minhash_sketch(s, k, num) if s else np.empty(0, np.uint64))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for nm, sk in pool.map(_proc, names):
            out[nm] = sk
    covered = sum(1 for v in out.values() if v.size)
    log.info("[cluster] sketched %d entities (k=%d, num=%d); %d with genomes",
             len(names), k, num, covered)
    return out


# --- similarity + clustering -------------------------------------------------

def _jaccard(s1: np.ndarray, s2: np.ndarray, num: int) -> float:
    if s1.size == 0 or s2.size == 0:
        return 0.0
    merged = np.union1d(s1, s2)          # sorted unique
    bottom = merged[:num]
    in1 = np.isin(bottom, s1, assume_unique=True)
    in2 = np.isin(bottom, s2, assume_unique=True)
    return float((in1 & in2).sum()) / len(bottom)


def _jaccard_threshold_for_distance(max_distance: float, k: int) -> float:
    """Jaccard >= this  <=>  Mash distance <= max_distance (so we skip logs)."""
    e = np.exp(-k * max_distance)
    return float(e / (2.0 - e))


def mash_distance(s1: np.ndarray, s2: np.ndarray, k: int, num: int) -> float:
    j = _jaccard(s1, s2, num)
    if j <= 0:
        return 1.0
    if j >= 1:
        return 0.0
    return float(-np.log(2 * j / (1 + j)) / k)


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_clusters(names, sketches: dict, max_distance: float = 0.05,
                   k: int = 21, num: int = 256) -> dict:
    """Single-linkage clustering at a Mash-form distance threshold.

    The distance transforms the in-house bottom-k Jaccard estimate using Mash's
    formula; this function does not invoke Mash, and the cutoff is not a direct
    ANI measurement. Entities without a sketch get singleton clusters. Returns
    name -> integer cluster id."""
    names = list(names)
    n = len(names)
    sk = [sketches.get(nm, np.empty(0, np.uint64)) for nm in names]
    j_thr = _jaccard_threshold_for_distance(max_distance, k)
    uf = _UnionFind(n)
    have = [i for i in range(n) if sk[i].size]
    comparisons = 0
    for a in range(len(have)):
        i = have[a]
        si = sk[i]
        for b in range(a + 1, len(have)):
            j = have[b]
            if uf.find(i) == uf.find(j):
                continue
            comparisons += 1
            if _jaccard(si, sk[j], num) >= j_thr:
                uf.union(i, j)
    roots = {}
    name_to_cluster = {}
    for i, nm in enumerate(names):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        name_to_cluster[nm] = roots[r]
    log.info("[cluster] %d entities -> %d clusters at Mash d<=%.3f "
             "(Jaccard>=%.3f, %d comparisons)", n, len(roots), max_distance,
             j_thr, comparisons)
    return name_to_cluster


# --- cluster-based folds -----------------------------------------------------

def combined_unseen_folds(df: pd.DataFrame, phage_cluster_col: str,
                          host_cluster_col: str, n_splits: int = 5,
                          seed: int = 42, min_pos: int = 3, min_neg: int = 3):
    """Block cross-validation holding out BOTH axes by genome cluster.

    Phage clusters and host clusters are each seeded, permuted, and assigned to
    `n_splits` bins round-robin (the bins are not pair/label balanced). For fold
    i: test = pairs whose phage-cluster AND host-cluster are both
    in bin i; train = pairs whose phage-cluster AND host-cluster are both NOT in
    bin i. Pairs with exactly one held-out side are DISCARDED as a leakage
    buffer. Guarantees: no test phage (or sequence-similar phage) and no test
    host (or sequence-similar host) appears in training -> true cold-start.
    """
    rng = np.random.default_rng(seed)
    pcs = rng.permutation(df[phage_cluster_col].unique())
    hcs = rng.permutation(df[host_cluster_col].unique())
    p_bin = {c: i % n_splits for i, c in enumerate(pcs)}
    h_bin = {c: i % n_splits for i, c in enumerate(hcs)}
    pb = df[phage_cluster_col].map(p_bin).to_numpy()
    hb = df[host_cluster_col].map(h_bin).to_numpy()
    idx = np.arange(len(df))
    labels = df["label"].to_numpy()
    for i in range(n_splits):
        test_mask = (pb == i) & (hb == i)
        train_mask = (pb != i) & (hb != i)
        n_pos = int(labels[test_mask].sum())
        n_neg = int((labels[test_mask] == 0).sum())
        if n_pos < min_pos or n_neg < min_neg:
            continue
        if labels[train_mask].sum() == 0 or (labels[train_mask] == 0).sum() == 0:
            continue
        yield Fold(name=f"block{i}", regime="combined",
                   train_idx=idx[train_mask], test_idx=idx[test_mask])


# --- leakage audit -----------------------------------------------------------

def audit_taxonomic_leakage(df: pd.DataFrame, group_col: str, cluster_col: str,
                            entity_col: str) -> dict:
    """Quantify leakage that a leave-one-`group_col`-out split would hide.

    If a genome cluster spans >1 group value, then holding out one group still
    leaves sequence-similar (same-cluster) entities of another group in
    training -> homology leakage. Reports how many clusters/entities/rows are
    affected."""
    spans = df.groupby(cluster_col)[group_col].nunique()
    multi = set(spans[spans > 1].index)
    affected_rows = int(df[cluster_col].isin(multi).sum())
    affected_entities = int(df[df[cluster_col].isin(multi)][entity_col].nunique())
    total_entities = int(df[entity_col].nunique())
    return {
        "group_col": group_col,
        "n_clusters": int(df[cluster_col].nunique()),
        "n_groups": int(df[group_col].nunique()),
        "clusters_spanning_multiple_groups": len(multi),
        "leaky_rows": affected_rows,
        "leaky_row_fraction": affected_rows / max(1, len(df)),
        "leaky_entities": affected_entities,
        "total_entities": total_entities,
        "leaky_entity_fraction": affected_entities / max(1, total_entities),
    }
