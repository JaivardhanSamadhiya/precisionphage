"""Genome-derived features.

Node features (intrinsic to one genome, no leakage): canonical k-mer spectrum,
codon usage, GC content, genome length, dinucleotide bias.

Edge features (pairwise, computable for ANY candidate pair, no leakage):
k-mer cosine distance, GC difference, oligonucleotide-frequency correlation,
and a homology proxy (k-mer Jaccard) — BLASTN can replace the proxy when the
binary is available.

Dimensionality reduction (PCA) is deliberately NOT done here; it is fit inside
each training fold to preserve the leakage-free guarantee.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..utils import get_logger

log = get_logger(__name__)

_NUC = "ACGT"
_NUC_IDX = {c: i for i, c in enumerate(_NUC)}
_COMP = str.maketrans("ACGT", "TGCA")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _canonical_kmer_index(k: int):
    """Map each of the 4^k k-mers to a canonical-class column index."""
    n = 4 ** k
    rep_to_col: dict[int, int] = {}
    code_to_col = np.empty(n, dtype=np.int64)
    bases = _NUC
    def to_code(s):
        c = 0
        for ch in s:
            c = c * 4 + _NUC_IDX[ch]
        return c
    from itertools import product
    for tup in product(bases, repeat=k):
        s = "".join(tup)
        code = to_code(s)
        canon = min(s, _revcomp(s))
        ccode = to_code(canon)
        if ccode not in rep_to_col:
            rep_to_col[ccode] = len(rep_to_col)
        code_to_col[code] = rep_to_col[ccode]
    return code_to_col, len(rep_to_col)


_KMER_CACHE: dict[int, tuple] = {}

# 256-entry lookup table: ASCII byte -> base index (A0 C1 G2 T3), else -1.
_LUT = np.full(256, -1, dtype=np.int64)
for _c, _i in _NUC_IDX.items():
    _LUT[ord(_c)] = _i


def _encode(seq: str) -> np.ndarray:
    """Map an ACGT(N) sequence to an int64 array of base codes (-1 for non-ACGT).

    Sequences are pre-uppercased/filtered to ACGTN upstream; this is vectorized
    so it scales to megabase bacterial genomes."""
    arr = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
    return _LUT[arr]


def kmer_spectrum(seq: str, k: int = 4) -> np.ndarray:
    """Strand-independent (canonical) normalized k-mer frequency vector."""
    if k not in _KMER_CACHE:
        _KMER_CACHE[k] = _canonical_kmer_index(k)
    code_to_col, n_cols = _KMER_CACHE[k]
    out = np.zeros(n_cols, dtype=np.float64)
    if not seq or len(seq) < k:
        return out.astype(np.float32)
    a = _encode(seq)
    n = a.shape[0]
    if n < k:
        return out.astype(np.float32)
    valid = a >= 0
    aclip = np.where(valid, a, 0)
    m = n - k + 1
    codes = np.zeros(m, dtype=np.int64)
    vw = np.ones(m, dtype=bool)
    for j in range(k):
        codes = codes * 4 + aclip[j:j + m]
        vw &= valid[j:j + m]
    codes = codes[vw]
    if codes.size == 0:
        return out.astype(np.float32)
    counts = np.bincount(code_to_col[codes], minlength=n_cols).astype(np.float64)
    out += counts
    s = out.sum()
    return (out / s if s > 0 else out).astype(np.float32)


def codon_usage(seq: str) -> np.ndarray:
    """64-dim codon frequency over all three forward frames."""
    out = np.zeros(64, dtype=np.float64)
    if not seq or len(seq) < 3:
        return out.astype(np.float32)
    a = _encode(seq)
    valid = a >= 0
    aclip = np.where(valid, a, 0)
    for frame in (0, 1, 2):
        s = aclip[frame:]
        v = valid[frame:]
        m = (s.shape[0] // 3) * 3
        if m == 0:
            continue
        tri = s[:m].reshape(-1, 3)
        vtri = v[:m].reshape(-1, 3)
        codes = tri[:, 0] * 16 + tri[:, 1] * 4 + tri[:, 2]
        good = vtri.all(axis=1)
        out += np.bincount(codes[good], minlength=64).astype(np.float64)
    s = out.sum()
    return (out / s if s > 0 else out).astype(np.float32)


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    a = _encode(seq)
    valid = a >= 0
    n = int(valid.sum())
    if n == 0:
        return 0.0
    gc = int(((a == 1) | (a == 2)).sum())  # C=1, G=2
    return gc / n


def _node_vector(seq: str, k: int, use_codon: bool, use_dinuc: bool) -> np.ndarray:
    parts = [kmer_spectrum(seq, k)]
    if use_dinuc:
        parts.append(kmer_spectrum(seq, 2))
    if use_codon:
        parts.append(codon_usage(seq))
    parts.append(np.array([gc_content(seq),
                           np.log10(len(seq) + 1.0)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _node_dim(k: int, use_codon: bool, use_dinuc: bool) -> int:
    dim = _canonical_kmer_index(k)[1]
    if use_dinuc:
        dim += _canonical_kmer_index(2)[1]
    if use_codon:
        dim += 64
    return dim + 2  # gc + log10(len)


def build_node_features(names, genome_index, k: int = 4, use_codon: bool = True,
                        use_dinuc: bool = True, n_workers: int = 8):
    """Return (feature_matrix [N, D], has_genome_mask [N]) aligned to `names`.

    Memory-aware: each worker loads ONE genome, computes the small D-dim feature
    vector, and discards the (possibly megabase) sequence immediately, so peak
    memory is ~n_workers genomes rather than the whole corpus. Entities without
    a genome get a zero vector and has_genome=False (never silently imputed).
    """
    names = list(names)
    n = len(names)
    dim = _node_dim(k, use_codon, use_dinuc)
    X = np.zeros((n, dim), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    workers = max(1, min(n_workers, 10))

    def _proc(i):
        s = genome_index.load_sequence(names[i])
        if not s:
            return i, None
        return i, _node_vector(s, k, use_codon, use_dinuc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, v in pool.map(_proc, range(n)):
            if v is not None:
                X[i] = v
                mask[i] = True
    if not mask.any():
        raise RuntimeError("No genomes could be loaded for node features.")
    log.info("[features] node matrix %s; genomes present for %d/%d entities "
             "(%d workers)", X.shape, int(mask.sum()), n, workers)
    return X, mask


def edge_features_from_spectra(p_kmer: np.ndarray, h_kmer: np.ndarray) -> np.ndarray:
    """Leakage-safe pairwise features from two k-mer spectra.

    Returns [cosine_dist, l1_dist, pearson_corr, jaccard_proxy].
    """
    a = p_kmer.astype(np.float64)
    b = h_kmer.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
    l1 = float(np.abs(a - b).sum())
    am, bm = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(am) * np.linalg.norm(bm)
    pear = float(am @ bm / denom) if denom > 0 else 0.0
    inter = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum()
    jac = float(inter / union) if union > 0 else 0.0
    return np.array([1.0 - cos, l1, pear, jac], dtype=np.float32)
