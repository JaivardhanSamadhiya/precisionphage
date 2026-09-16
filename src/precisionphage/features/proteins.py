"""Protein-level, leakage-free pair features (a tblastx-style proxy).

Nucleotide homology only catches near-identical DNA; many true homologs have
diverged at the DNA level (synonymous/codon wobble) yet stay conserved as
protein. Translating both genomes in all six frames and measuring exact
amino-acid k-mer sharing recovers that divergent homology.

Like the nucleotide homology features, every value depends ONLY on the phage and
host genome of the pair (six-frame translation already covers both strands), so
the features are pair-intrinsic and leakage-free.

Pure NumPy; no external tools. Scales to megabase genomes.
"""
from __future__ import annotations

import numpy as np

from ..utils import get_logger
from .genomic import _encode

log = get_logger(__name__)

_UINT64 = np.uint64
_AA = "ACDEFGHIKLMNPQRSTVWY"          # 20-letter index 0..19; stop -> -1
_AA_IDX = {c: i for i, c in enumerate(_AA)}

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# codon integer (16*b0+4*b1+b2 with A0 C1 G2 T3) -> aa index (0..19) or -1 (stop)
_AA_LUT = np.full(64, -1, dtype=np.int64)
_B = {"A": 0, "C": 1, "G": 2, "T": 3}
for _cod, _aa in _CODON_TABLE.items():
    _ci = 16 * _B[_cod[0]] + 4 * _B[_cod[1]] + _B[_cod[2]]
    _AA_LUT[_ci] = _AA_IDX[_aa] if _aa != "*" else -1


def _mix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64, copy=False)
    x = x ^ (x >> _UINT64(33))
    x = x * _UINT64(0xFF51AFD7ED558CCD)
    x = x ^ (x >> _UINT64(33))
    x = x * _UINT64(0xC4CEB9FE1A85EC53)
    x = x ^ (x >> _UINT64(33))
    return x


def _rc_bases(a: np.ndarray) -> np.ndarray:
    r = a[::-1].copy()
    v = r >= 0
    r[v] = 3 - r[v]
    return r


def _translate_strand(a: np.ndarray):
    """Return the 3 forward-frame amino-acid arrays for base-code array `a`
    (aa index 0..19; -1 marks a stop or an invalid codon, i.e. a peptide break)."""
    n = a.shape[0]
    frames = []
    for o in range(3):
        m = (n - o) // 3
        if m <= 0:
            frames.append(np.empty(0, np.int64))
            continue
        tri = a[o:o + 3 * m].reshape(m, 3)
        valid = (tri >= 0).all(axis=1)
        tclip = np.where(tri >= 0, tri, 0)
        codon = 16 * tclip[:, 0] + 4 * tclip[:, 1] + tclip[:, 2]
        aa = _AA_LUT[codon]
        aa[~valid] = -1
        frames.append(aa)
    return frames


def _filter_peptides(aa: np.ndarray, min_len: int) -> np.ndarray:
    """Mask (set to -1) amino acids that lie in stop-delimited peptides shorter
    than `min_len`, removing spurious short 6-frame ORFs that otherwise saturate
    k-mer space. Real genes are long; random ORFs average ~21 aa."""
    if min_len <= 1 or aa.size == 0:
        return aa
    valid = aa >= 0
    pad = np.concatenate(([0], valid.astype(np.int8), [0]))
    chg = np.diff(pad)
    rs = np.where(chg == 1)[0]
    re = np.where(chg == -1)[0]
    out = aa.copy()
    for s, e in zip(rs, re):
        if e - s < min_len:
            out[s:e] = -1
    return out


def six_frame(seq: str, min_pep_len: int = 0):
    a = _encode(seq)
    frames = _translate_strand(a) + _translate_strand(_rc_bases(a))
    if min_pep_len > 0:
        frames = [_filter_peptides(f, min_pep_len) for f in frames]
    return frames


def _codes_ordered_aa(aa: np.ndarray, k: int):
    """Ordered amino-acid k-mer codes (base 20); window invalid if it spans a
    stop/invalid (aa == -1). Returns (codes, valid)."""
    m = aa.shape[0] - k + 1
    if m <= 0:
        return np.empty(0, np.int64), np.empty(0, bool)
    valid = aa >= 0
    aclip = np.where(valid, aa, 0)
    codes = np.zeros(m, dtype=np.int64)
    vw = np.ones(m, dtype=bool)
    for j in range(k):
        codes = codes * 20 + aclip[j:j + m]
        vw &= valid[j:j + m]
    return codes, vw


def _hash_codes(codes: np.ndarray) -> np.ndarray:
    return _mix64(codes.astype(np.uint64))


def protein_kmer_set(seq: str, k: int = 6, min_pep_len: int = 60) -> np.ndarray:
    """Sorted unique hashed amino-acid k-mers over real ORFs in all six frames."""
    parts = []
    for aa in six_frame(seq, min_pep_len):
        codes, vw = _codes_ordered_aa(aa, k)
        if codes.size and vw.any():
            parts.append(_hash_codes(codes[vw]))
    if not parts:
        return np.empty(0, np.uint64)
    return np.unique(np.concatenate(parts))


def _membership(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    if q.size == 0 or s.size == 0:
        return np.zeros(q.shape[0], dtype=bool)
    ins = np.clip(np.searchsorted(s, q), 0, len(s) - 1)
    return s[ins] == q


PROT_COLS = ["prot_frac_kmers", "prot_cov_frac", "prot_log_shared",
             "prot_longest_frac"]


def protein_homology_features(pseq: str, host_pset: np.ndarray,
                              k: int = 6, min_pep_len: int = 60) -> dict:
    """Phage-in-host protein homology (host_pset spans all six host frames)."""
    out = {c: 0.0 for c in PROT_COLS}
    if host_pset.size == 0:
        return out
    ncodon = max(1, len(pseq) // 3)
    shared = total = covered = total_aa = 0
    longest = 0
    for aa in six_frame(pseq, min_pep_len):
        naa = int((aa >= 0).sum())
        total_aa += naa
        codes, vw = _codes_ordered_aa(aa, k)
        if codes.size == 0 or not vw.any():
            continue
        idxv = np.where(vw)[0]
        hit = _membership(_hash_codes(codes[idxv]), host_pset)
        shared += int(hit.sum())
        total += idxv.size
        if hit.any():
            present = np.zeros(aa.shape[0], dtype=bool)
            present[idxv] = hit
            L = aa.shape[0]
            diff = np.zeros(L + 1, dtype=np.int32)
            starts = np.where(present)[0]
            np.add.at(diff, starts, 1)
            np.add.at(diff, np.minimum(starts + k, L), -1)
            covered += int((np.cumsum(diff)[:L] > 0).sum())
            pad = np.concatenate(([0], present.astype(np.int8), [0]))
            chg = np.diff(pad)
            rs = np.where(chg == 1)[0]
            re = np.where(chg == -1)[0]
            if rs.size:
                longest = max(longest, int((re - rs).max()) + k - 1)
    if total > 0:
        out["prot_frac_kmers"] = shared / float(total)
        out["prot_log_shared"] = float(np.log1p(shared))
    if total_aa > 0:
        out["prot_cov_frac"] = covered / float(total_aa)
    out["prot_longest_frac"] = longest / float(ncodon) if longest else 0.0
    return out
