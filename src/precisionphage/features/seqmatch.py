"""Pair-intrinsic, leakage-free sequence-match features.

Both families below depend ONLY on the phage genome and the host genome of a
pair (never on interaction labels or on any other pair), so they are safe to
precompute once and reuse across all folds without leakage.

  * Homology (a BLASTN-style proxy without the binary): long exact k-mer
    containment of the phage in the host (both strands), fraction of the phage
    genome covered by >=k exact matches, and the longest contiguous exact match.
    Captures prophage / shared-region signal that strongly informs host range.

  * CRISPR: detect CRISPR arrays in the HOST genome as conserved repeat anchors
    that recur at regular spacing, extract the intervening spacers, and test
    whether those spacers occur in the phage genome (either strand). A spacer
    match is a high-precision immunity signal.

No external tools required; everything is vectorized NumPy and scales to
megabase host genomes.
"""
from __future__ import annotations

import numpy as np

from ..utils import get_logger
from .genomic import _encode

log = get_logger(__name__)

_UINT64 = np.uint64


def _mix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64, copy=False)
    x = x ^ (x >> _UINT64(33))
    x = x * _UINT64(0xFF51AFD7ED558CCD)
    x = x ^ (x >> _UINT64(33))
    x = x * _UINT64(0xC4CEB9FE1A85EC53)
    x = x ^ (x >> _UINT64(33))
    return x


def _rc_bases(a: np.ndarray) -> np.ndarray:
    """Reverse-complement a base-code array (A0 C1 G2 T3; -1 invalid)."""
    r = a[::-1].copy()
    v = r >= 0
    r[v] = 3 - r[v]
    return r


def _codes_ordered(a: np.ndarray, k: int):
    """Ordered k-mer integer codes for a base array; returns (codes, valid)."""
    n = a.shape[0]
    if n < k:
        return np.empty(0, np.int64), np.empty(0, bool)
    valid = a >= 0
    aclip = np.where(valid, a, 0)
    m = n - k + 1
    codes = np.zeros(m, dtype=np.int64)
    vw = np.ones(m, dtype=bool)
    for j in range(k):
        codes = codes * 4 + aclip[j:j + m]
        vw &= valid[j:j + m]
    return codes, vw


def _hash_codes(codes: np.ndarray) -> np.ndarray:
    return _mix64(codes.astype(np.uint64))


def host_kmer_set(hseq: str, k: int) -> np.ndarray:
    """Sorted unique hashed k-mers of the host, BOTH strands."""
    a = _encode(hseq)
    fwd, vf = _codes_ordered(a, k)
    rc, vr = _codes_ordered(_rc_bases(a), k)
    parts = []
    if fwd.size:
        parts.append(_hash_codes(fwd[vf]))
    if rc.size:
        parts.append(_hash_codes(rc[vr]))
    if not parts:
        return np.empty(0, np.uint64)
    return np.unique(np.concatenate(parts))


def phage_kmer_set(pseq: str, k: int) -> np.ndarray:
    """Sorted unique hashed k-mers of the phage, BOTH strands (for spacer hits)."""
    return host_kmer_set(pseq, k)


def _membership(query_hashes: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    if query_hashes.size == 0 or sorted_set.size == 0:
        return np.zeros(query_hashes.shape[0], dtype=bool)
    ins = np.searchsorted(sorted_set, query_hashes)
    ins = np.clip(ins, 0, len(sorted_set) - 1)
    return sorted_set[ins] == query_hashes


def homology_features(pseq: str, host_set: np.ndarray, k: int,
                      prefix: str = "hom") -> dict:
    """Phage-in-host exact homology features (host_set must include both strands)."""
    a = _encode(pseq)
    plen = a.shape[0]
    codes, vw = _codes_ordered(a, k)
    m = codes.shape[0]
    keys = (f"{prefix}_frac_kmers", f"{prefix}_cov_frac",
            f"{prefix}_log_shared", f"{prefix}_longest_frac")
    out = {kk: 0.0 for kk in keys}
    if m == 0 or host_set.size == 0:
        return out
    idxv = np.where(vw)[0]
    if idxv.size == 0:
        return out
    qh = _hash_codes(codes[idxv])
    hit = _membership(qh, host_set)
    shared = int(hit.sum())
    out[keys[0]] = shared / float(idxv.size)
    out[keys[2]] = float(np.log1p(shared))
    if shared == 0:
        return out
    present = np.zeros(m, dtype=bool)
    present[idxv] = hit
    # coverage of the phage genome by >=k exact matches
    diff = np.zeros(plen + 1, dtype=np.int32)
    starts = np.where(present)[0]
    np.add.at(diff, starts, 1)
    np.add.at(diff, np.minimum(starts + k, plen), -1)
    covered = int((np.cumsum(diff)[:plen] > 0).sum())
    out[keys[1]] = covered / float(max(1, plen))
    # longest run of consecutive present anchors -> contiguous match length
    pad = np.concatenate(([0], present.astype(np.int8), [0]))
    chg = np.diff(pad)
    run_starts = np.where(chg == 1)[0]
    run_ends = np.where(chg == -1)[0]
    longest = int((run_ends - run_starts).max()) if run_starts.size else 0
    out[keys[3]] = (longest + k - 1) / float(max(1, plen)) if longest else 0.0
    return out


# --- CRISPR ------------------------------------------------------------------

def find_crispr_spacers(hseq: str, repeat_k: int = 23, min_copies: int = 3,
                        max_copies: int = 100, spacer_min: int = 20,
                        spacer_max: int = 75, max_spacers: int = 500) -> list:
    """Extract CRISPR spacers from a host genome.

    Heuristic, tool-free: a conserved repeat appears as a k-mer (anchor) that
    recurs >= min_copies times at regular spacing (period in
    [repeat_k+spacer_min, repeat_k+spacer_max]). The sequence between successive
    anchors is taken as a spacer. The regular-spacing constraint suppresses
    rRNA/IS repeats. Returns a list of spacer strings.
    """
    a = _encode(hseq)
    codes, vw = _codes_ordered(a, repeat_k)
    if codes.size == 0:
        return []
    valid_idx = np.where(vw)[0]
    if valid_idx.size == 0:
        return []
    cv = codes[valid_idx]
    order = np.argsort(cv, kind="stable")
    cs = cv[order]
    positions = valid_idx[order]                 # ascending within equal codes
    # group boundaries of identical codes
    change = np.nonzero(np.diff(cs))[0] + 1
    group_starts = np.concatenate(([0], change))
    group_ends = np.concatenate((change, [len(cs)]))
    period_min = repeat_k + spacer_min
    period_max = repeat_k + spacer_max
    seq = hseq
    spacers: list[str] = []
    seen: set[str] = set()
    for gs, ge in zip(group_starts, group_ends):
        size = ge - gs
        if size < min_copies or size > max_copies:
            continue
        gp = np.sort(positions[gs:ge])
        gaps = np.diff(gp)
        in_range = (gaps >= period_min) & (gaps <= period_max)
        if in_range.sum() < (min_copies - 1):
            continue
        for i in np.where(in_range)[0]:
            s0 = gp[i] + repeat_k
            s1 = gp[i + 1]
            sp = seq[s0:s1]
            ln = len(sp)
            if spacer_min <= ln <= spacer_max and sp not in seen:
                seen.add(sp)
                spacers.append(sp)
                if len(spacers) >= max_spacers:
                    return spacers
    return spacers


def crispr_match_features(spacers: list, phage_set: np.ndarray, k: int,
                          match_frac: float = 0.6) -> dict:
    """Do host spacers occur in the phage genome (phage_set = both strands)?"""
    out = {"crispr_n_spacers": float(len(spacers)), "crispr_n_hits": 0.0,
           "crispr_best_frac": 0.0, "crispr_has_hit": 0.0}
    if not spacers or phage_set.size == 0:
        return out
    n_hits = 0
    best = 0.0
    for sp in spacers:
        a = _encode(sp)
        codes, vw = _codes_ordered(a, k)
        if codes.size == 0 or not vw.any():
            continue
        qh = _hash_codes(codes[vw])
        frac = float(_membership(qh, phage_set).mean())
        if frac > best:
            best = frac
        if frac >= match_frac:
            n_hits += 1
    out["crispr_n_hits"] = float(n_hits)
    out["crispr_best_frac"] = best
    out["crispr_has_hit"] = 1.0 if n_hits > 0 else 0.0
    return out


CRISPR_COLS = ["crispr_n_spacers", "crispr_n_hits", "crispr_best_frac",
               "crispr_has_hit"]


def hom_cols(ks) -> list:
    cols = []
    for k in ks:
        p = f"hom{k}"
        cols += [f"{p}_frac_kmers", f"{p}_cov_frac", f"{p}_log_shared",
                 f"{p}_longest_frac"]
    return cols


def pair_feature_cols(cfg) -> list:
    ks = cfg["features"].get("homology_ks", [cfg["features"].get("homology_k", 20)])
    cols = hom_cols(ks) + CRISPR_COLS
    if cfg["features"].get("use_protein_features", False):
        from .proteins import PROT_COLS
        cols = cols + PROT_COLS
    return cols


# --- parallel pair-feature precompute ----------------------------------------
# Pair features are grouped by host so each host genome is read and sketched
# once; phage genomes are cached within a worker. Hosts are sharded across
# spawn workers (<=10), each pinned to one thread.
_W: dict = {}


def _seq_init(phage_dirs, host_dirs, ks, mk, rk, crispr_kwargs, threads, prot):
    from ..data import GenomeIndex
    from ..utils import limit_threads
    limit_threads(threads)
    _W["pidx"] = GenomeIndex([__import__("pathlib").Path(d) for d in phage_dirs])
    _W["hidx"] = GenomeIndex([__import__("pathlib").Path(d) for d in host_dirs])
    _W["ks"] = ks
    _W["mk"] = mk
    _W["rk"] = rk
    _W["crispr"] = crispr_kwargs
    _W["prot"] = prot                 # None or {"k":.., "min_pep":..}
    _W["pcache"] = {}
    cols = hom_cols(ks) + CRISPR_COLS
    if prot:
        from .proteins import PROT_COLS
        cols = cols + PROT_COLS
    _W["cols"] = cols


def _zeros_row(phage, host):
    d = {"phage": phage, "host": host}
    d.update({c: 0.0 for c in _W["cols"]})
    return d


def _host_worker(item):
    host, phages = item
    hidx, pidx, ks, mk = _W["hidx"], _W["pidx"], _W["ks"], _W["mk"]
    prot = _W.get("prot")
    hseq = hidx.load_sequence(host)
    if not hseq:
        return [_zeros_row(p, host) for p in phages]
    host_sets = {k: host_kmer_set(hseq, k) for k in ks}
    spacers = find_crispr_spacers(hseq, repeat_k=_W["rk"], **_W["crispr"])
    host_pset_prot = None
    if prot:
        from .proteins import protein_kmer_set
        host_pset_prot = protein_kmer_set(hseq, prot["k"], prot["min_pep"])
    rows = []
    for ph in phages:
        pseq = _W["pcache"].get(ph)
        if pseq is None:
            pseq = pidx.load_sequence(ph) or ""
            _W["pcache"][ph] = pseq
        if not pseq:
            rows.append(_zeros_row(ph, host))
            continue
        row = {"phage": ph, "host": host}
        for k in ks:
            row.update(homology_features(pseq, host_sets[k], k, prefix=f"hom{k}"))
        pset = phage_kmer_set(pseq, mk)
        row.update(crispr_match_features(spacers, pset, mk))
        if prot:
            from .proteins import protein_homology_features
            row.update(protein_homology_features(pseq, host_pset_prot,
                                                 prot["k"], prot["min_pep"]))
        rows.append(row)
    return rows


def compute_pair_features(cov, cfg, n_jobs: int = 10):
    """Compute homology + CRISPR features for every (phage, host) in `cov`.

    Returns a DataFrame with columns ['phage','host'] + pair_feature_cols(cfg).
    Leakage-free: every value depends only on the two genomes of the pair."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    from ..utils import resolve_n_jobs

    fcfg = cfg["features"]
    ks = [int(x) for x in fcfg.get("homology_ks", [fcfg.get("homology_k", 20)])]
    mk = int(fcfg.get("crispr_match_k", 20))
    rk = int(fcfg["crispr_repeat_k"])
    crispr_kwargs = {"min_copies": 3, "spacer_min": 20, "spacer_max": 75}
    prot = None
    if fcfg.get("use_protein_features", False):
        prot = {"k": int(fcfg.get("protein_k", 7)),
                "min_pep": int(fcfg.get("protein_min_pep", 90))}
    groups = (cov.groupby("host")["phage"]
              .apply(lambda s: sorted(s.unique())).to_dict())
    items = list(groups.items())
    n = resolve_n_jobs(cfg, len(items)) if n_jobs is None else max(1, min(n_jobs, 10, len(items)))
    pdirs = [str(cfg["paths"]["phage_fasta_dir"])]
    hdirs = [str(cfg["paths"]["host_fasta_dir"])]
    threads = int(cfg.get("compute", {}).get("threads_per_job", 1))
    log.info("[seqmatch] computing pair features for %d hosts / %d pairs "
             "(ks=%s, match_k=%d, repeat_k=%d, protein=%s, %d workers)",
             len(items), len(cov), ks, mk, rk, prot, n)

    all_rows = []
    args = (pdirs, hdirs, ks, mk, rk, crispr_kwargs, threads, prot)
    if n <= 1:
        _seq_init(*args)
        for it in items:
            all_rows.extend(_host_worker(it))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                                 initializer=_seq_init, initargs=args) as ex:
            for rows in ex.map(_host_worker, items):
                all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    first_hom = f"hom{ks[0]}_frac_kmers"
    log.info("[seqmatch] done: %d pair rows; crispr hit rate=%.3f, "
             "homology(k=%d)>0 rate=%.3f", len(df),
             float((df["crispr_has_hit"] > 0).mean()), ks[0],
             float((df[first_hom] > 0).mean()))
    return df
