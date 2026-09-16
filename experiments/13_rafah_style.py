#!/usr/bin/env python3
"""Step 13: RaFAH-style external baseline (in-house reimplementation).

IMPORTANT: This is NOT the published, pretrained RaFAH. The real RaFAH could not
be run in this environment: it requires an R runtime (its random forest is an R
'ranger' model) which is absent here, and its pretrained model/HMM database is
hosted on figshare, which is network-blocked (HTTP 403) from this cluster.

Instead we implement a RaFAH-inspired proxy for a limited methodological
comparison: predict the bacterial host TAXON (genus) from a phage's PROTEIN
content using a Random Forest. Proteins are obtained by six-frame ORF translation
(min ORF 60 aa); each phage is encoded as a presence vector of feature-hashed
amino-acid k-mers (proxy for RaFAH's protein-cluster/HMM presence features). The
RF is trained on known phage->host-genus associations within each training fold
and predicts a host-genus distribution for test phages; a (phage, host) pair is
scored by P(predicted genus == host's genus).

Evaluated on the SAME leakage-controlled test pairs as the GBM (reusing the
aligned GBM predictions saved by step 12) and compared with DeLong's test.

Run (from /tmp, then cd in):
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python -u experiments/13_rafah_style.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from precisionphage.eval import (  # noqa: E402
    benjamini_hochberg, bootstrap_auc_diff, delong_auc_ci, delong_test,
    permutation_auc_test,
)
from precisionphage.features.assembly import build_covered_dataset  # noqa: E402
from precisionphage.data import genome_set_digest  # noqa: E402
from precisionphage.features.proteins import protein_kmer_set  # noqa: E402
from precisionphage.splits import (  # noqa: E402
    combined_unseen_folds, leave_one_group_out, load_or_build_clusters,
)
from precisionphage.utils import (  # noqa: E402
    ensure_dirs, get_logger, limit_threads, load_config, set_determinism,
)

log = get_logger("rafah_style")
FP_DIM = 65536            # large hash space so protein-word presence stays sparse
PROT_K = 6               # 6-aa words approximate conserved protein motifs/families
MIN_PEP = 60

RLAB = {"loso_species": "Unseen species (LOSO)",
        "host_cluster": "Unseen host cluster",
        "phage_cluster": "Unseen phage cluster",
        "combined_unseen": "Both unseen (cold start)"}


def _assign_clusters(cfg, data):
    return load_or_build_clusters(cfg, data)


def _phage_fingerprints(data, cfg):
    """Binary feature-hashed AA-kmer presence vector per phage (RaFAH-style
    protein-content features). Cached to interim_dir."""
    cache = cfg["paths"]["interim_dir"] / f"phage_protein_fp_d{FP_DIM}_k{PROT_K}.npy"
    meta_path = cache.with_suffix(".json")
    source_sha256 = genome_set_digest(data.phages, data.phage_index)
    expected_meta = {"source_sha256": source_sha256, "n_phages": len(data.phages),
                     "fp_dim": FP_DIM, "protein_k": PROT_K, "min_pep": MIN_PEP}
    cached_meta = (json.loads(meta_path.read_text(encoding="utf-8"))
                   if meta_path.exists() else None)
    if cache.exists() and cached_meta == expected_meta:
        fp = np.load(cache)
        if fp.shape == (len(data.phages), FP_DIM):
            log.info("[rafah] loaded cached phage fingerprints %s", fp.shape)
            return fp
    fp = np.zeros((len(data.phages), FP_DIM), dtype=np.float32)
    for i, name in enumerate(data.phages):
        seq = data.phage_index.load_sequence(name)
        if not seq:
            continue
        hashes = protein_kmer_set(seq, k=PROT_K, min_pep_len=MIN_PEP)
        if hashes.size:
            bins = (hashes % np.uint64(FP_DIM)).astype(np.int64)
            fp[i, np.unique(bins)] = 1.0
        if (i + 1) % 300 == 0:
            log.info("[rafah] fingerprinted %d/%d phages", i + 1, len(data.phages))
    np.save(cache, fp)
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    log.info("[rafah] computed phage fingerprints %s (mean density %.3f)",
             fp.shape, float(fp.mean()))
    return fp


def _rafah_oof(cov, fp, folds, seed):
    """Per-fold RF (phage protein fp -> host genus), trained on positive train
    pairs; returns OOF pair scores aligned to cov rows for pooled test pairs."""
    genus = cov["host_genus"].to_numpy()
    pidx = cov["pidx"].to_numpy()
    y = cov["label"].to_numpy().astype(int)
    score = np.full(len(cov), np.nan, dtype=np.float64)
    for f in folds:
        tr, te = f.train_idx, f.test_idx
        pos = tr[y[tr] == 1]
        if pos.size < 5 or np.unique(genus[pos]).size < 2:
            score[te] = 0.0
            continue
        Xtr = fp[pidx[pos]]
        ytr = genus[pos]
        rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                    n_jobs=8, random_state=seed, class_weight="balanced")
        rf.fit(Xtr, ytr)
        classes = list(rf.classes_)
        cls_idx = {c: j for j, c in enumerate(classes)}
        proba = rf.predict_proba(fp[pidx[te]])
        for r, row in zip(te, proba):
            g = genus[r]
            score[r] = row[cls_idx[g]] if g in cls_idx else 0.0
    return score


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    seed = cfg["seed"]
    set_determinism(seed)
    ensure_dirs(cfg)
    limit_threads(1)
    rd = cfg["paths"]["results_dir"]

    data = build_covered_dataset(cfg)
    cov = data.df.reset_index(drop=True)
    pc, hc = _assign_clusters(cfg, data)
    cov["phage_cluster"] = cov["phage"].map(pc).astype(int)
    cov["host_cluster"] = cov["host"].map(hc).astype(int)
    fp = _phage_fingerprints(data, cfg)

    # reuse the aligned GBM predictions saved by step 12
    gbm_npz = np.load(rd / "phist_pooled_preds.npz", allow_pickle=True)

    mpos, mneg = cfg["data"]["min_pos_per_group"], cfg["data"]["min_neg_per_group"]
    sp = cfg["splits"]
    regimes = {
        "loso_species": list(leave_one_group_out(cov, "host_species", "loso", mpos, mneg)),
        "host_cluster": list(leave_one_group_out(cov, "host_cluster", "host_cluster", mpos, mneg)),
        "phage_cluster": list(leave_one_group_out(cov, "phage_cluster", "phage_cluster", mpos, mneg)),
        "combined_unseen": list(combined_unseen_folds(cov, "phage_cluster", "host_cluster",
                                                      sp["n_combined_splits"], seed, mpos, mneg)),
    }

    rows, cmp_rows, cmp_pvals, preds_out = [], [], [], {}
    for rname, folds in regimes.items():
        if not folds:
            continue
        pooled_idx = np.concatenate([f.test_idx for f in folds])
        y = cov["label"].to_numpy()[pooled_idx].astype(int)
        if not np.array_equal(y, gbm_npz[f"{rname}__y"].astype(int)):
            raise AssertionError(f"{rname}: pooled order differs from step 12")
        gbm = gbm_npz[f"{rname}__GBM"].astype(float)
        oof = _rafah_oof(cov, fp, folds, seed)
        rafah = oof[pooled_idx]
        if np.isnan(rafah).any():
            rafah = np.nan_to_num(rafah, nan=0.0)
        preds_out[f"{rname}__y"] = y
        preds_out[f"{rname}__GBM"] = gbm
        preds_out[f"{rname}__RaFAH_style"] = rafah

        for mdl, sc in (("GBM", gbm), ("RaFAH_style", rafah)):
            ci = delong_auc_ci(y, sc, cfg["eval"]["bootstrap_ci"])
            perm = permutation_auc_test(y, sc, cfg["eval"]["permutation_n"], seed)
            rows.append({"regime": rname, "model": mdl, "n": len(y),
                         "auc": ci["auc"], "auc_lo": ci["lo"], "auc_hi": ci["hi"],
                         "auprc": average_precision_score(y, sc), "perm_p": perm["p"]})
        dt = delong_test(y, gbm, rafah)
        bd = bootstrap_auc_diff(y, gbm, rafah, cfg["eval"]["bootstrap_n"],
                                cfg["eval"]["bootstrap_ci"], seed)
        cmp_rows.append({"regime": rname, "auc_gbm": dt["auc1"],
                         "auc_rafah": dt["auc2"], "auc_diff": dt["auc_diff"],
                         "diff_lo": bd["lo"], "diff_hi": bd["hi"],
                         "delong_z": dt["z"], "delong_p": dt["p"]})
        cmp_pvals.append(dt["p"])
        log.info("[%s] n=%d  GBM AUC=%.3f  RaFAH-style AUC=%.3f  diff=%.3f",
                 rname, len(y), dt["auc1"], dt["auc2"], dt["auc_diff"])

    for row, q in zip(rows, benjamini_hochberg([r["perm_p"] for r in rows])):
        row["perm_q_bh"] = float(q)
    for row, q in zip(cmp_rows, benjamini_hochberg(cmp_pvals)):
        row["delong_q_bh"] = float(q)
        row["gbm_beats_rafah_fdr05"] = bool(q < 0.05 and row["auc_diff"] > 0)

    skill = pd.DataFrame(rows)
    cmp = pd.DataFrame(cmp_rows)
    skill.to_csv(rd / "rafah_skill.csv", index=False)
    cmp.to_csv(rd / "rafah_modelcmp.csv", index=False)
    np.savez(rd / "rafah_pooled_preds.npz", **preds_out)
    (rd / "rafah_compare.json").write_text(json.dumps(
        {"note": "RaFAH-style in-house reimplementation; NOT the pretrained RaFAH "
                 "(R runtime absent; model host figshare network-blocked).",
         "fp_dim": FP_DIM, "protein_k": PROT_K, "min_pep": MIN_PEP,
         "skill": rows, "comparison": cmp_rows}, indent=2, default=float))
    log.info("SKILL (GBM vs RaFAH-style):\n%s", skill.round(4).to_string(index=False))
    log.info("COMPARISON (DeLong, BH):\n%s", cmp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
