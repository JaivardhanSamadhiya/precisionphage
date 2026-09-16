#!/usr/bin/env python3
"""Report held-out-source performance for the full-table four-feature audit."""
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precisionphage.data import load_interactions  # noqa: E402
from precisionphage.eval import calibration_curve_ece  # noqa: E402
from precisionphage.models import fit_predict_gbm, fit_predict_mlp, run_grouped_cv  # noqa: E402
from precisionphage.splits import cross_study_folds  # noqa: E402
from precisionphage.utils import ensure_dirs, load_config, set_determinism  # noqa: E402


EDGE_FEATS = ["k3dist", "k6dist", "GCdiff", "Homology"]


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    seed = int(cfg["seed"])
    set_determinism(seed)
    ensure_dirs(cfg)
    df = load_interactions(cfg).df.reset_index(drop=True)
    X = df[EDGE_FEATS].to_numpy(dtype=np.float32)
    folds = list(cross_study_folds(
        df, "study", cfg["data"]["min_pos_per_group"],
        cfg["data"]["min_neg_per_group"]))

    rows = []
    for model_name, model_fn in (("GBM", fit_predict_gbm),
                                 ("EdgeMLP", fit_predict_mlp)):
        # Match experiment 02 exactly: sequential folds with one model thread.
        result = run_grouped_cv(df, X, folds, model_fn, seed)
        offset = 0
        for fold in folds:
            n = len(fold.test_idx)
            y = result["pooled_y"][offset:offset + n]
            p = result["pooled_p"][offset:offset + n]
            offset += n
            held_out = fold.name
            rows.append({
                "held_out_source": held_out,
                "model": model_name,
                "n_pairs": int(n),
                "n_positive": int(y.sum()),
                "positive_fraction": float(y.mean()),
                "auroc": float(roc_auc_score(y, p)),
                "auprc": float(average_precision_score(y, p)),
                "ece": float(calibration_curve_ece(y, p, cfg["eval"]["calibration_bins"])["ece"]),
            })
        if offset != len(result["pooled_y"]):
            raise AssertionError("cross-study fold reconstruction did not consume all predictions")

    detail = pd.DataFrame(rows)
    out_dir = cfg["paths"]["results_dir"]
    detail.to_csv(out_dir / "cross_study_detail.csv", index=False)
    summary = {
        "features": EDGE_FEATS,
        "interpretation": (
            "Leave-one-study-out audit of the four features available across all sources; "
            "not a cross-source test of the 24-feature sequence model."
        ),
        "rows": detail.to_dict(orient="records"),
    }
    (out_dir / "cross_study_detail.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
