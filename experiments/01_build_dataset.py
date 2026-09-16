#!/usr/bin/env python3
"""Step 1: build the canonical experimental interaction dataset and report
genome-linking coverage. Verifies the data foundation before any modeling.

Run:
  env -u PYTHONHOME -u PYTHONPATH .venv/bin/python experiments/01_build_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precisionphage.data import GenomeIndex, load_interactions  # noqa: E402
from precisionphage.utils import ensure_dirs, get_logger, load_config, set_determinism  # noqa: E402

log = get_logger("build_dataset")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_determinism(cfg["seed"])
    ensure_dirs(cfg)

    ds = load_interactions(cfg)
    summary = ds.summary()
    log.info("Dataset summary: %s", json.dumps(summary, indent=2, default=str))

    # Genome linking
    phage_idx = GenomeIndex([cfg["paths"]["phage_fasta_dir"]],
                            cache_path=cfg["paths"]["cache_dir"] / "phage_resolution.json")
    host_idx = GenomeIndex([cfg["paths"]["host_fasta_dir"]],
                           cache_path=cfg["paths"]["cache_dir"] / "host_resolution.json")

    phages = sorted(ds.df["phage"].unique().tolist())
    hosts = sorted(ds.df["host"].unique().tolist())
    pcov = phage_idx.coverage(phages)
    hcov = host_idx.coverage(hosts)
    log.info("[coverage] phages resolved: %d/%d (%.1f%%)",
             pcov["resolved"], pcov["n"], 100 * pcov["fraction"])
    log.info("[coverage] hosts  resolved: %d/%d (%.1f%%)",
             hcov["resolved"], hcov["n"], 100 * hcov["fraction"])

    # How many pairs have BOTH genomes (the modelable subset)
    phset = {p for p in phages if phage_idx.resolve(p) is not None}
    hset = {h for h in hosts if host_idx.resolve(h) is not None}
    both = ds.df[ds.df["phage"].isin(phset) & ds.df["host"].isin(hset)]
    log.info("[coverage] pairs with BOTH genomes: %d/%d (pos=%d neg=%d)",
             len(both), len(ds.df), int((both["label"] == 1).sum()),
             int((both["label"] == 0).sum()))
    log.info("[coverage] modelable by study:\n%s",
             both["study"].value_counts().to_string())

    phage_idx.save_resolution()
    host_idx.save_resolution()

    # Persist canonical artifacts
    interim = cfg["paths"]["interim_dir"]
    ds.df.to_parquet(interim / "interactions.parquet") if _has_parquet() \
        else ds.df.to_csv(interim / "interactions.csv", index=False)
    both.to_csv(interim / "interactions_modelable.csv", index=False)
    (interim / "dataset_summary.json").write_text(
        json.dumps({"summary": summary,
                    "phage_coverage": {k: v for k, v in pcov.items() if k != "missing"},
                    "host_coverage": {k: v for k, v in hcov.items() if k != "missing"},
                    "n_modelable": int(len(both))}, indent=2, default=str),
        encoding="utf-8")
    log.info("Wrote canonical artifacts to %s", interim)


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
