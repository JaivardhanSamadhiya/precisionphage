"""Load real experimental phage-host interactions into a canonical dataset.

Source: data/raw/VirusHostInter.csv — 8,849 experimentally assayed pairs with
true Inf/NoInf labels across three studies (NahantCollection, NCBI_HR,
StaphStudy). Unlike v1, NO negatives are constructed: the negatives here are
real experimental NoInf results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger
from .naming import clean_name, genus_of, species_of

log = get_logger(__name__)

# Precomputed pairwise features supplied by VirusHostInter.csv. They are used by
# the full-table baseline and included in the saved 24-feature sequence model;
# this repository does not regenerate these four columns.
_VHI_PAIR_FEATS = ["k3dist", "k6dist", "GCdiff", "Homology"]


@dataclass
class InteractionDataset:
    """Canonical interaction table plus provenance metadata."""
    df: pd.DataFrame
    studies: list[str]
    n_conflicts: int = 0
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        d = self.df
        return {
            "n_pairs": int(len(d)),
            "n_pos": int((d["label"] == 1).sum()),
            "n_neg": int((d["label"] == 0).sum()),
            "n_phages": int(d["phage"].nunique()),
            "n_hosts": int(d["host"].nunique()),
            "n_host_species": int(d["host_species"].nunique()),
            "n_host_genera": int(d["host_genus"].nunique()),
            "by_study": d["study"].value_counts().to_dict(),
            "n_conflicts": int(self.n_conflicts),
        }


def _detect_columns(df: pd.DataFrame) -> dict:
    lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None
    return {
        "host": pick("hostname", "host", "host_name", "bacterium"),
        "phage": pick("phagename", "phage", "phage_name", "virus"),
        "infection": pick("infection", "label", "outcome", "interacts"),
        "study": pick("data", "study", "source", "dataset"),
    }


def load_interactions(cfg: dict) -> InteractionDataset:
    """Read VirusHostInter.csv, keep configured studies, return canonical table.

    Canonical columns: phage, host, host_species, host_genus, label (1/0),
    study, plus the precomputed VHI pair features when present.
    """
    path = Path(cfg["paths"]["vhi_csv"])
    if not path.exists():
        raise FileNotFoundError(f"VirusHostInter.csv not found at {path}")
    raw = pd.read_csv(path)
    cols = _detect_columns(raw)
    for key in ("host", "phage", "infection"):
        if cols[key] is None:
            raise ValueError(f"Could not detect '{key}' column in {path.name}; "
                             f"columns were {list(raw.columns)}")

    pos_tokens = set(t.lower() for t in cfg["data"]["positive_tokens"])
    neg_tokens = set(t.lower() for t in cfg["data"]["negative_tokens"])
    keep_studies = set(cfg["data"]["studies"])

    phage_raw = raw[cols["phage"]].where(raw[cols["phage"]].notna(), "")
    host_raw = raw[cols["host"]].where(raw[cols["host"]].notna(), "")
    infection = (raw[cols["infection"]].where(raw[cols["infection"]].notna(), "")
                 .astype(str).str.strip().str.lower())
    unknown = sorted(set(infection) - pos_tokens - neg_tokens)
    if unknown:
        raise ValueError(f"Unrecognized infection labels: {unknown[:10]}")

    out = pd.DataFrame({
        "phage": phage_raw.astype(str).map(clean_name),
        "host": host_raw.astype(str).map(clean_name),
        "study": (raw[cols["study"]].astype(str) if cols["study"]
                  else "unknown"),
    })
    out["label"] = infection.isin(pos_tokens).astype(int)
    for c in _VHI_PAIR_FEATS:
        if c in raw.columns:
            out[c] = pd.to_numeric(raw[c], errors="coerce")

    # filter
    out = out[(out["phage"] != "") & (out["host"] != "")]
    if cols["study"] and keep_studies:
        before = len(out)
        out = out[out["study"].isin(keep_studies)]
        log.info("[load] kept %d/%d rows from studies %s",
                 len(out), before, sorted(keep_studies))

    out["host_species"] = out["host"].map(species_of)
    out["host_genus"] = out["host"].map(genus_of)

    # Resolve duplicate (phage, host) pairs. Conflicts = same pair with both
    # Inf and NoInf across rows/studies. We resolve conservatively: a pair is
    # positive if ANY assay observed infection (infection is the harder-to-fake
    # observation), and we count conflicts for transparency.
    grp = out.groupby(["phage", "host"], sort=False)
    label_max = grp["label"].transform("max")
    label_min = grp["label"].transform("min")
    n_conflicts = int(((label_max != label_min)
                       .groupby([out["phage"], out["host"]]).first().sum()))
    out["label"] = label_max
    dedup = (out.sort_values("label", ascending=False)
             .drop_duplicates(["phage", "host"], keep="first")
             .reset_index(drop=True))

    log.info("[load] canonical pairs=%d pos=%d neg=%d phages=%d hosts=%d "
             "species=%d genera=%d conflicts=%d",
             len(dedup), int((dedup["label"] == 1).sum()),
             int((dedup["label"] == 0).sum()), dedup["phage"].nunique(),
             dedup["host"].nunique(), dedup["host_species"].nunique(),
             dedup["host_genus"].nunique(), n_conflicts)

    ds = InteractionDataset(df=dedup, studies=sorted(out["study"].unique()),
                            n_conflicts=n_conflicts)
    return ds
