"""Name normalization for hosts and phages.

We deliberately preserve strain-level identity (the Nahant Vibrio collection is
strain-resolved) while exposing species- and genus-level groupings for the
leakage-safe split regimes. We do NOT strip host tokens from phage names here;
phage-name text is never used as a model feature (see DESIGN.md), so taxonomic
name leakage cannot enter through features.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9]+")
# strain/serovar qualifiers used only when collapsing to species
_STRAIN_QUALIFIERS = re.compile(
    r"\b(strain|str|subsp|ssp|serovar|sv|pv|biovar|bv)\b.*$", re.IGNORECASE)


def clean_name(name: str) -> str:
    """Canonical display/ID form: lowercase, underscores->spaces, collapsed."""
    if not isinstance(name, str):
        return ""
    n = name.strip().lower().replace("_", " ")
    n = _WS.sub(" ", n).strip()
    return n


def genus_of(host: str) -> str:
    h = clean_name(host)
    return h.split(" ", 1)[0] if h else ""


def species_of(host: str) -> str:
    """Genus + species epithet, dropping strain qualifiers and 'sp.' noise."""
    h = clean_name(host)
    if not h:
        return ""
    h = _STRAIN_QUALIFIERS.sub("", h).strip()
    parts = h.split(" ")
    if len(parts) >= 2 and parts[1] not in ("sp.", "sp"):
        return f"{parts[0]} {parts[1]}"
    return parts[0]


def slugify(name: str) -> str:
    """Filesystem-safe slug for FASTA matching."""
    n = clean_name(name)
    return _NONALNUM.sub("_", n).strip("_")
