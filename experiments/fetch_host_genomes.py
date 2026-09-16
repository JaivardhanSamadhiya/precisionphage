#!/usr/bin/env python3
"""Fetch host reference genomes into data/fastas/hosts/ for reproducibility.

Uses species names from data/cache_v2/host_resolution.json (or unique hosts in
the VHI table) and downloads one RefSeq representative assembly per species from
NCBI. Writes data/raw/host_genome_manifest.json mapping host -> accession.

Run:
  python experiments/fetch_host_genomes.py
  python experiments/fetch_host_genomes.py --limit 10   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precisionphage.data.naming import slugify  # noqa: E402
from precisionphage.utils import get_logger, load_config  # noqa: E402

log = get_logger("fetch_hosts")


def _species_list(cfg: dict) -> list[str]:
    cache = cfg["paths"]["cache_dir"] / "host_resolution.json"
    if cache.exists():
        obj = json.loads(cache.read_text(encoding="utf-8"))
        return sorted(obj.keys())
    import pandas as pd
    df = pd.read_csv(cfg["paths"]["vhi_csv"])
    col = "hostname" if "hostname" in df.columns else "host"
    names = df[col].astype(str).str.lower().str.replace("_", " ", regex=False)
    return sorted({n.strip() for n in names if n.strip()})


def _esearch(Entrez, db: str, term: str, retmax: int = 1, sleep: float = 0.34):
    for attempt in range(5):
        try:
            with Entrez.esearch(db=db, term=term, retmax=retmax) as h:
                return Entrez.read(h).get("IdList", [])
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 2 ** (attempt + 1)
                log.warning("NCBI rate limit; sleeping %ds", wait)
                time.sleep(wait)
                continue
            raise
        finally:
            time.sleep(sleep)


def _pick_assembly(Entrez, species: str, sleep: float) -> str | None:
    queries = [
        f'"{species}"[Organism] AND latest[filter] AND refseq[filter]',
        f"{species}[Organism] AND refseq[filter] AND complete genome[Title]",
        f"{species}[Organism] AND refseq[filter]",
    ]
    for term in queries:
        ids = _esearch(Entrez, "assembly", term, sleep=sleep)
        if ids:
            with Entrez.esummary(db="assembly", id=ids[0]) as h:
                rec = Entrez.read(h)["DocumentSummarySet"]["DocumentSummary"][0]
            acc = rec.get("AssemblyAccession") or rec.get("Accession")
            if acc:
                return str(acc)
            time.sleep(sleep)
    return None


def _fetch_fasta(Entrez, species: str, assembly_acc: str | None, sleep: float) -> str | None:
    if assembly_acc:
        ids = _esearch(Entrez, "nuccore",
                       f"{assembly_acc}[Assembly] AND refseq[filter]", sleep=sleep)
        if ids:
            with Entrez.efetch(db="nuccore", id=ids[0], rettype="fasta",
                               retmode="text") as h:
                text = h.read()
            if text and len(text) > 500:
                return text
            time.sleep(sleep)
    # Fallback: representative complete RefSeq genome by organism name
    ids = _esearch(Entrez, "nuccore",
                   f'"{species}"[Organism] AND refseq[filter] AND complete genome[Title]',
                   sleep=sleep)
    if not ids:
        ids = _esearch(Entrez, "nuccore",
                       f"{species}[Organism] AND refseq[filter]", sleep=sleep)
    if not ids:
        return None
    with Entrez.efetch(db="nuccore", id=ids[0], rettype="fasta", retmode="text") as h:
        text = h.read()
    return text if text and len(text) > 500 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max species to fetch (0=all)")
    ap.add_argument("--sleep", type=float, default=1.0, help="NCBI rate limit (s)")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    from Bio import Entrez
    Entrez.email = cfg.get("fetch", {}).get("ncbi_email", "research@example.com")

    out_dir = Path(cfg["paths"]["host_fasta_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg["paths"]["raw_dir"] / "host_genome_manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    species = _species_list(cfg)
    if args.limit:
        species = species[: args.limit]
    log.info("Host species to resolve: %d -> %s", len(species), out_dir)

    n_have = n_new = n_fail = 0
    for i, sp in enumerate(species):
        slug = slugify(sp)
        target = out_dir / f"{slug}.fasta"
        if target.exists() and target.stat().st_size > 1000:
            n_have += 1
            continue
        try:
            acc = manifest.get(sp, {}).get("assembly")
            if not acc:
                acc = _pick_assembly(Entrez, sp, args.sleep)
            fasta = _fetch_fasta(Entrez, sp, acc, args.sleep)
            if not fasta:
                n_fail += 1
                log.warning("efetch failed for %s (%s)", sp, acc)
                continue
            target.write_text(fasta, encoding="utf-8")
            manifest[sp] = {"assembly": acc, "file": str(target.relative_to(ROOT))}
            n_new += 1
        except Exception as e:
            n_fail += 1
            log.warning("failed %s: %s", sp, e)
        if (i + 1) % 25 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            log.info("progress %d/%d (new=%d have=%d fail=%d)",
                     i + 1, len(species), n_new, n_have, n_fail)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("DONE hosts: new=%d already=%d fail=%d total=%d manifest=%s",
             n_new, n_have, n_fail, len(species), manifest_path)


if __name__ == "__main__":
    main()
