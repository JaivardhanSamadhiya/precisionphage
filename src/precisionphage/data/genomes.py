"""Resolve phage/host names to genome FASTA files and load sequences.

Pairs whose phage or host genome cannot be resolved are reported and excluded
from sequence-based models (never zero-imputed, which silently corrupts metrics).
A resolution map is cached so linking is deterministic and auditable.
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

from ..utils import get_logger
from .naming import slugify

log = get_logger(__name__)

_VALID = set("ACGTN")
_FASTA_EXT = (".fasta", ".fa", ".fna", ".fasta.gz", ".fna.gz")


def genome_set_digest(names, genome_index) -> str:
    """SHA-256 over ordered entity names and exact resolved FASTA bytes."""
    key = tuple(str(name) for name in names)
    cache = getattr(genome_index, "_content_digest_cache", {})
    if key in cache:
        return cache[key]
    digest = hashlib.sha256()
    for name in key:
        path = genome_index.resolve(name)
        if path is None:
            raise FileNotFoundError(f"missing genome while hashing input: {name}")
        digest.update(str(name).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    value = digest.hexdigest()
    cache[key] = value
    genome_index._content_digest_cache = cache
    return value


def _read_fasta(path: Path) -> str | None:
    try:
        if path.suffix == ".gz":
            import gzip
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    seq = "".join(line.strip() for line in text.splitlines()
                  if line and not line.startswith(">"))
    seq = "".join(c for c in seq.upper().replace("U", "T") if c in _VALID)
    return seq if len(seq) >= 200 else None


class GenomeIndex:
    """Index FASTA directories and resolve entity names to files."""

    def __init__(self, fasta_dirs: list[Path], cache_path: Path | None = None):
        self.fasta_dirs = [Path(d) for d in fasta_dirs if Path(d).exists()]
        self.cache_path = Path(cache_path) if cache_path else None
        self._slug_to_path: dict[str, Path] = {}
        self._build_index()
        self._resolution: dict[str, str] = {}

    def _build_index(self) -> None:
        for d in self.fasta_dirs:
            for p in d.iterdir():
                if p.is_file() and p.name.lower().endswith(_FASTA_EXT):
                    stem = p.name
                    for ext in _FASTA_EXT:
                        if stem.lower().endswith(ext):
                            stem = stem[: -len(ext)]
                            break
                    file_slug = slugify(stem)
                    self._slug_to_path.setdefault(file_slug, p)
                    # NCBI accessions in the interaction table omit sequence
                    # versions (for example ``nc 000866``), while FASTA names
                    # retain them (``NC_000866.4``). Add this deterministic
                    # accession alias instead of falling back to fuzzy matching.
                    match = re.fullmatch(r"(n[cmrtwz]_\d+)_\d+", file_slug)
                    if match:
                        self._slug_to_path.setdefault(match.group(1), p)
        log.info("[genomes] indexed %d FASTA files across %d dir(s)",
                 len(self._slug_to_path), len(self.fasta_dirs))

    def resolve(self, name: str) -> Path | None:
        """Return the FASTA path for a name, or ``None``.

        Resolution uses only exact normalized names and explicit accession
        aliases created while indexing. Substring matching is intentionally not
        used because it can silently associate the wrong genome.
        """
        slug = slugify(name)
        if not slug:
            return None
        if slug in self._slug_to_path:
            self._resolution[name] = str(self._slug_to_path[slug])
            return self._slug_to_path[slug]
        return None

    def coverage(self, names: list[str]) -> dict:
        resolved = {n: self.resolve(n) for n in names}
        n_ok = sum(1 for v in resolved.values() if v is not None)
        return {"n": len(names), "resolved": n_ok,
                "fraction": (n_ok / len(names) if names else 0.0),
                "missing": [n for n, v in resolved.items() if v is None]}

    def load_sequence(self, name: str) -> str | None:
        path = self.resolve(name)
        return _read_fasta(path) if path is not None else None

    def save_resolution(self) -> None:
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            portable = {}
            # cache files live at <root>/data/cache_v2/*.json
            project_root = self.cache_path.resolve().parents[2]
            for name, raw_path in self._resolution.items():
                path = Path(raw_path)
                try:
                    portable[name] = path.resolve().relative_to(
                        project_root).as_posix()
                except ValueError:
                    portable[name] = path.as_posix()
            self.cache_path.write_text(json.dumps(portable, indent=2),
                                       encoding="utf-8")
            log.info("[genomes] wrote resolution map (%d entries) -> %s",
                     len(self._resolution), self.cache_path.name)
