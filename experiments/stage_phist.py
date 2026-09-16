#!/usr/bin/env python3
"""Stage covered phage/host genomes as symlink dirs for PHIST, and record the
mapping from PHIST genome IDs (FASTA filenames) back to our entity names."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precisionphage.features.assembly import build_covered_dataset  # noqa: E402
from precisionphage.utils import get_logger, load_config  # noqa: E402

log = get_logger("stage_phist")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data = build_covered_dataset(cfg)
    out = ROOT / "external" / "phist_run"
    pdir, hdir = out / "phages", out / "hosts"
    source_dirs = (cfg["paths"]["phage_fasta_dir"].resolve(),
                   cfg["paths"]["host_fasta_dir"].resolve())
    target_dirs = (pdir.resolve(), hdir.resolve())
    in_place = source_dirs == target_dirs
    for d in (pdir, hdir):
        d.mkdir(parents=True, exist_ok=True)
        if in_place:
            continue
        # Only links created by this staging script are disposable. Never
        # unlink regular FASTAs, which may be frozen analysis inputs.
        for f in d.iterdir():
            if f.is_symlink():
                f.unlink()

    phage_map, host_map = {}, {}     # fasta filename -> entity name
    for name in data.phages:
        p = data.phage_index.resolve(name)
        if p is None:
            continue
        link = pdir / p.name
        if not link.exists() and not in_place:
            link.symlink_to(p.resolve())
        phage_map[p.name] = name
    for name in data.hosts:
        p = data.host_index.resolve(name)
        if p is None:
            continue
        link = hdir / p.name
        if not link.exists() and not in_place:
            link.symlink_to(p.resolve())
        host_map.setdefault(p.name, []).append(name)

    (out / "phage_map.json").write_text(json.dumps(phage_map, indent=2))
    (out / "host_map.json").write_text(json.dumps(host_map, indent=2))
    log.info("staged %d phage files, %d host files -> %s",
             len(list(pdir.iterdir())), len(list(hdir.iterdir())), out)
    log.info("phage_map=%d ids, host_map=%d ids (some map to multiple strains)",
             len(phage_map), len(host_map))


if __name__ == "__main__":
    main()
