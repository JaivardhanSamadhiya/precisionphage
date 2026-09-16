"""Content-addressed caching for sequence-cluster assignments."""
from __future__ import annotations

import json

from ..data import genome_set_digest
from ..utils import get_logger
from .cluster import build_clusters, sketch_entities

log = get_logger(__name__)


def load_or_build_clusters(cfg, data):
    """Load clusters only when the exact FASTA contents and knobs match."""
    split_cfg = cfg["splits"]
    cache = (cfg["paths"]["cache_dir"] /
             f"clusters_k{split_cfg['mash_k']}_d{split_cfg['mash_max_distance']}.json")
    meta = {
        "method": "forward-strand bottom-k MinHash; Mash-form distance; single linkage",
        "k": int(split_cfg["mash_k"]),
        "num": int(split_cfg["minhash_num"]),
        "max_distance": float(split_cfg["mash_max_distance"]),
        "phage_source_sha256": genome_set_digest(data.phages, data.phage_index),
        "host_source_sha256": genome_set_digest(data.hosts, data.host_index),
    }
    if cache.exists():
        obj = json.loads(cache.read_text(encoding="utf-8"))
        if (obj.get("meta") == meta
                and len(obj.get("phage", {})) == len(data.phages)
                and len(obj.get("host", {})) == len(data.hosts)):
            log.info("[clusters] loaded content-matched cache %s", cache.name)
            return obj["phage"], obj["host"]
        log.info("[clusters] cache invalidated: FASTA contents or settings changed")

    p_sketch = sketch_entities(
        data.phages, data.phage_index, split_cfg["mash_k"],
        split_cfg["minhash_num"], cfg["features"]["n_workers"])
    h_sketch = sketch_entities(
        data.hosts, data.host_index, split_cfg["mash_k"],
        split_cfg["minhash_num"], cfg["features"]["n_workers"])
    phage = build_clusters(
        data.phages, p_sketch, split_cfg["mash_max_distance"],
        split_cfg["mash_k"], split_cfg["minhash_num"])
    host = build_clusters(
        data.hosts, h_sketch, split_cfg["mash_max_distance"],
        split_cfg["mash_k"], split_cfg["minhash_num"])
    cache.write_text(json.dumps({"meta": meta, "phage": phage, "host": host}, indent=2),
                     encoding="utf-8")
    log.info("[clusters] wrote content-addressed cache %s", cache.name)
    return phage, host
