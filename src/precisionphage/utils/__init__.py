"""Shared utilities: config loading, logging, determinism."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    """Load YAML config and resolve paths relative to the config file.

    Resolving ``paths.root`` against the process working directory made an
    otherwise identical command select different data when launched elsewhere.
    """
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    configured_root = Path(cfg["paths"]["root"])
    root = (configured_root if configured_root.is_absolute()
            else path.parent / configured_root).resolve()
    cfg["paths"]["root"] = root
    for key, val in cfg["paths"].items():
        if key == "root":
            continue
        cfg["paths"][key] = (root / val).resolve()
    return cfg


def get_logger(name: str = "precisionphage") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def set_determinism(seed: int) -> None:
    """Set all RNG seeds and enable deterministic torch where possible."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dirs(cfg: dict) -> None:
    for key in ("interim_dir", "results_dir", "plots_dir", "cache_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def limit_threads(n: int = 1) -> None:
    """Pin BLAS/torch to `n` threads in the current process.

    Used inside worker processes so that n_jobs processes * threads_per_job
    stays within the core budget (no oversubscription / thrashing)."""
    n = max(1, int(n))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    os.environ["PP_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except Exception:
        pass


def resolve_n_jobs(cfg: dict, n_items: int) -> int:
    """Worker count: min(config n_jobs, hard cap 10, #items, #cpus)."""
    j = int(cfg.get("compute", {}).get("n_jobs", 1))
    cpus = os.cpu_count() or 1
    return max(1, min(j, 10, n_items, cpus))
