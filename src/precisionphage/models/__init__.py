"""Models: feature-based baselines and the GNN encoder + Edge-MLP decoder."""
from .baseline import (
    EdgeMLP, fit_predict_gbm, fit_predict_gbm_tuned, fit_predict_mlp,
    run_grouped_cv,
)

__all__ = ["fit_predict_gbm", "fit_predict_gbm_tuned", "EdgeMLP",
           "fit_predict_mlp", "run_grouped_cv", "run_gnn_cv", "BipartiteSAGE"]


def __getattr__(name):
    # Lazy import so torch_geometric is only required when the GNN is used.
    if name in ("run_gnn_cv", "BipartiteSAGE"):
        from . import gnn
        return getattr(gnn, name)
    raise AttributeError(name)
