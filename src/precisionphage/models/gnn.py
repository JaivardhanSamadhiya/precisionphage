"""Inductive bipartite GraphSAGE encoder + Edge-MLP decoder.

Design (see DESIGN.md §4):
  * Nodes: phages [0, P) and hosts [P, P+H). Node features are genomic
    (k-mer/codon/GC), reduced by PCA fit on TRAIN nodes only.
  * Encoder: GraphSAGE message passing over TRAIN-POSITIVE edges only
    (both directions). Inductive: an unseen node with no training edges still
    gets an embedding from its own features (fixes v1's isolated-node failure).
  * Decoder: MLP over [z_phage ‖ z_host ‖ edge_features]. Because the decoder
    always sees genomic pairwise edge features, predictions are informative even
    for graph-isolated nodes; the graph *adds* relational signal.

Leakage guarantees: PCA, edge scaling, and the early-stopping graph are derived
from the inner-training partition only, so inner-validation rows do not enter
preprocessing or message passing.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..eval.metrics import aggregate_folds, binary_metrics
from ..utils import get_logger, limit_threads, resolve_n_jobs

log = get_logger(__name__)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.nn import SAGEConv  # noqa: E402


class BipartiteSAGE(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 128,
                 embed: int = 64, n_layers: int = 2, dropout: float = 0.3,
                 dec_hidden: int = 128):
        super().__init__()
        self.p_proj = nn.Linear(in_dim, hidden)
        self.h_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        dims = [hidden] * (n_layers - 1) + [embed]
        prev = hidden
        for d in dims:
            self.convs.append(SAGEConv(prev, d))
            prev = d
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(embed * 2 + edge_dim, dec_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden, dec_hidden // 2), nn.ReLU(),
            nn.Linear(dec_hidden // 2, 1),
        )

    def encode(self, x_p, x_h, edge_index):
        x = torch.cat([self.p_proj(x_p), self.h_proj(x_h)], dim=0)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        return x

    def decode(self, z, src, dst, edge_feats, n_p):
        zp = z[src]
        zh = z[dst + n_p]
        return self.decoder(torch.cat([zp, zh, edge_feats], dim=1)).squeeze(-1)

    def forward(self, x_p, x_h, edge_index, src, dst, edge_feats, n_p):
        z = self.encode(x_p, x_h, edge_index)
        return self.decode(z, src, dst, edge_feats, n_p)


def _fit_fold(P_raw, H_raw, E_raw, pidx, hidx, y, tr, te, n_p, cfg, seed,
              use_graph: bool = True):
    """Train one fold; return test probabilities.

    `use_graph=False` runs the identical architecture with an empty edge set so
    SAGEConv keeps only its self-transform (W_root x) -> a pure MLP on
    node+edge features. This isolates the contribution of message passing."""
    torch.manual_seed(seed)
    mcfg = cfg["model"]

    # Split before fitting any preprocessing or graph construction so the
    # inner validation rows are genuinely held out for early stopping and
    # calibration. Stratification prevents accidental single-class validation.
    ti, vi = train_test_split(
        np.asarray(tr), test_size=mcfg["val_frac"], random_state=seed,
        stratify=y[tr])

    # PCA on node features, fit on INNER-TRAIN nodes only
    train_phages = np.unique(pidx[ti])
    train_hosts = np.unique(hidx[ti])
    pdim = min(cfg["features"]["pca_phage_dim"],
               P_raw.shape[1], max(2, len(train_phages) - 1))
    hdim = min(cfg["features"]["pca_host_dim"],
               H_raw.shape[1], max(2, len(train_hosts) - 1))
    p_pca = PCA(n_components=pdim, random_state=seed).fit(P_raw[train_phages])
    h_pca = PCA(n_components=hdim, random_state=seed).fit(H_raw[train_hosts])
    P = p_pca.transform(P_raw).astype(np.float32)
    H = h_pca.transform(H_raw).astype(np.float32)
    # pad to equal in_dim so a single projection space is consistent
    in_dim = max(P.shape[1], H.shape[1])
    P = np.pad(P, ((0, 0), (0, in_dim - P.shape[1])))
    H = np.pad(H, ((0, 0), (0, in_dim - H.shape[1])))
    p_scaler = StandardScaler().fit(P[train_phages])
    h_scaler = StandardScaler().fit(H[train_hosts])
    P = np.nan_to_num(p_scaler.transform(P)).astype(np.float32)
    H = np.nan_to_num(h_scaler.transform(H)).astype(np.float32)

    # edge feature scaling on inner-training rows only
    e_scaler = StandardScaler().fit(E_raw[ti])
    E = np.nan_to_num(e_scaler.transform(E_raw)).astype(np.float32)

    x_p = torch.tensor(P)
    x_h = torch.tensor(H)
    src_all = torch.tensor(pidx, dtype=torch.long)
    dst_all = torch.tensor(hidx, dtype=torch.long)
    e_all = torch.tensor(E)
    y_all = torch.tensor(y.astype(np.float32))

    ti_t = torch.tensor(ti, dtype=torch.long)
    vi_t = torch.tensor(vi, dtype=torch.long)
    te_t = torch.tensor(te, dtype=torch.long)

    # Early-stopping graph = INNER-TRAIN positive edges only. Building this
    # from all outer-training positives would expose inner-validation labels to
    # the encoder and invalidate early stopping and isotonic calibration.
    if use_graph:
        pos_ti = ti[y[ti] == 1]
        src = pidx[pos_ti]
        dst = hidx[pos_ti] + n_p
        ei = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
        edge_index = torch.tensor(ei, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    model = BipartiteSAGE(in_dim=in_dim, edge_dim=E.shape[1],
                          hidden=mcfg["hidden_dim"], embed=mcfg["embed_dim"],
                          n_layers=mcfg["n_layers"], dropout=mcfg["dropout"],
                          dec_hidden=mcfg["edge_mlp_hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"],
                           weight_decay=mcfg["weight_decay"])
    pos_w = float((y[ti] == 0).sum()) / max(1.0, float((y[ti] == 1).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))

    best_auc, best_state, bad = -np.inf, None, 0
    for ep in range(mcfg["epochs"]):
        model.train()
        opt.zero_grad()
        logits = model(x_p, x_h, edge_index, src_all[ti_t], dst_all[ti_t],
                       e_all[ti_t], n_p)
        loss = loss_fn(logits, y_all[ti_t])
        loss.backward()
        opt.step()
        if (ep + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(x_p, x_h, edge_index, src_all[vi_t],
                                         dst_all[vi_t], e_all[vi_t], n_p)).numpy()
            yv = y[vi]
            vauc = roc_auc_score(yv, vp) if len(np.unique(yv)) > 1 else 0.5
            if vauc > best_auc + 1e-4:
                best_auc, best_state, bad = vauc, deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= mcfg["patience"]:
                    break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_p, x_h, edge_index, src_all[te_t],
                                    dst_all[te_t], e_all[te_t], n_p)).numpy()
        vp = torch.sigmoid(model(x_p, x_h, edge_index, src_all[vi_t],
                                 dst_all[vi_t], e_all[vi_t], n_p)).numpy()
    probs = probs.astype(np.float32)
    # post-hoc isotonic calibration on the held-out inner-val split; the
    # order-preserving epsilon keeps the test ranking (AUC) unchanged.
    if len(np.unique(y[vi])) == 2:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(vp, y[vi])
        cal = iso.predict(probs).astype(np.float32) + np.float32(1e-6) * probs
        if np.unique(cal).size > 1:
            return cal
    return probs


# --- parallel fold execution -------------------------------------------------
_G: dict = {}


def _gnn_init(P_raw, H_raw, E_raw, pidx, hidx, y, n_p, cfg, seed, threads,
              use_graph):
    limit_threads(threads)
    _G.update(P=P_raw, H=H_raw, E=E_raw, pidx=pidx, hidx=hidx, y=y,
              n_p=n_p, cfg=cfg, seed=seed, use_graph=use_graph)


def _gnn_worker(item):
    name, tr, te = item
    probs = _fit_fold(_G["P"], _G["H"], _G["E"], _G["pidx"], _G["hidx"],
                      _G["y"], tr, te, _G["n_p"], _G["cfg"], _G["seed"],
                      _G["use_graph"])
    return name, te, probs


def run_gnn_cv(df, P_raw, H_raw, E_raw, folds, cfg, seed,
               cluster_col: str = "host_genus", use_graph: bool = True) -> dict:
    """Leakage-safe GNN cross-validation. df must contain integer columns
    'pidx' (phage node index) and 'hidx' (host node index).

    Folds run across up to compute.n_jobs (<=10) worker processes; each worker
    is pinned to threads_per_job torch/BLAS threads."""
    pidx = df["pidx"].to_numpy()
    hidx = df["hidx"].to_numpy()
    y = df["label"].to_numpy().astype(np.float32)
    n_p = int(pidx.max()) + 1
    folds = list(folds)
    items = [(f.name, f.train_idx, f.test_idx) for f in folds]
    n_jobs = resolve_n_jobs(cfg, len(items))
    threads = int(cfg.get("compute", {}).get("threads_per_job", 1))

    if n_jobs <= 1:
        _gnn_init(P_raw, H_raw, E_raw, pidx, hidx, y, n_p, cfg, seed, threads,
                  use_graph)
        results = [_gnn_worker(it) for it in items]
    else:
        # 'spawn' (not 'fork'): forking a process that has imported torch can
        # deadlock on inherited thread locks. Spawn starts clean interpreters
        # that honour the thread-cap env vars set at entry-point import time.
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
                max_workers=n_jobs, mp_context=ctx, initializer=_gnn_init,
                initargs=(P_raw, H_raw, E_raw, pidx, hidx, y, n_p, cfg, seed,
                          threads, use_graph)) as ex:
            results = list(ex.map(_gnn_worker, items))

    by_name = {name: (te, probs) for name, te, probs in results}
    per_fold, fold_aucs, fold_clusters = [], {}, {}
    pooled_y, pooled_p = [], []
    for fold in folds:
        te, probs = by_name[fold.name]
        m = binary_metrics(y[te], probs)
        per_fold.append(m)
        if m is not None:
            fold_aucs[fold.name] = m["roc_auc"]
            try:
                fold_clusters[fold.name] = df.iloc[te][cluster_col].mode().iloc[0]
            except Exception:
                fold_clusters[fold.name] = "na"
        pooled_y.append(y[te])
        pooled_p.append(probs)
    pooled_y = np.concatenate(pooled_y) if pooled_y else np.array([])
    pooled_p = np.concatenate(pooled_p) if pooled_p else np.array([])
    return {"agg": aggregate_folds(per_fold, pooled_y, pooled_p),
            "fold_aucs": fold_aucs, "fold_clusters": fold_clusters,
            "pooled_y": pooled_y, "pooled_p": pooled_p}
