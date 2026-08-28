#!/usr/bin/env python
"""
Benchmark latent-SDE / MomentumNetwork vs scVelo-style graph velocity and CellRank.

Fair primary metric (trajectory–time PCC)
-----------------------------------------
All methods use the **same** protocol:

    velocity field → graph / Markov order reconstruction → PCC(order, bio_time)

LDH-scRNA therefore uses the learned MomentumNetwork field (projected to the
embedding), **not** the supervised ``pseudotime_head`` (which is trained with
MSE against true time and is reported only as a diagnostic column).

Datasets without unspliced counts:
  - scVelo uses a kNN time-ordered displacement velocity in embedding space
    (documented as kinetics-free proxy).
  - CellRank builds a first-order Markov transition matrix from that velocity
    field and reports expected hitting time to biological terminals
    (CellRank-style absorbing-Markov fate analysis) plus sink coherence.
"""

from __future__ import annotations

import argparse
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from plot_utils import PALETTE, configure_headless, style_axis
from methods_enhancement_utils import (
    fig_path,
    methods_outdir,
    result_path,
    sink_convergence_score,
    trajectory_time_pcc,
    write_output_file_index,
)

configure_headless()

from celltype_analysis import DATASET_REGISTRY, load_annotated_adata
from dataset_pipeline import recommended_checkpoint_dir
from latent_embeddings import ensure_latent_embeddings


def _biological_time(adata, dataset_key: str) -> np.ndarray:
    if dataset_key == "GSE141259" and "stage" in adata.obs:
        stage_ord = {"D0": 0.0, "D3": 3.0, "D7": 7.0, "D10": 10.0, "D14": 14.0, "D21": 21.0, "D28": 28.0}
        return adata.obs["stage"].astype(str).map(lambda s: stage_ord.get(s, np.nan)).astype(float).values
    if dataset_key == "HGSOC" and "stage" in adata.obs:
        stage_ord = {"IIIC": 0.0, "IVA": 1.0, "IVB": 2.0}
        return adata.obs["stage"].astype(str).map(lambda s: stage_ord.get(s, np.nan)).astype(float).values
    if "time" in adata.obs:
        return adata.obs["time"].astype(float).values
    if "condition" in adata.obs and dataset_key == "GSE155622":
        from run_gse155622_analysis import _ensure_time_column

        ad = adata.copy()
        _ensure_time_column(ad)
        return ad.obs["time"].astype(float).values
    if "treatment_phase" in adata.obs:
        phase_ord = {"naive": 0.0, "post-nact": 1.0, "post": 1.0, "pre": 0.0}
        return (
            adata.obs["treatment_phase"]
            .astype(str)
            .str.lower()
            .map(lambda s: phase_ord.get(s, np.nan))
            .astype(float)
            .values
        )
    if "stage" in adata.obs:
        uniq = sorted(adata.obs["stage"].astype(str).unique())
        stage_ord = {s: float(i) for i, s in enumerate(uniq)}
        return adata.obs["stage"].astype(str).map(lambda s: stage_ord.get(s, np.nan)).astype(float).values
    if "pseudotime" in adata.obs:
        return adata.obs["pseudotime"].astype(float).values
    return np.full(adata.n_obs, np.nan)


def _embedding_coords(
    adata,
    *,
    prefer_latent_pca: bool = True,
    n_dims: int = 10,
) -> Tuple[np.ndarray, str]:
    """Coordinates for velocity/order reconstruction.

    Prefer ``X_latent_pca`` (same linear map as MomentumNetwork's latent) over
    UMAP — projecting latent momentum onto UMAP axes via ``p[:, :2]`` is invalid.
    """
    if prefer_latent_pca and "X_latent_pca" in adata.obsm:
        arr = np.asarray(adata.obsm["X_latent_pca"], dtype=float)
        d = min(int(n_dims), arr.shape[1])
        if d >= 2:
            return arr[:, :d], f"X_latent_pca:{d}"
    for key in ("X_latent_pca", "X_latent", "X_umap"):
        if key in adata.obsm and adata.obsm[key].shape[1] >= 2:
            arr = np.asarray(adata.obsm[key], dtype=float)
            d = min(int(n_dims), arr.shape[1])
            return arr[:, :d], f"{key}:{d}"
    raise KeyError("No embedding with ≥2 dims in adata.obsm")


def _project_latent_velocity_to_embedding(
    p_latent: np.ndarray,
    adata,
    emb_key: str,
    n_dims: int,
) -> np.ndarray:
    """Map latent momentum into the embedding used for the shared order protocol."""
    p = np.asarray(p_latent, dtype=float)
    if emb_key.startswith("X_latent_pca"):
        comps = None
        for ck in ("X_latent_pca_pca_components", "pca_components"):
            if ck in getattr(adata, "uns", {}):
                comps = np.asarray(adata.uns[ck], dtype=float)
                break
        if comps is not None and comps.ndim == 2 and comps.shape[1] == p.shape[1]:
            d = min(int(n_dims), comps.shape[0], p.shape[1])
            return p @ comps[:d].T
        # Fallback: PCA space already aligns with latent principal axes ≈ first PCs of z
        if "X_latent" in adata.obsm:
            from sklearn.decomposition import PCA

            z = np.asarray(adata.obsm["X_latent"], dtype=float)
            d = min(int(n_dims), z.shape[1], max(2, z.shape[0] - 1))
            pca = PCA(n_components=d, random_state=0).fit(z)
            return p @ pca.components_.T
    if emb_key.startswith("X_latent"):
        d = min(int(n_dims), p.shape[1])
        return p[:, :d]
    # UMAP / other: local linear map from latent displacements → embedding displacements
    if "X_latent" not in adata.obsm:
        d = min(int(n_dims), p.shape[1])
        return p[:, :d]
    from sklearn.neighbors import NearestNeighbors

    z = np.asarray(adata.obsm["X_latent"], dtype=float)
    # Recover raw embedding array from key prefix
    raw_key = emb_key.split(":")[0]
    x = np.asarray(adata.obsm[raw_key], dtype=float)[:, :n_dims]
    n = len(z)
    k = min(30, max(3, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k).fit(z)
    _, idx = nbrs.kneighbors(z)
    v = np.zeros((n, x.shape[1]), dtype=float)
    for i in range(n):
        neigh = idx[i, 1:]
        A = z[neigh] - z[i]
        B = x[neigh] - x[i]
        # least squares: A @ M.T ≈ B  => M.T = lstsq(A, B)
        try:
            M_t, *_ = np.linalg.lstsq(A, B, rcond=None)
            v[i] = p[i] @ M_t
        except Exception:
            v[i] = 0.0
    return v


def _knn_time_velocity(coords: np.ndarray, time: np.ndarray, *, n_neighbors: int = 30) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=float)
    t = np.asarray(time, dtype=float).reshape(-1)
    k = min(n_neighbors, max(3, len(coords) - 1))
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    vel = np.zeros_like(coords)
    for i in range(len(coords)):
        neigh = idx[i, 1:]
        later = neigh[t[neigh] >= t[i] - 1e-9]
        if later.size == 0:
            later = neigh
        disp = coords[later] - coords[i]
        w = np.clip(t[later] - t[i], 0.0, None) + 0.05
        vel[i] = (disp * w[:, None]).sum(axis=0) / (w.sum() + 1e-8)
    return vel


def _momentum_velocity(
    adata,
    checkpoint: Path,
    device: str = "cpu",
    *,
    emb_key: str = "X_latent_pca:10",
    n_dims: int = 10,
) -> Optional[np.ndarray]:
    """Learned MomentumNetwork field projected into the benchmark embedding."""
    try:
        import torch
        from run_gse155622_analysis import _load_hamiltonian_bundle_from_checkpoint
    except Exception:
        return None
    bundle = _load_hamiltonian_bundle_from_checkpoint(checkpoint, device=device)
    if bundle is None:
        return None
    ensure_latent_embeddings(adata, checkpoint_dir=str(checkpoint), warn=False)
    if "X_latent" not in adata.obsm:
        return None
    z = np.asarray(adata.obsm["X_latent"], dtype=float)
    if "time" not in adata.obs and "stage" in adata.obs:
        stage_ord = {
            "D0": 0.0,
            "D3": 3.0,
            "D7": 7.0,
            "D10": 10.0,
            "D14": 14.0,
            "D21": 21.0,
            "D28": 28.0,
        }
        tvals = adata.obs["stage"].astype(str).map(lambda s: stage_ord.get(s, 0.0)).astype(float).values
    elif "time" in adata.obs:
        tvals = adata.obs["time"].astype(float).values
    else:
        tvals = np.zeros(adata.n_obs)
    with torch.no_grad():
        z_t = torch.tensor(z, dtype=torch.float32, device=device)
        t_t = torch.tensor(tvals, dtype=torch.float32, device=device).unsqueeze(1)
        p = bundle.initial_momentum(z_t, t_t).detach().cpu().numpy()
    return _project_latent_velocity_to_embedding(p, adata, emb_key, n_dims)


def _velocity_transition_matrix(
    coords: np.ndarray,
    velocity: np.ndarray,
    *,
    n_neighbors: int = 30,
    connectivity_mix: float = 0.25,
    terminal_mask: Optional[np.ndarray] = None,
    terminal_leak: float = 0.05,
) -> np.ndarray:
    """Dense row-stochastic first-order Markov kernel from embedding velocity.

    If ``terminal_mask`` is given, each transient cell receives a small
    probability mass toward its nearest terminals so absorbing classes remain
    reachable (critical when terminals are rare / spatially isolated).
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    n = len(coords)
    k = min(n_neighbors, max(3, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    dist, idx = nbrs.kneighbors(coords)
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        neigh = idx[i, 1:]
        d = dist[i, 1:] + 1e-8
        disp = coords[neigh] - coords[i]
        # velocity projection onto neighbor edges
        proj = np.einsum("ij,j->i", disp, vel[i])
        proj = proj / (np.linalg.norm(disp, axis=1) + 1e-8)
        # softplus-like nonnegativity + distance decay
        score = np.maximum(proj, 0.0) / d + connectivity_mix / d
        if not np.isfinite(score).any() or score.sum() <= 0:
            score = 1.0 / d
        score = np.clip(score, 0.0, None)
        score = score / (score.sum() + 1e-12)
        T[i, neigh] = score
        # tiny self-loop for numerical stability
        T[i, i] += 1e-6
        T[i] /= T[i].sum()

    if terminal_mask is not None:
        term = np.asarray(terminal_mask, dtype=bool)
        term_idx = np.flatnonzero(term)
        if term_idx.size >= 1 and terminal_leak > 0:
            # nearest up to 5 terminals for each transient
            t_nbrs = NearestNeighbors(n_neighbors=min(5, term_idx.size)).fit(coords[term_idx])
            t_dist, t_loc = t_nbrs.kneighbors(coords)
            for i in range(n):
                if term[i]:
                    continue
                locs = term_idx[t_loc[i]]
                dd = t_dist[i] + 1e-8
                w = (1.0 / dd)
                w = w / (w.sum() + 1e-12)
                T[i] *= 1.0 - terminal_leak
                T[i, locs] += terminal_leak * w
                T[i] /= T[i].sum()
    return T


def _markov_absorption_pseudotime(
    T: np.ndarray,
    terminal_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    CellRank-style absorbing Markov analysis via expected hitting time τ:

        (I - Q) τ = 1   for transient states,   τ_term = 0

    Pseudotime is 1 − normalized(τ) (near-sink → high). When Q is nearly
    stochastic, a mild damping α<1 keeps the linear system well-conditioned.
    """
    n = T.shape[0]
    term = np.asarray(terminal_mask, dtype=bool).copy()
    if term.sum() < 3:
        return np.full(n, np.nan), {"status": "too_few_terminals"}
    # Cap terminals for speed / conditioning
    if term.sum() > max(120, n // 8):
        idx = np.flatnonzero(term)
        keep = np.random.default_rng(0).choice(idx, size=max(120, n // 8), replace=False)
        term = np.zeros(n, dtype=bool)
        term[keep] = True

    trans = ~term
    if trans.sum() < 5:
        return np.full(n, np.nan), {"status": "too_few_transient"}

    T_abs = T.copy()
    for i in np.flatnonzero(term):
        T_abs[i, :] = 0.0
        T_abs[i, i] = 1.0

    Q = T_abs[np.ix_(trans, trans)]
    row_leak = 1.0 - np.clip(Q.sum(axis=1), 0.0, 1.0)
    # Damping keeps (I - αQ) invertible even with poorly connected sinks
    alpha = 0.995 if float(row_leak.mean()) > 1e-4 else 0.98
    I = np.eye(Q.shape[0])
    ones = np.ones(Q.shape[0], dtype=float)
    try:
        tau_trans = np.linalg.solve(I - alpha * Q, ones)
    except np.linalg.LinAlgError:
        tau_trans = np.linalg.lstsq(I - alpha * Q, ones, rcond=None)[0]
    tau_trans = np.asarray(tau_trans, dtype=float).reshape(-1)
    # Rare negative numerical artifacts → replace with median of positives
    if np.any(tau_trans < 0):
        pos = tau_trans[tau_trans > 0]
        fill = float(np.median(pos)) if pos.size else 1.0
        tau_trans = np.where(tau_trans < 0, fill, tau_trans)

    tau = np.zeros(n, dtype=float)
    tau[trans] = tau_trans
    tau[term] = 0.0

    sink_coherence = float(np.nanmean(tau[trans]) - np.nanmean(tau[term]))
    metrics = {
        "status": "ok",
        "terminal_prob_mean": float(np.nanmean(tau[term])),
        "mean_hitting_time": float(np.nanmean(tau)),
        "sink_coherence": sink_coherence,
        "n_terminals": int(term.sum()),
        "n_transient": int(trans.sum()),
        "mean_row_leak": float(row_leak.mean()),
        "damping_alpha": float(alpha),
    }
    tmax = float(np.nanmax(tau)) + 1e-8
    pt = 1.0 - (tau / tmax)
    return pt, metrics


def _orient_pseudotime(pt: np.ndarray, bio_t: np.ndarray) -> np.ndarray:
    """Flip orientation so PCC(pt, bio_t) ≥ 0 when both are finite."""
    pt = np.asarray(pt, dtype=float).copy()
    bio = np.asarray(bio_t, dtype=float)
    m = np.isfinite(pt) & np.isfinite(bio)
    if m.sum() < 3 or np.nanstd(pt[m]) < 1e-12 or np.nanstd(bio[m]) < 1e-12:
        return pt
    if float(np.corrcoef(pt[m], bio[m])[0, 1]) < 0:
        pt = 1.0 - pt
    return pt


def _run_cellrank(
    adata,
    velocity: np.ndarray,
    terminal_mask: np.ndarray,
    *,
    bio_t: np.ndarray,
) -> Dict[str, float]:
    """
    CellRank-style first-order Markov absorption on an embedding velocity field.

    Uses expected hitting time to biologically defined terminals (same math as
    CellRank absorbing-Markov fate / absorption-time analysis). GPCCA autodetection
    is skipped for speed and stability when unspliced counts are unavailable.
    """
    out = {
        "trajectory_time_pcc": float("nan"),
        "cellrank_terminal_prob_mean": float("nan"),
        "cellrank_sink_coherence": float("nan"),
        "cellrank_fate_pcc": float("nan"),
        "cellrank_status": "failed",
    }
    if int(np.asarray(terminal_mask).sum()) < 5:
        out["cellrank_status"] = "too_few_terminals"
        return out

    coords, _ = _embedding_coords(adata)
    try:
        T = _velocity_transition_matrix(
            coords,
            velocity,
            n_neighbors=30,
            connectivity_mix=0.25,
            terminal_mask=terminal_mask,
            terminal_leak=0.05,
        )
        pt, metrics = _markov_absorption_pseudotime(T, terminal_mask)
        if metrics.get("status") != "ok" or not np.isfinite(pt).any():
            out["cellrank_status"] = metrics.get("status", "absorption_failed")
            return out
        pt = _orient_pseudotime(pt, bio_t)
        pcc = trajectory_time_pcc(pt, bio_t)
        out.update(
            {
                "trajectory_time_pcc": float(pcc),
                "cellrank_terminal_prob_mean": metrics["terminal_prob_mean"],
                "cellrank_sink_coherence": metrics["sink_coherence"],
                "cellrank_fate_pcc": float(pcc),
                "cellrank_status": "ok_markov_hitting_time",
                "_pseudotime": pt,
            }
        )
        try:
            import cellrank as cr  # noqa: F401

            out["cellrank_status"] = "ok_cellrank_style_hitting_time"
        except ImportError:
            out["cellrank_status"] = "ok_markov_hitting_time_no_pkg"
        return out
    except Exception as exc:
        out["cellrank_status"] = f"error:{type(exc).__name__}:{exc}"
        warnings.warn(f"CellRank-style absorption failed: {exc}", UserWarning)
        traceback.print_exc()
        return out


def _graph_pseudotime_from_velocity(
    coords: np.ndarray,
    velocity: np.ndarray,
    bio_t: np.ndarray,
) -> np.ndarray:
    """Shared diffusion-like order from an embedding velocity field.

    Used for both MomentumNetwork and the scVelo kNN proxy so trajectory–time
    PCC compares like-with-like (velocity → order), not supervised time heads.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    n = len(coords)
    if vel.shape[0] != n:
        raise ValueError(f"velocity length {vel.shape[0]} != n_cells {n}")
    if vel.shape[1] < coords.shape[1]:
        raise ValueError("velocity dim must be >= embedding dim")
    vel = vel[:, : coords.shape[1]]

    k = min(30, max(3, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    dist, idx = nbrs.kneighbors(coords)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j, d in zip(idx[i, 1:], dist[i, 1:]):
            w = max(0.0, float(np.dot(vel[i], coords[j] - coords[i]))) / (d + 1e-6)
            adj[i, j] = w
    adj = (adj + adj.T) * 0.5
    deg = adj.sum(axis=1) + 1e-8
    lap = np.diag(deg) - adj
    try:
        _, v = np.linalg.eigh(lap)
        pt = v[:, 1]
    except Exception:
        pt = np.cumsum(np.linalg.norm(vel, axis=1))
    pt = (pt - np.nanmin(pt)) / (np.nanmax(pt) - np.nanmin(pt) + 1e-8)
    return _orient_pseudotime(pt, bio_t)


def _scvelo_pseudotime(adata, bio_t: np.ndarray) -> np.ndarray:
    """Diffusion-like pseudotime from kNN time-ordered velocity."""
    coords, _ = _embedding_coords(adata)
    vel = _knn_time_velocity(coords, bio_t)
    return _graph_pseudotime_from_velocity(coords, vel, bio_t)


def _terminal_mask(adata, dataset_key: str) -> np.ndarray:
    """Biological sink cells used as absorbing states for CellRank-style analysis."""
    if dataset_key == "GSE141259" and "stage" in adata.obs:
        return adata.obs["stage"].astype(str).values == "D28"
    if dataset_key == "GSE155622" and "condition" in adata.obs:
        # Latest injury time point only (not the broad 2d/7d/14d mixture)
        return adata.obs["condition"].astype(str).values == "SNI 14d"
    if dataset_key == "HGSOC" and "potential" in adata.obs:
        # Prefer deep-valley cells: stage-IVB is too rare/isolated in embedding kNN
        pot = adata.obs["potential"].astype(float).values
        return pot <= np.nanpercentile(pot, 10)
    if dataset_key == "HGSOC" and "stage" in adata.obs:
        return adata.obs["stage"].astype(str).values == "IVB"
    return np.zeros(adata.n_obs, dtype=bool)


def _sink_query_points(adata, dataset_key: str) -> Optional[np.ndarray]:
    coords, _ = _embedding_coords(adata, prefer_latent_pca=True, n_dims=10)
    if dataset_key == "GSE141259":
        if "cell.type" in adata.obs:
            ct = adata.obs["cell.type"].astype(str)
            sink = ct.isin(["Club cells", "AT2 cells", "AT1 cells"])
            if sink.any():
                return coords[sink.values]
        if "metacelltype" in adata.obs:
            ct = adata.obs["metacelltype"].astype(str)
            sink = ct.isin(["club_cells", "alv_epithelium"])
            if sink.any():
                return coords[sink.values]
        if "stage" in adata.obs:
            late = adata.obs["stage"].astype(str) == "D28"
            if late.any():
                return coords[late.values]
    if dataset_key == "GSE155622" and "condition" in adata.obs:
        late = adata.obs["condition"].astype(str).str.contains("2d|14d")
        if late.any():
            return coords[late.values]
    if dataset_key == "HGSOC" and "potential" in adata.obs:
        pot = adata.obs["potential"].astype(float).values
        deep = pot <= np.nanpercentile(pot, 15)
        if deep.any():
            return coords[deep]
    return coords[np.random.default_rng(0).choice(len(coords), size=min(200, len(coords)), replace=False)]


def _subset_for_benchmark(adata, dataset_key: str):
    if dataset_key == "GSE141259":
        if "metacelltype" in adata.obs:
            ct = adata.obs["metacelltype"].astype(str)
            mask = ct.isin(["club_cells", "alv_epithelium", "fibroblasts"])
            if mask.sum() >= 200:
                return adata[mask].copy()
        if "cell.type" in adata.obs:
            ct = adata.obs["cell.type"].astype(str)
            mask = ct.isin(["Club cells", "AT2 cells", "Krt8 ADI", "AT1 cells", "Fibroblasts", "Myofibroblasts"])
            if mask.sum() >= 200:
                return adata[mask].copy()
    if dataset_key == "GSE155622":
        from run_gse155622_analysis import _neuron

        try:
            neu = _neuron(adata)
            if neu.n_obs >= 100:
                return neu
        except Exception:
            pass
    if dataset_key == "HGSOC":
        col = "annotation" if "annotation" in adata.obs else "cell_type"
        if col in adata.obs:
            eoc = adata[adata.obs[col].astype(str).str.contains("EOC", case=False)].copy()
            if eoc.n_obs >= 100:
                return eoc
    return adata


def run_benchmark(
    dataset_key: str,
    checkpoint_dir: Path,
    device: str = "cpu",
    *,
    max_cells: int = 5000,
    n_embed_dims: int = 10,
) -> pd.DataFrame:
    profile = DATASET_REGISTRY[dataset_key]
    adata = load_annotated_adata(profile, str(checkpoint_dir))
    ensure_latent_embeddings(adata, checkpoint_dir=str(checkpoint_dir), warn=False)
    adata = _subset_for_benchmark(adata, dataset_key)
    coords, emb_key = _embedding_coords(adata, prefer_latent_pca=True, n_dims=n_embed_dims)
    finite = np.isfinite(coords).all(axis=1)
    if "potential" in adata.obs:
        finite &= np.isfinite(adata.obs["potential"].astype(float).values)
    if finite.sum() < 100:
        raise ValueError(f"Too few finite cells for benchmark: {finite.sum()}")
    if not finite.all():
        adata = adata[finite].copy()
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=0, copy=False)

    bio_t = _biological_time(adata, dataset_key)
    coords, emb_key = _embedding_coords(adata, prefer_latent_pca=True, n_dims=n_embed_dims)
    out = methods_outdir(checkpoint_dir)
    term_mask = _terminal_mask(adata, dataset_key)

    knn_vel = _knn_time_velocity(coords, bio_t)
    mom_vel = _momentum_velocity(
        adata,
        checkpoint_dir,
        device=device,
        emb_key=emb_key,
        n_dims=n_embed_dims,
    )
    if mom_vel is None:
        warnings.warn("MomentumNetwork velocity unavailable; using kNN velocity as fallback", UserWarning)
        mom_vel = knn_vel
    if mom_vel.shape[1] != coords.shape[1]:
        raise ValueError(
            f"Momentum velocity dim {mom_vel.shape[1]} != embedding dim {coords.shape[1]} ({emb_key})"
        )

    sink_pts = _sink_query_points(adata, dataset_key)
    rows: List[dict] = []

    # Supervised head: diagnostic only (trained with MSE vs true time; not comparable).
    if "pseudotime" in adata.obs:
        supervised_head_pcc = float(
            trajectory_time_pcc(adata.obs["pseudotime"].astype(float).values, bio_t)
        )
    else:
        supervised_head_pcc = float("nan")

    # --- Method 1: MomentumNetwork (fair: velocity → same graph order as scVelo proxy) ---
    pt_mom = _graph_pseudotime_from_velocity(coords, mom_vel, bio_t)
    # Sink diagnostics are 2D-only.
    sink_pts_2d = None if sink_pts is None else np.asarray(sink_pts, dtype=float)[:, :2]
    sink_mom = (
        sink_convergence_score(coords[:, :2], mom_vel[:, :2], sink_pts_2d)
        if sink_pts_2d is not None
        else {}
    )
    cr_on_mom = _run_cellrank(adata, mom_vel, term_mask, bio_t=bio_t)
    rows.append(
        {
            "dataset": dataset_key,
            "method": "MomentumNetwork",
            "trajectory_time_pcc": trajectory_time_pcc(pt_mom, bio_t),
            "pseudotime_protocol": "velocity_graph_laplacian",
            "supervised_pseudotime_head_pcc": supervised_head_pcc,
            "markov_hitting_pcc_on_momentum": float(cr_on_mom.get("trajectory_time_pcc", np.nan)),
            "sink_strength": sink_mom.get("sink_strength", np.nan),
            "sink_divergence": sink_mom.get("divergence", np.nan),
            "cellrank_terminal_prob_mean": cr_on_mom.get("cellrank_terminal_prob_mean", np.nan),
            "cellrank_sink_coherence": cr_on_mom.get("cellrank_sink_coherence", np.nan),
            "cellrank_fate_pcc": cr_on_mom.get("cellrank_fate_pcc", np.nan),
            "cellrank_status": cr_on_mom.get("cellrank_status", ""),
            "n_cells": adata.n_obs,
            "embedding": emb_key,
        }
    )

    # --- Method 2: scVelo kNN proxy ---
    pt_scv = _scvelo_pseudotime(adata, bio_t)
    sink_scv = (
        sink_convergence_score(coords[:, :2], knn_vel[:, :2], sink_pts_2d)
        if sink_pts_2d is not None
        else {}
    )
    cr_on_scv = _run_cellrank(adata, knn_vel, term_mask, bio_t=bio_t)
    rows.append(
        {
            "dataset": dataset_key,
            "method": "scVelo_kNN_proxy",
            "trajectory_time_pcc": trajectory_time_pcc(pt_scv, bio_t),
            "pseudotime_protocol": "velocity_graph_laplacian",
            "supervised_pseudotime_head_pcc": np.nan,
            "markov_hitting_pcc_on_momentum": np.nan,
            "sink_strength": sink_scv.get("sink_strength", np.nan),
            "sink_divergence": sink_scv.get("divergence", np.nan),
            "cellrank_terminal_prob_mean": cr_on_scv.get("cellrank_terminal_prob_mean", np.nan),
            "cellrank_sink_coherence": cr_on_scv.get("cellrank_sink_coherence", np.nan),
            "cellrank_fate_pcc": cr_on_scv.get("cellrank_fate_pcc", np.nan),
            "cellrank_status": cr_on_scv.get("cellrank_status", ""),
            "n_cells": adata.n_obs,
            "embedding": emb_key,
        }
    )

    # --- Method 3: CellRank (first-order Markov on kNN velocity field) ---
    cr_method = _run_cellrank(adata, knn_vel, term_mask, bio_t=bio_t)
    pt_cr = cr_method.get("_pseudotime")
    if pt_cr is None:
        pt_cr = np.full(adata.n_obs, np.nan)
    # Vector-field sink for CellRank uses same first-order knn velocity
    sink_cr = sink_scv
    rows.append(
        {
            "dataset": dataset_key,
            "method": "CellRank",
            "trajectory_time_pcc": float(cr_method.get("trajectory_time_pcc", np.nan)),
            "pseudotime_protocol": "absorbing_markov_hitting_time",
            "supervised_pseudotime_head_pcc": np.nan,
            "markov_hitting_pcc_on_momentum": np.nan,
            "sink_strength": sink_cr.get("sink_strength", np.nan),
            "sink_divergence": sink_cr.get("divergence", np.nan),
            "cellrank_terminal_prob_mean": cr_method.get("cellrank_terminal_prob_mean", np.nan),
            "cellrank_sink_coherence": cr_method.get("cellrank_sink_coherence", np.nan),
            "cellrank_fate_pcc": cr_method.get("cellrank_fate_pcc", np.nan),
            "cellrank_status": cr_method.get("cellrank_status", ""),
            "n_cells": adata.n_obs,
            "embedding": emb_key,
        }
    )

    df = pd.DataFrame(rows)
    df.to_csv(result_path(out, f"sota_benchmark_{dataset_key}.csv"), index=False)

    # Figures: PCC + sink + CellRank coherence
    if not df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
        methods_order = df["method"].tolist()
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(methods_order))]

        pcc = df["trajectory_time_pcc"].values
        axes[0].bar(range(len(pcc)), pcc, color=colors)
        axes[0].set_xticks(range(len(pcc)))
        axes[0].set_xticklabels(methods_order, rotation=18, ha="right")
        axes[0].set_ylabel("PCC(velocity-derived order, biological time)")
        axes[0].set_title(f"{dataset_key}: trajectory–time alignment (fair)")
        axes[0].axhline(0.0, color="#999999", lw=0.8)
        style_axis(axes[0], grid_axis="y")

        sink_v = df["sink_strength"].values
        axes[1].bar(range(len(sink_v)), sink_v, color=colors)
        axes[1].set_xticks(range(len(sink_v)))
        axes[1].set_xticklabels(methods_order, rotation=18, ha="right")
        axes[1].set_ylabel("Sink convergence strength")
        axes[1].set_title("Vector-field sink at terminal region")
        style_axis(axes[1], grid_axis="y")

        coh = df["cellrank_sink_coherence"].values
        axes[2].bar(range(len(coh)), coh, color=colors)
        axes[2].set_xticks(range(len(coh)))
        axes[2].set_xticklabels(methods_order, rotation=18, ha="right")
        axes[2].set_ylabel("Mean hitting-time gap (trans − term)")
        axes[2].set_title("CellRank terminal hitting-time coherence")
        axes[2].axhline(0.0, color="#999999", lw=0.8)
        style_axis(axes[2], grid_axis="y")

        fig.suptitle(
            f"{dataset_key}: MomentumNetwork vs scVelo proxy vs CellRank",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(fig_path(out, f"sota_benchmark_{dataset_key}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    write_output_file_index(out, dataset_key=dataset_key)
    return df


def _write_cross_dataset_summaries(frames: List[pd.DataFrame], out_dir: Path) -> None:
    if not frames:
        return
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(result_path(out_dir, "sota_benchmark_all_datasets.csv"), index=False)
    pcc = all_df.pivot_table(index="dataset", columns="method", values="trajectory_time_pcc", aggfunc="first")
    pcc.to_csv(result_path(out_dir, "sota_benchmark_PCC_summary.csv"))
    coh = all_df.pivot_table(
        index="dataset", columns="method", values="cellrank_sink_coherence", aggfunc="first"
    )
    coh.to_csv(result_path(out_dir, "sota_benchmark_CellRank_coherence_summary.csv"))


def main(argv=None):
    p = argparse.ArgumentParser(description="SOTA velocity / CellRank benchmark")
    p.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()) + ["ALL"],
        required=True,
    )
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-cells", type=int, default=5000)
    args = p.parse_args(argv)

    if args.dataset == "ALL":
        keys = ["GSE155622", "GSE141259", "HGSOC"]
        frames = []
        for key in keys:
            ckpt = Path(recommended_checkpoint_dir(key))
            print(f"\n===== Benchmark {key} @ {ckpt} =====", flush=True)
            df = run_benchmark(key, ckpt, device=args.device, max_cells=args.max_cells)
            print(df.to_string(index=False), flush=True)
            frames.append(df)
            # also write combined summaries into each methods_enhancement/
            _write_cross_dataset_summaries(frames, methods_outdir(ckpt))
        # final copy into first dataset outdir already done; also GSE155622 as canonical
        _write_cross_dataset_summaries(frames, methods_outdir(Path(recommended_checkpoint_dir("GSE155622"))))
        return

    ckpt = Path(args.checkpoint_dir or recommended_checkpoint_dir(args.dataset))
    df = run_benchmark(args.dataset, ckpt, device=args.device, max_cells=args.max_cells)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
