"""
Density regularization utilities for aligning learned potential with -log KDE density.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from landscape_core import estimate_potential_from_density


def zscore_tensor(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-batch z-score normalization."""
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x - mean) / (std + eps)


def compute_kde_neglogp_numpy(
    coords: np.ndarray,
    bandwidth: Optional[float] = None,
    n_neighbors: int = 30,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate -log p from embedding coordinates via KDE."""
    coords = np.asarray(coords, dtype=float)
    neg_logp, log_prob = estimate_potential_from_density(coords, bandwidth=bandwidth)
    if bandwidth is None:
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(coords))).fit(coords)
        dists, _ = nbrs.kneighbors(coords)
        bandwidth = float(np.median(dists[:, 1:]))
    return np.asarray(neg_logp, dtype=float), np.asarray(log_prob, dtype=float), float(bandwidth)


def ensure_pca_basis_for_density(adata, *, n_pcs: int = 50) -> str:
    """Compute ``X_pca`` on processed expression if no density basis exists yet."""
    for key in ("X_pca", "X_latent_pca"):
        if key in adata.obsm:
            return key
    import scanpy as sc

    n_comps = min(int(n_pcs), adata.n_vars - 1, adata.n_obs - 1)
    if n_comps < 2:
        raise ValueError(
            f"Cannot compute PCA for density regularization: n_obs={adata.n_obs}, n_vars={adata.n_vars}"
        )
    sc.tl.pca(adata, n_comps=n_comps)
    return "X_pca"


def _resolve_basis(adata, basis: str, n_pcs: int) -> Tuple[np.ndarray, str]:
    if basis not in adata.obsm:
        ensure_pca_basis_for_density(adata, n_pcs=max(int(n_pcs), 50))
    if basis in adata.obsm:
        coords = np.asarray(adata.obsm[basis], dtype=float)
        if coords.shape[1] > n_pcs:
            coords = coords[:, :n_pcs]
        return coords, basis
    for fallback in ("X_pca", "X_latent_pca"):
        if fallback in adata.obsm:
            coords = np.asarray(adata.obsm[fallback], dtype=float)
            if coords.shape[1] > n_pcs:
                coords = coords[:, :n_pcs]
            return coords, fallback
    raise KeyError(f"Basis {basis!r} not found and no X_pca / X_latent_pca fallback in adata.obsm")


def attach_density_target_to_adata(
    adata,
    basis: str = "X_pca",
    target_key: str = "density_neglogp",
    n_pcs: int = 20,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Attach per-cell KDE -log p target to adata.obs."""
    coords, used_basis = _resolve_basis(adata, basis, n_pcs)
    neg_logp, _, bw = compute_kde_neglogp_numpy(coords, bandwidth=bandwidth)
    adata.obs[target_key] = neg_logp
    adata.uns.setdefault("density_regularization", {})
    adata.uns["density_regularization"].update(
        {"basis": used_basis, "n_pcs": n_pcs, "bandwidth": bw}
    )
    return neg_logp


def density_regularization_loss(
    model_potential: torch.Tensor,
    density_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE between z-scored model potential and z-scored density target."""
    pred = model_potential.reshape(-1)
    target = density_target.reshape(-1).detach()
    if mask is not None:
        pred = pred[mask.reshape(-1)]
        target = target[mask.reshape(-1)]
    if pred.numel() == 0:
        return torch.zeros((), device=model_potential.device, dtype=model_potential.dtype)
    return F.mse_loss(zscore_tensor(pred), zscore_tensor(target))


def batch_knn_neglog_density(
    z: torch.Tensor,
    *,
    k: int = 10,
    cell_type: Optional[torch.Tensor] = None,
    within_type: bool = True,
) -> torch.Tensor:
    """
    Differentiable-free proxy for -log p(z) via k-NN distance in latent space.

    When within_type=True, density is estimated separately per cell type so U
    reflects within-lineage rarity rather than global type composition.
    """
    z = z.detach()
    if z.shape[0] < 2:
        return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

    def _knn_neglog(group_z: torch.Tensor) -> torch.Tensor:
        n = group_z.shape[0]
        if n < 2:
            return torch.zeros(n, device=group_z.device, dtype=group_z.dtype)
        kk = min(int(k), n - 1)
        dists = torch.cdist(group_z, group_z)
        knn = torch.topk(dists, kk + 1, largest=False).values[:, -1]
        return knn.pow(2)

    if within_type and cell_type is not None and cell_type.numel() == z.shape[0]:
        out = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for ct in torch.unique(cell_type):
            mask = cell_type == ct
            if int(mask.sum()) < 2:
                continue
            out[mask] = _knn_neglog(z[mask])
        return out
    return _knn_neglog(z)


def density_regularization_loss_latent_batch(
    model_potential: torch.Tensor,
    z_latent: torch.Tensor,
    cell_type: Optional[torch.Tensor] = None,
    *,
    within_type: bool = True,
    k: int = 10,
) -> torch.Tensor:
    """Align U0(z) with batch k-NN -log density in latent space."""
    target = batch_knn_neglog_density(
        z_latent, k=k, cell_type=cell_type, within_type=within_type
    )
    return density_regularization_loss(model_potential, target)


def attach_density_target_within_type(
    adata,
    basis: str = "X_pca",
    target_key: str = "density_neglogp",
    cell_type_key: str = "cell_type",
    n_pcs: int = 20,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Attach per-cell KDE -log p with within-type normalization."""
    coords, used_basis = _resolve_basis(adata, basis, n_pcs)
    ctypes = adata.obs[cell_type_key].values
    neg_logp = np.zeros(adata.n_obs, dtype=float)
    for ct in np.unique(ctypes):
        mask = ctypes == ct
        idx = np.where(mask)[0]
        if idx.size < 3:
            neg_logp[idx] = 0.0
            continue
        sub_coords = coords[idx]
        sub_neg, _, bw = compute_kde_neglogp_numpy(sub_coords, bandwidth=bandwidth)
        neg_logp[idx] = sub_neg
    adata.obs[target_key] = neg_logp
    adata.uns.setdefault("density_regularization", {})
    adata.uns["density_regularization"].update(
        {
            "basis": used_basis,
            "n_pcs": n_pcs,
            "within_cell_type": True,
            "cell_type_key": cell_type_key,
        }
    )
    return neg_logp
