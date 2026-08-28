"Model latent embedding extraction and LAP compute-space resolution."

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


def save_latent_embeddings_to_adata(
    model,
    adata,
    expression_tensor=None,
    cell_type_tensor=None,
    device="cuda",
    n_latent_pca=20,
    latent_key="X_latent",
    latent_pca_key="X_latent_pca",
    time_key="time",
    cell_type_key="cell_type",
):
    "Encode cells and store latent + latent PCA in adata.obsm."
    import scipy.sparse as sp

    model.eval()
    dev = device if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    if expression_tensor is None:
        x_all = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        expression_tensor = torch.tensor(x_all, dtype=torch.float32)
    if cell_type_tensor is None:
        if cell_type_key not in adata.obs.columns:
            raise KeyError(f"Missing cell type key: {cell_type_key}")
        cell_type_tensor = torch.tensor(
            adata.obs[cell_type_key].values.astype(int), dtype=torch.long
        )

    with torch.no_grad():
        expression_tensor = expression_tensor.to(dev)
        cell_type_tensor = cell_type_tensor.to(dev)
        if hasattr(model, "encode"):
            z = model.encode(expression_tensor, cell_type_tensor)
        elif hasattr(model, "encoder"):
            z = model.encoder(expression_tensor)
            if hasattr(model, "cell_type_embedding"):
                z = z + model.cell_type_embedding(cell_type_tensor)
        else:
            raise AttributeError("Model does not expose encode() or encoder.")

        z_np = z.detach().cpu().numpy()

    adata.obsm[latent_key] = z_np

    n_components = min(n_latent_pca, z_np.shape[1], z_np.shape[0] - 1)
    if n_components < 1:
        raise ValueError("Too few cells for latent PCA.")
    pca = PCA(n_components=n_components, random_state=0)
    adata.obsm[latent_pca_key] = pca.fit_transform(z_np)

    adata.uns[f"{latent_pca_key}_pca_components"] = pca.components_
    adata.uns[f"{latent_pca_key}_pca_mean"] = pca.mean_
    adata.uns[f"{latent_pca_key}_explained_variance_ratio"] = pca.explained_variance_ratio_

    return adata


def save_latent_embeddings_checkpoint(adata, save_dir: str) -> Optional[Path]:
    "Persist latent arrays aligned to obs_names for downstream LAP validation."
    if "X_latent" not in adata.obsm or "X_latent_pca" not in adata.obsm:
        return None
    out = Path(save_dir) / "latent_embeddings.npz"
    payload = {
        "X_latent": np.asarray(adata.obsm["X_latent"], dtype=float),
        "X_latent_pca": np.asarray(adata.obsm["X_latent_pca"], dtype=float),
        "index": adata.obs_names.astype(str).values,
    }
    if "X_latent_pca_pca_components" in adata.uns:
        payload["pca_components"] = np.asarray(adata.uns["X_latent_pca_pca_components"])
        payload["pca_mean"] = np.asarray(adata.uns["X_latent_pca_pca_mean"])
    np.savez_compressed(out, **payload)
    return out


def merge_latent_embeddings_from_checkpoint(adata, checkpoint_dir: str) -> bool:
    "Load latent embeddings from checkpoint npz into adata.obsm by cell barcode."
    npz_path = Path(checkpoint_dir) / "latent_embeddings.npz"
    if not npz_path.is_file():
        return False
    data = np.load(npz_path, allow_pickle=True)
    index = pd.Index(data["index"].astype(str))
    latent = pd.DataFrame(data["X_latent"], index=index)
    latent_pca = pd.DataFrame(data["X_latent_pca"], index=index)
    overlap = adata.obs_names.astype(str)
    aligned_latent = latent.reindex(overlap)
    aligned_pca = latent_pca.reindex(overlap)
    if aligned_latent.isna().all(axis=None):
        return False
    adata.obsm["X_latent"] = aligned_latent.to_numpy(dtype=float)
    adata.obsm["X_latent_pca"] = aligned_pca.to_numpy(dtype=float)
    if "pca_components" in data:
        adata.uns["X_latent_pca_pca_components"] = data["pca_components"]
        adata.uns["X_latent_pca_pca_mean"] = data["pca_mean"]
    return True


def resolve_lap_compute_key(adata, preferred: str = "X_latent_pca") -> Tuple[str, bool]:
    """Resolve LAP compute embedding key.

Returns (key, used_fallback)."""
    for key in (preferred, "X_pca_lap", "X_pca"):
        if key in adata.obsm:
            return key, key != preferred
    raise KeyError(
        "No valid LAP compute embedding found. Expected X_latent_pca, X_pca_lap, or X_pca."
    )


def get_lap_compute_coords(adata, compute_key: str, n_pcs: Optional[int] = None) -> np.ndarray:
    "Return compute-space coordinates, optionally truncated to first k PCs."
    coords = np.asarray(adata.obsm[compute_key], dtype=float)
    if n_pcs is not None and coords.shape[1] > n_pcs:
        coords = coords[:, :n_pcs]
    return coords


def ensure_latent_embeddings(
    adata,
    checkpoint_dir: Optional[str] = None,
    model=None,
    device: str = "cuda",
    warn: bool = True,
) -> Tuple[str, bool]:
    """Ensure latent PCA is available; merge from checkpoint or compute if model given.

Returns (compute_key, used_fallback)."""
    if "X_latent_pca" in adata.obsm:
        return "X_latent_pca", False

    if checkpoint_dir:
        merged = merge_latent_embeddings_from_checkpoint(adata, checkpoint_dir)
        if merged and "X_latent_pca" in adata.obsm:
            return "X_latent_pca", False

    if model is not None:
        save_latent_embeddings_to_adata(model, adata, device=device)
        return "X_latent_pca", False

    key, used_fallback = resolve_lap_compute_key(adata)
    if warn and used_fallback:
        warnings.warn(
            f"WARNING: X_latent_pca missing; LAP computed in fallback space {key!r}.",
            UserWarning,
            stacklevel=2,
        )
    return key, used_fallback
