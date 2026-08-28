"""
Density-based LAP endpoint selection.

Replaces heuristic endpoint strategies with:
  - start: low-density centroid within start_state
  - end: high-density centroid within end_state
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _local_density(coords: np.ndarray, *, k_neighbors: int = 15) -> np.ndarray:
    """Inverse mean kNN distance as a local density proxy."""
    n = len(coords)
    if n == 0:
        return np.array([], dtype=float)
    k = min(max(1, int(k_neighbors)), max(n - 1, 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    mean_dist = np.mean(dists[:, 1:], axis=1)
    return 1.0 / np.maximum(mean_dist, 1e-12)


def density_centroid(
    coords: np.ndarray,
    labels: np.ndarray,
    state: str,
    *,
    role: str = "start",
    k_neighbors: int = 15,
    quantile: float = 0.25,
) -> np.ndarray:
    """
    Low-density centroid (start) or high-density centroid (end) for a stage label.

    Centroid is the mean position of cells in the bottom/top density quantile.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)
    mask = labels.astype(str) == str(state)
    if not np.any(mask):
        raise ValueError(f"No cells for state {state!r}")

    sub = coords[mask]
    density = _local_density(sub, k_neighbors=k_neighbors)
    q = float(quantile)
    if role == "start":
        cutoff = np.quantile(density, q) if len(density) > 1 else density[0]
        keep = density <= cutoff
    else:
        cutoff = np.quantile(density, 1.0 - q) if len(density) > 1 else density[0]
        keep = density >= cutoff
    if not np.any(keep):
        keep = np.ones(len(sub), dtype=bool)
    return np.mean(sub[keep], axis=0)


def select_density_endpoints(
    coords: np.ndarray,
    labels: np.ndarray,
    start_state: str,
    end_state: str,
    *,
    k_neighbors: int = 15,
    quantile: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (start_pos, end_pos, metadata) for LAP boundary conditions."""
    start_pos = density_centroid(
        coords, labels, start_state, role="start", k_neighbors=k_neighbors, quantile=quantile
    )
    end_pos = density_centroid(
        coords, labels, end_state, role="end", k_neighbors=k_neighbors, quantile=quantile
    )
    dist = float(np.linalg.norm(end_pos - start_pos))
    meta = {
        "selection_strategy": "density_centroid",
        "selected_mode": "density_centroid",
        "start_state": start_state,
        "end_state": end_state,
        "endpoint_distance": dist,
        "start_role": "low_density_centroid",
        "end_role": "high_density_centroid",
        "density_k_neighbors": int(k_neighbors),
        "density_quantile": float(quantile),
        "is_separable": dist > 1e-6,
    }
    return start_pos, end_pos, meta
