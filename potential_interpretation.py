"""
Biological interpretation scores for learned quasi-potential U.

High-level semantics (when U aligns with -log density):
- potential_relative_type / potential_deviation: departure from type-specific homeostasis
- plasticity_score: composite signal for transitional, multi-fate-capable states
- stability_score: inverse plasticity; low U attractor-like basins (recovery or pathology)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def _as_1d(values: Union[np.ndarray, pd.Series]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr


def within_type_zscore(
    values: Union[np.ndarray, pd.Series],
    cell_types: Union[np.ndarray, pd.Series],
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Z-score values within each cell type (captures stage variation inside a lineage)."""
    values = _as_1d(values)
    cell_types = np.asarray(cell_types)
    out = np.full(values.shape, np.nan, dtype=float)
    for ct in np.unique(cell_types):
        mask = cell_types == ct
        if mask.sum() < 2:
            out[mask] = 0.0
            continue
        v = values[mask]
        std = float(np.std(v))
        if std < eps:
            out[mask] = 0.0
        else:
            out[mask] = (v - float(np.mean(v))) / (std + eps)
    return out


def resolve_homeostasis_ref_time(
    times: Union[np.ndarray, pd.Series],
    ref_time: Optional[float] = None,
) -> float:
    """Reference time for homeostasis baseline (default: earliest observed time)."""
    times = _as_1d(times)
    if ref_time is not None:
        return float(ref_time)
    return float(np.nanmin(times))


def homeostasis_baseline_per_type(
    potential: Union[np.ndarray, pd.Series],
    times: Union[np.ndarray, pd.Series],
    cell_types: Union[np.ndarray, pd.Series],
    *,
    ref_time: Optional[float] = None,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Per-cell baseline U_ref(type) = mean U at the reference time within each cell type.
    """
    potential = _as_1d(potential)
    times = _as_1d(times)
    cell_types = np.asarray(cell_types)
    ref = resolve_homeostasis_ref_time(times, ref_time)
    baseline = np.full(potential.shape, np.nan, dtype=float)
    for ct in np.unique(cell_types):
        ref_mask = (cell_types == ct) & np.isclose(times, ref, atol=atol)
        if ref_mask.sum() == 0:
            ref_mask = cell_types == ct
        baseline[cell_types == ct] = float(np.nanmean(potential[ref_mask]))
    return baseline


def potential_deviation_from_homeostasis(
    potential: Union[np.ndarray, pd.Series],
    times: Union[np.ndarray, pd.Series],
    cell_types: Union[np.ndarray, pd.Series],
    *,
    ref_time: Optional[float] = None,
) -> np.ndarray:
    """U - U_ref(type): positive => departed from type homeostasis at reference time."""
    potential = _as_1d(potential)
    baseline = homeostasis_baseline_per_type(
        potential, times, cell_types, ref_time=ref_time
    )
    return potential - baseline


def transitionness_from_pseudotime(pseudotime: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Peaks at mid-pseudotime (0.5): cells between start and end states.
    Range [0, 1].
    """
    pt = _as_1d(pseudotime)
    return 1.0 - 2.0 * np.abs(pt - 0.5)


def compute_plasticity_score(
    potential_relative_type: Union[np.ndarray, pd.Series],
    potential_deviation: Union[np.ndarray, pd.Series],
    pseudotime: Union[np.ndarray, pd.Series],
    diffusion_eff: Union[np.ndarray, pd.Series],
    residual_ratio: Union[np.ndarray, pd.Series],
    *,
    w_relative: float = 0.35,
    w_deviation: float = 0.25,
    w_transition: float = 0.20,
    w_diffusion: float = 0.10,
    w_nonconservative: float = 0.10,
) -> np.ndarray:
    """
    Composite plasticity index in [0, 1] after sigmoid squashing.

    Higher => more transitional / multi-fate-capable under the model:
    - elevated type-relative potential
    - departure from reference homeostasis
    - mid-pseudotime (between attractors)
    - higher effective diffusion
    - stronger non-conservative (residual) drive
    """
    rel = _as_1d(potential_relative_type)
    dev = _as_1d(potential_deviation)
    trans = transitionness_from_pseudotime(pseudotime)
    diff = _as_1d(diffusion_eff)
    res = _as_1d(residual_ratio)

    def _z(x: np.ndarray) -> np.ndarray:
        std = float(np.std(x))
        if std < 1e-8 or not np.isfinite(std):
            return np.zeros_like(x)
        return (x - float(np.nanmean(x))) / (std + 1e-6)

    raw = (
        w_relative * _z(rel)
        + w_deviation * _z(dev)
        + w_transition * _z(trans)
        + w_diffusion * _z(diff)
        + w_nonconservative * _z(res)
    )
    return 1.0 / (1.0 + np.exp(-raw))


def compute_stability_score(plasticity_score: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Inverse of plasticity: high => settled attractor-like state (recovery or pathology)."""
    pl = _as_1d(plasticity_score)
    return 1.0 - pl


def attach_interpretation_scores(
    adata,
    *,
    time_key: str = "time",
    cell_type_key: str = "cell_type",
    potential_key: str = "potential_stationary",
    pseudotime_key: str = "pseudotime",
    diffusion_key: str = "diffusion_eff",
    residual_ratio_key: str = "residual_ratio",
    homeostasis_ref_time: Optional[float] = None,
) -> None:
    """
    Add interpretation columns to adata.obs:
    - potential_relative_type
    - potential_deviation
    - plasticity_score
    - stability_score
    """
    obs = adata.obs
    if potential_key not in obs.columns:
        potential_key = "potential"
    if potential_key not in obs.columns:
        raise KeyError("Need potential_stationary or potential in adata.obs")

    pot = obs[potential_key].astype(float).values
    times = obs[time_key].astype(float).values
    ctypes = obs[cell_type_key].values

    rel = within_type_zscore(pot, ctypes)
    dev = potential_deviation_from_homeostasis(
        pot, times, ctypes, ref_time=homeostasis_ref_time
    )
    obs["potential_relative_type"] = rel
    obs["potential_deviation"] = dev

    pseudo = obs[pseudotime_key].astype(float).values if pseudotime_key in obs.columns else np.full(len(obs), 0.5)
    diff = obs[diffusion_key].astype(float).values if diffusion_key in obs.columns else np.zeros(len(obs))
    res = (
        obs[residual_ratio_key].astype(float).values
        if residual_ratio_key in obs.columns
        else np.full(len(obs), 0.5)
    )

    pl = compute_plasticity_score(rel, dev, pseudo, diff, res)
    obs["plasticity_score"] = pl
    obs["stability_score"] = compute_stability_score(pl)

    adata.uns.setdefault("potential_interpretation", {})
    adata.uns["potential_interpretation"].update(
        {
            "potential_key": potential_key,
            "homeostasis_ref_time": resolve_homeostasis_ref_time(times, homeostasis_ref_time),
            "semantics": {
                "potential_relative_type": "within-type z-scored U; high => rare/transitional within lineage",
                "potential_deviation": "U - U_ref(type) at reference time; high => departed homeostasis",
                "plasticity_score": "composite transitional index in [0,1]",
                "stability_score": "1 - plasticity; high => attractor-like endpoint",
            },
        }
    )
