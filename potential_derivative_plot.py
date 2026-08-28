"""
Spline fit of potential vs pseudotime with derivative extrema and canonical transition marker.

Used by celltype LAP workflow and dataset *_analysis.py scripts.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema

from lap_helpers import path_nearest_cell_indices

PEAK_COLOR = "#c61586"
TROUGH_COLOR = "#1f77b4"
TRANSITION_LINE_COLOR = "red"


def _slug(value: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(value).strip())


def derivative_curve_plot_filename(dataset_key: str, cell_type: str) -> str:
    """Return plot basename, e.g. GSE225948_Brain_local_extrema_of_DC_derivative_curve.png."""
    return f"{dataset_key}_local_extrema_of_{_slug(cell_type)}_derivative_curve.png"


def derivative_curve_plot_path(figure_dir: str, dataset_key: str, cell_type: str) -> str:
    return f"{figure_dir}/{derivative_curve_plot_filename(dataset_key, cell_type)}"


def transition_pseudotime_from_path_result(
    adata,
    analyzer,
    path_result: dict,
    clustering_key: str = "stage",
) -> float:
    """Pseudotime of the transition-state cell nearest to the LAP path saddle."""
    indices, _, _, _ = path_nearest_cell_indices(
        analyzer, path_result, path_result["start_state"], path_result["end_state"], clustering_key
    )
    transition_idx = int(path_result["transition_state_idx"])
    global_idx = int(indices[transition_idx].flatten()[0])
    if "pseudotime" not in adata.obs:
        raise KeyError("adata.obs['pseudotime'] required for transition pseudotime")
    return float(adata.obs["pseudotime"].iloc[global_idx])


def fit_potential_spline(
    pseudotime: np.ndarray,
    potential: np.ndarray,
    *,
    smooth_factor_scale: float = 0.5,
    n_grid: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, UnivariateSpline]:
    """
    Fit U(pt) with a smoothing spline; return sorted grid, U_smooth, dU/dpt, spline object.
    """
    pt = np.asarray(pseudotime, dtype=float).reshape(-1)
    pot = np.asarray(potential, dtype=float).reshape(-1)
    mask = np.isfinite(pt) & np.isfinite(pot)
    pt, pot = pt[mask], pot[mask]
    if pt.size < 4:
        raise ValueError(f"Need at least 4 finite (pseudotime, potential) points; got {pt.size}")

    df = pd.DataFrame({"pt": pt, "pot": pot}).drop_duplicates(subset=["pt"])
    if len(df) < 4:
        raise ValueError(f"After deduplicating pseudotime, only {len(df)} points remain (need >= 4)")
    df = df.sort_values("pt").reset_index(drop=True)
    pt_sorted = df["pt"].values
    pot_sorted = df["pot"].values

    smooth_s = max(len(pt_sorted) * smooth_factor_scale, 1e-6)
    spl = UnivariateSpline(pt_sorted, pot_sorted, s=smooth_s)
    pt_grid = np.linspace(pt_sorted.min(), pt_sorted.max(), n_grid)
    pot_smooth = spl(pt_grid)
    deriv = spl.derivative()(pt_grid)
    return pt_grid, pot_smooth, deriv, spl


def find_derivative_extrema(
    pt_grid: np.ndarray,
    deriv: np.ndarray,
    *,
    order: int = 5,
) -> Dict[str, np.ndarray]:
    """Local maxima / minima of dU/d(pseudotime) on the fitted grid."""
    max_idx = argrelextrema(deriv, np.greater, order=order)[0]
    min_idx = argrelextrema(deriv, np.less, order=order)[0]
    return {
        "peak_pt": pt_grid[max_idx],
        "peak_deriv": deriv[max_idx],
        "trough_pt": pt_grid[min_idx],
        "trough_deriv": deriv[min_idx],
    }


def plot_potential_derivative_extrema_figure(
    adata,
    *,
    cell_type_label: str,
    transition_pseudotime: Optional[float] = None,
    interpretation_label: Optional[str] = None,
    potential_key: str = "potential",
    pseudotime_key: str = "pseudotime",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    smooth_factor_scale: float = 0.5,
    n_grid: int = 1000,
    figsize: Tuple[float, float] = (6.2, 4.2),
    dpi: int = 300,
    show: bool = False,
) -> Dict[str, object]:
    """
    Dual-axis plot: smoothed potential and dU/d(pseudotime) with extrema markers.

    Peak (max |dU/dpt|): magenta triangles (^). Trough (min rate): blue triangles (v).
    Canonical transition pseudotime: red solid vertical line (if provided).
    """
    if potential_key not in adata.obs:
        raise KeyError(f"Missing {potential_key!r} in adata.obs")
    if pseudotime_key not in adata.obs:
        raise KeyError(f"Missing {pseudotime_key!r} in adata.obs")

    pt = adata.obs[pseudotime_key].values
    pot = adata.obs[potential_key].values
    pt_grid, pot_smooth, deriv, spl = fit_potential_spline(
        pt, pot, smooth_factor_scale=smooth_factor_scale, n_grid=n_grid
    )
    extrema = find_derivative_extrema(pt_grid, deriv)
    peak_pt, peak_deriv = extrema["peak_pt"], extrema["peak_deriv"]
    trough_pt, trough_deriv = extrema["trough_pt"], extrema["trough_deriv"]

    print(f"[{cell_type_label}] derivative peak (pt, rate):", list(zip(peak_pt, peak_deriv)))
    print(f"[{cell_type_label}] derivative trough (pt, rate):", list(zip(trough_pt, trough_deriv)))
    if transition_pseudotime is not None and np.isfinite(transition_pseudotime):
        print(f"[{cell_type_label}] canonical transition pseudotime: {transition_pseudotime:.6g}")

    fig, ax1 = plt.subplots(figsize=figsize)
    color_pot = "#3a6ea5"
    ax1.set_xlabel("Pseudotime", fontsize=12)
    ax1.set_ylabel("Potential  U", color=color_pot, fontsize=12)
    ax1.plot(pt_grid, pot_smooth, color=color_pot, linewidth=2.4, label="Potential U", zorder=3)
    ax1.fill_between(pt_grid, pot_smooth, np.nanmin(pot_smooth),
                     color=color_pot, alpha=0.08, zorder=1)
    if len(peak_pt):
        ax1.scatter(peak_pt, spl(peak_pt), c=PEAK_COLOR, marker="^", s=90,
                    edgecolors="white", linewidths=0.8, zorder=6)
    if len(trough_pt):
        ax1.scatter(trough_pt, spl(trough_pt), c=TROUGH_COLOR, marker="v", s=90,
                    edgecolors="white", linewidths=0.8, zorder=6)
    if transition_pseudotime is not None and np.isfinite(transition_pseudotime):
        ax1.axvline(
            x=float(transition_pseudotime),
            color=TRANSITION_LINE_COLOR,
            linewidth=2.2,
            linestyle="-",
            zorder=4,
            label=interpretation_label or "Path-local maximum",
        )
    ax1.tick_params(axis="y", labelcolor=color_pot, labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)

    ax2 = ax1.twinx()
    color_deriv = "#1f2933"
    ax2.set_ylabel("Potential change rate  dU/dt", color=color_deriv, fontsize=12)
    ax2.plot(pt_grid, deriv, color=color_deriv, linewidth=2.2, linestyle="--",
             label="dU/dt", zorder=3)
    ax2.axhline(0.0, color="#b0b8c1", linewidth=0.8, zorder=1)
    if len(peak_pt):
        ax2.scatter(
            peak_pt,
            peak_deriv,
            c=PEAK_COLOR,
            marker="^",
            s=90,
            edgecolors="white",
            linewidths=0.8,
            label="Peak (max rate)",
            zorder=6,
        )
    if len(trough_pt):
        ax2.scatter(trough_pt, trough_deriv, c=TROUGH_COLOR, marker="v", s=90,
                    edgecolors="white", linewidths=0.8, label="Trough (min rate)", zorder=6)
    ax2.tick_params(axis="y", labelcolor=color_deriv, labelsize=10)

    for ax in (ax1, ax2):
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#7b8794")
            spine.set_linewidth(0.9)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        leg = ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center",
                         bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8.5,
                         frameon=False, columnspacing=1.2, handlelength=1.6)
        leg.set_zorder(10)

    plot_title = title or f"Local extrema of {cell_type_label} derivative curve"
    ax1.set_title(plot_title, fontsize=12.5, fontweight="bold", pad=34)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "pt_grid": pt_grid,
        "pot_smooth": pot_smooth,
        "deriv": deriv,
        "peak_pt": peak_pt,
        "peak_deriv": peak_deriv,
        "trough_pt": trough_pt,
        "trough_deriv": trough_deriv,
        "transition_pseudotime": transition_pseudotime,
        "n_cells": int(adata.n_obs),
    }


def plot_potential_derivative_with_canonical_path(
    adata,
    analyzer,
    canonical_path_result: dict,
    *,
    cell_type_label: str,
    clustering_key: str = "stage",
    save_path: str,
    interpretation_label: Optional[str] = None,
    **plot_kwargs,
) -> Dict[str, object]:
    """Plot spline/extrema figure and mark transition pseudotime from canonical medoid LAP."""
    ts_pt = transition_pseudotime_from_path_result(
        adata, analyzer, canonical_path_result, clustering_key=clustering_key
    )
    return plot_potential_derivative_extrema_figure(
        adata,
        cell_type_label=cell_type_label,
        transition_pseudotime=ts_pt,
        interpretation_label=interpretation_label,
        save_path=save_path,
        **plot_kwargs,
    )


def run_analysis_potential_derivative_plots(
    adata,
    *,
    dataset_key: str,
    figure_dir: str,
    cell_type: Optional[str] = None,
    cell_type_column: Optional[str] = None,
    start_state: Optional[str] = None,
    end_state: Optional[str] = None,
    clustering_key: str = "stage",
    cell_types: Optional[Sequence[str]] = None,
    file_prefix: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """
    For *_analysis.py: per cell type, compute canonical medoid LAP and save derivative plot.

    Uses the same LAP preprocessing as celltype_analysis (on a per-type copy).
    """
    from celltype_analysis import (
        DATASET_REGISTRY,
        CellTypeLAPConfig,
        _compute_path_between_states,
        _make_analyzer,
        subset_and_preprocess,
    )

    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {dataset_key!r}; choose from {sorted(DATASET_REGISTRY)}")
    profile = DATASET_REGISTRY[dataset_key]
    col = cell_type_column or profile.cell_type_column
    if col not in adata.obs.columns:
        raise KeyError(f"Cell type column {col!r} not in adata.obs")

    if cell_types is not None:
        types_to_run = list(cell_types)
    elif cell_type:
        types_to_run = [cell_type]
    else:
        types_to_run = list(profile.available_cell_types)

    results: Dict[str, Dict[str, object]] = {}

    for ct in types_to_run:
        ct_str = str(ct)
        if ct_str not in set(adata.obs[col].astype(str)):
            print(f"Skip {ct_str!r}: not in {col}")
            continue
        if start_state and end_state:
            start, end = start_state, end_state
        else:
            start, end = profile.lap_path_for_cell_type(ct_str)
        print(f"\nPotential-derivative plot: {col}={ct_str!r} ({start} → {end})")
        cfg = CellTypeLAPConfig(
            profile=profile,
            cell_type=ct_str,
            start_state=start if start_state else None,
            end_state=end if end_state else None,
            clustering_key=clustering_key,
            run_bootstrap=False,
            paths_only=True,
        )
        try:
            sub = adata[adata.obs[col].astype(str) == ct_str].copy()
            if "potential" not in sub.obs.columns:
                print(f"  Skip: no 'potential' in obs")
                continue
            sub = subset_and_preprocess(sub, cfg)
            analyzer = _make_analyzer(sub, profile)
            canonical = _compute_path_between_states(
                analyzer,
                sub,
                profile,
                cfg,
                start,
                end,
                "medoid",
            )
            slug = _slug(ct_str)
            save_path = derivative_curve_plot_path(figure_dir, dataset_key, ct_str)
            out = plot_potential_derivative_with_canonical_path(
                sub,
                analyzer,
                canonical,
                cell_type_label=ct_str,
                clustering_key=clustering_key,
                save_path=save_path,
            )
            results[ct_str] = out
            print(f"  Saved: {save_path}")
        except Exception as exc:
            print(f"  Failed for {ct_str!r}: {exc}")
            results[ct_str] = {"error": str(exc)}

    return results
