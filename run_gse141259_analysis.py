#!/usr/bin/env python
"""
GSE141259 lung-injury complete analysis pipeline (Club_cells + AT2).

Pipeline
--------
1. **Non-conservative vector field** (`VectorField.py`)
   Latent-space streamlines for ``club_cells`` and ``alv_epithelium`` (AT2),
   highlighting convergence toward the high-potential Krt8+ transitional state.

2. **Path uncertainty Ω(φ)** (`path_uncertainty.py`)
   D0→D28 remodeling LAP + bootstrap endpoints → path-level and local Ω(φ).

3. **Switch-gene screen** (`potential_derivative_plot.py`)
   At the highest-Ω fate-decision locus (aligned with |dU/d(pseudotime)| peaks),
   rank genes with abrupt expression slope changes.

4. **Pioneer TFs** (`PioneerGene.py` + ``allTFs_mm.txt``)
   Rank mouse transcription factors that act during the potential-climb phase
   before the LAP transition barrier.

Example
-------
python run_gse141259_analysis.py \\
  --bootstrap-n 15 --n-permutations 20
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from plot_utils import (
    ACCENT_HI,
    PALETTE,
    configure_headless,
    gradient_barh,
    polished_colorbar,
    style_axis,
)

configure_headless()

from PioneerGene import PioneerGeneIdentifier
from VectorField import VectorFieldAnalyzer
from celltype_analysis import (
    CellTypeLAPConfig,
    GSE141259_PROFILE,
    _compute_path_between_states,
    _make_analyzer,
    _path_n_points,
    load_annotated_adata,
    subset_and_preprocess,
)
from landscape_core import stage_core_cell_indices
from lap_helpers import project_path_to_display_space
from dataset_pipeline import PROJECT_ROOT, resolve_checkpoint_dir
from analysis_protocol_utils import (
    annotate_tf_families,
    bootstrap_geodesic_paths,
    fig_path,
    geodesic_path,
    init_protocol_outdir,
    knn_potential_gradient,
    result_path,
    select_fate_core_indices,
    write_json,
    write_output_file_index,
)
from landscape_core import mean_neighbor_spacing
from path_uncertainty import _interp_path, compute_path_uncertainty
from potential_derivative_plot import (
    fit_potential_spline,
    find_derivative_extrema,
    plot_potential_derivative_extrema_figure,
    transition_pseudotime_from_path_result,
)

FOCUS_CELL_TYPES = ("club_cells", "alv_epithelium")
CELL_TYPE_DISPLAY = {
    "club_cells": "Club_cells",
    "alv_epithelium": "alv_epithelium (AT2)",
}
DEFAULT_CHECKPOINT = (
    "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
KRT8_ALIASES = ("Krt8", "KRT8", "krt8")
TF_LIST = PROJECT_ROOT / "allTFs_mm.txt"
# 2D latent LAP: LinearND in ≥10D is impractically slow for this dataset size.
DEFAULT_LAP_N_PCS = 2


def _display_name(cell_type: str) -> str:
    return CELL_TYPE_DISPLAY.get(str(cell_type), str(cell_type))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve_ckpt(override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p.resolve()
    default = PROJECT_ROOT / DEFAULT_CHECKPOINT
    if default.exists():
        return default
    return Path(resolve_checkpoint_dir(GSE141259_PROFILE.spec)).resolve()


def _name_stem(tag: str) -> str:
    """Descriptive file-name stem for flat protocol outputs."""
    t = str(tag)
    if t in ("club_AT2_combined", "_combined"):
        return "combined_Club_AT2"
    if t.startswith("Krt8_to_"):
        return f"fate_{t}"
    return f"remodeling_{t}"


def _find_gene(names, aliases: Sequence[str]) -> Optional[str]:
    name_set = set(map(str, names))
    for g in aliases:
        if g in name_set:
            return g
    lower = {str(g).lower(): str(g) for g in names}
    for g in aliases:
        if g.lower() in lower:
            return lower[g.lower()]
    return None


def _expr_matrix(adata) -> np.ndarray:
    x = adata.X
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=float)
    return np.asarray(x, dtype=float)


def attach_marker_expression(adata_sub, adata_full, aliases: Sequence[str], obs_key: str) -> Optional[str]:
    """Copy marker expression from full AnnData onto subset obs (survives HVG filter)."""
    gene = _find_gene(adata_full.var_names, aliases)
    if gene is None:
        return None
    shared = adata_sub.obs_names.isin(adata_full.obs_names)
    if not shared.any():
        return None
    full_idx = adata_full.obs_names.get_indexer(adata_sub.obs_names)
    col = adata_full.var_names.get_loc(gene)
    x = adata_full.X
    vals = np.full(adata_sub.n_obs, np.nan, dtype=float)
    ok = full_idx >= 0
    if hasattr(x, "toarray"):
        # sparse row slice
        rows = full_idx[ok]
        vals[ok] = np.asarray(x[rows, col].toarray(), dtype=float).reshape(-1)
    else:
        vals[ok] = np.asarray(x[full_idx[ok], col], dtype=float).reshape(-1)
    adata_sub.obs[obs_key] = vals
    return gene


# ---------------------------------------------------------------------------
# Non-conservative velocity proxy for VectorField
# ---------------------------------------------------------------------------

def estimate_embedding_velocity(
    positions: np.ndarray,
    potential: np.ndarray,
    pseudotime: np.ndarray,
    *,
    n_neighbors: int = 30,
) -> np.ndarray:
    """
    Heuristic non-conservative flow in embedding space.

    Average displacement toward neighbors with higher pseudotime (injury
    progression), plus a small anti-gradient term so the field is not pure −∇U.
    """
    positions = np.asarray(positions, dtype=float)
    pot = np.asarray(potential, dtype=float).reshape(-1)
    pt = np.asarray(pseudotime, dtype=float).reshape(-1)
    n = positions.shape[0]
    k = min(n_neighbors, max(2, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k).fit(positions)
    _, idx = nbrs.kneighbors(positions)

    vel = np.zeros_like(positions)
    for i in range(n):
        neigh = idx[i, 1:]
        later = neigh[pt[neigh] >= pt[i] - 1e-9]
        if later.size == 0:
            later = neigh
        disp = positions[later] - positions[i]
        w = np.clip(pt[later] - pt[i], 0.0, None) + 0.05
        vel[i] = (disp * w[:, None]).sum(axis=0) / (w.sum() + 1e-8)

    try:
        grad = knn_potential_gradient(positions, pot, n_neighbors=k)
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8
        vel = vel - 0.15 * grad / grad_norm
    except Exception:
        pass
    return vel


def _resolve_streamline_embedding(adata, embedding_key: str = "auto") -> tuple[np.ndarray, str, str, str]:
    """Prefer 2D latent PCA for 隐空间 streamlines; fall back to UMAP."""
    if embedding_key == "auto":
        if "X_latent_pca" in adata.obsm and adata.obsm["X_latent_pca"].shape[1] >= 2:
            embedding_key = "X_latent_pca"
        elif "_lap_compute_slice" in adata.obsm:
            embedding_key = "_lap_compute_slice"
        else:
            embedding_key = "X_umap"
    coords = np.asarray(adata.obsm[embedding_key][:, :2], dtype=float)
    if embedding_key in ("X_latent_pca", "_lap_compute_slice"):
        return coords, embedding_key, "Latent PC1", "Latent PC2"
    return coords, embedding_key, "UMAP1", "UMAP2"


def plot_vector_field_streamlines(
    adata,
    *,
    cell_type_label: str,
    out_dir: Path,
    krt8_obs_key: Optional[str] = None,
    krt8_gene: Optional[str] = None,
    potential_key: str = "potential_stationary",
    embedding_key: str = "auto",
    n_streamlines: int = 40,
) -> Dict[str, object]:
    """Draw latent/UMAP streamlines; mark high-U Krt8+ cells when available."""
    if potential_key not in adata.obs.columns:
        potential_key = "potential"
    coords, emb_key, xlab, ylab = _resolve_streamline_embedding(adata, embedding_key)
    pot = adata.obs[potential_key].astype(float).values
    pt = (
        adata.obs["pseudotime"].astype(float).values
        if "pseudotime" in adata.obs.columns
        else np.zeros(adata.n_obs)
    )
    finite = np.isfinite(coords).all(axis=1) & np.isfinite(pot)
    if "pseudotime" in adata.obs.columns:
        finite &= np.isfinite(pt)
    if finite.sum() < 50:
        raise ValueError(f"Too few finite cells for vector field: {finite.sum()}")
    if not finite.all():
        adata = adata[finite].copy()
        coords = coords[finite]
        pot = pot[finite]
        pt = pt[finite]
    velocities = estimate_embedding_velocity(coords, pot, pt)

    vfa = VectorFieldAnalyzer(n_neighbors=30, grid_points=80, max_interpolation_points=4000)
    field = vfa.compute_vector_field_dynamo_style(
        coords, velocities, potentials=pot.reshape(-1, 1), n_dims=2
    )
    streams = vfa.compute_streamlines(n_streamlines=n_streamlines, max_length=400)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    if field["potential_field"] is not None:
        pf = np.asarray(field["potential_field"], dtype=float)
        try:
            from scipy.ndimage import gaussian_filter

            pf = gaussian_filter(pf, sigma=1.2, mode="nearest")
        except Exception:
            pass
        ax.contourf(
            field["grid_x"],
            field["grid_y"],
            pf,
            levels=14,
            cmap="Spectral_r",
            alpha=0.35,
            zorder=0,
        )
    scatt = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=pot,
        s=9,
        cmap="viridis",
        alpha=0.75,
        linewidths=0,
        zorder=2,
    )
    polished_colorbar(scatt, ax, label=_display_name(potential_key))

    def _smooth_path(a: np.ndarray, w: int = 5) -> np.ndarray:
        if len(a) < w:
            return a
        ker = np.ones(w) / w
        xs = np.convolve(a[:, 0], ker, mode="valid")
        ys = np.convolve(a[:, 1], ker, mode="valid")
        return np.column_stack([xs, ys])

    for i, sl in enumerate(streams):
        arr = np.asarray(sl, dtype=float)
        if len(arr) < 4:
            continue
        arr = _smooth_path(arr)
        if len(arr) < 2:
            continue
        ax.plot(arr[:, 0], arr[:, 1], color="#37474f", lw=0.8, alpha=0.45, zorder=3)
        # Sparse, small direction arrows to avoid dense black blobs
        if i % 2 == 0:
            ax.annotate(
                "",
                xy=arr[-1],
                xytext=arr[-2],
                arrowprops=dict(arrowstyle="-|>", color="#37474f",
                                lw=0.8, alpha=0.6, mutation_scale=8),
                zorder=4,
            )

    label_gene = krt8_gene or "Krt8"
    marker = None
    if krt8_obs_key and krt8_obs_key in adata.obs.columns:
        expr = adata.obs[krt8_obs_key].astype(float).values
        finite = np.isfinite(expr)
        if finite.any():
            thr = np.nanpercentile(expr, 75)
            high_u = pot >= np.percentile(pot, 70)
            marker = finite & (expr >= thr) & high_u
    elif krt8_gene and krt8_gene in adata.var_names:
        expr = _expr_matrix(adata)[:, list(adata.var_names).index(krt8_gene)]
        thr = np.percentile(expr, 75)
        high_u = pot >= np.percentile(pot, 70)
        marker = (expr >= thr) & high_u

    sink = None
    if marker is not None and marker.any():
        ax.scatter(
            coords[marker, 0],
            coords[marker, 1],
            s=28,
            facecolors="none",
            edgecolors="#ff4d6d",
            linewidths=0.9,
            label=f"high-U {label_gene}+ (p75)",
            zorder=5,
        )
        try:
            sink = vfa.sink_strength_at_points(coords[marker])
            ax.set_title(
                f"{_display_name(cell_type_label)}: non-conservative streamlines ({emb_key})\n"
                f"toward high-U {label_gene}+ | sink_strength={sink['sink_strength']:.3g} "
                f"(inflow={sink['inflow_fraction']:.2f})",
                fontsize=9,
            )
        except Exception as exc:
            warnings.warn(f"Sink metric failed: {exc}", UserWarning)
        ax.legend(frameon=False, fontsize=7, loc="best")
    else:
        ax.set_title(
            f"{_display_name(cell_type_label)}: non-conservative streamlines ({emb_key})\n"
            f"vector trend toward high-U {label_gene}+ transitional state",
            fontsize=9,
        )

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    style_axis(ax, grid_axis="none")
    ax.margins(0.04)
    fig.tight_layout()
    out_fig = fig_path(out_dir, f"{_name_stem(cell_type_label)}_vectorfield_streamlines.png")
    fig.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[vectorfield] saved {out_fig}")
    if sink:
        print(
            f"[sink] strength={sink['sink_strength']:.4f} "
            f"mean_div={sink['mean_divergence']:.4f} inflow={sink['inflow_fraction']:.3f}"
        )
    return {
        "figure": str(out_fig),
        "n_streamlines": len(streams),
        "embedding_key": emb_key,
        "krt8_gene": krt8_gene,
        "n_highlighted": int(marker.sum()) if marker is not None else 0,
        "sink": sink,
    }


def plot_combined_epithelial_vectorfield(
    adata_full,
    *,
    out_dir: Path,
    cell_types: Sequence[str] = FOCUS_CELL_TYPES,
    potential_key: str = "potential_stationary",
) -> Optional[str]:
    """Joint Club + AT2 streamline plot on a shared UMAP."""
    col = GSE141259_PROFILE.cell_type_column
    mask = adata_full.obs[col].astype(str).isin(list(cell_types))
    if int(mask.sum()) < 50:
        warnings.warn("Too few Club/AT2 cells for combined vector field.", UserWarning)
        return None
    sub = adata_full[mask].copy()
    if "X_umap" not in sub.obsm:
        if "X_pca" not in sub.obsm:
            sc.pp.pca(sub, n_comps=min(50, max(2, sub.n_vars - 1)))
        sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca")
        sc.tl.umap(sub)
    krt8 = attach_marker_expression(sub, adata_full, KRT8_ALIASES, "Krt8_expr")
    pot = potential_key if potential_key in sub.obs.columns else "potential"
    result = plot_vector_field_streamlines(
        sub,
        cell_type_label="club_AT2_combined",
        out_dir=out_dir,
        krt8_obs_key="Krt8_expr" if krt8 else None,
        krt8_gene=krt8,
        potential_key=pot,
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    coords, emb_key, xlab, ylab = _resolve_streamline_embedding(sub, "auto")
    for ct, color in zip(cell_types, (PALETTE[0], PALETTE[1])):
        m = sub.obs[col].astype(str).values == ct
        ax.scatter(
            coords[m, 0], coords[m, 1], s=12, c=color, alpha=0.7,
            label=_display_name(ct), linewidths=0.2, edgecolors="white",
        )
    ax.legend(loc="upper right", markerscale=1.6, fontsize=9)
    ax.set_title(
        "Club + AT2 epithelium: joint non-conservative flow\n"
        "toward the high-U Krt8$^+$ transitional basin",
        fontsize=12,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    style_axis(ax, grid_axis="none")
    ax.margins(0.04)
    fig.tight_layout()
    overview = fig_path(out_dir, "combined_Club_AT2_embedding_overview.png")
    fig.savefig(overview, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return result["figure"]


# ---------------------------------------------------------------------------
# Fast bootstrap LAPs (lighter than celltype_analysis.bootstrap_canonical_paths)
# ---------------------------------------------------------------------------

def bootstrap_paths_fast(
    analyzer,
    adata,
    profile,
    cfg: CellTypeLAPConfig,
    start_state: str,
    end_state: str,
    *,
    n_bootstrap: int,
    n_points: int = 8,
    mode: str = "endpoint_geodesic",
) -> dict:
    """
    Resample stage-core endpoints for Ω(φ).

    mode='endpoint_geodesic' (default): linear compute-space segments between
    resampled cores (fast; captures endpoint/geometry uncertainty).
    mode='lap': full Hamiltonian LAPs (slow, ~45s each on club_cells).
    """
    compute_key = getattr(analyzer, "embedding_2d_key", profile.lap_embedding_key)
    display_key = profile.lap_display_key
    positions = np.asarray(adata.obsm[compute_key], dtype=float)
    labels = adata.obs[cfg.clustering_key].values
    pseudotime = (
        np.asarray(adata.obs["pseudotime"].values, dtype=float)
        if "pseudotime" in adata.obs
        else None
    )
    start_core = stage_core_cell_indices(
        positions,
        labels,
        start_state,
        pseudotime=pseudotime,
        core_fraction=profile.bootstrap_core_fraction,
        by="medoid",
    )
    end_core = stage_core_cell_indices(
        positions,
        labels,
        end_state,
        pseudotime=pseudotime,
        core_fraction=profile.bootstrap_core_fraction,
        by="medoid",
    )
    if len(start_core) < 2 or len(end_core) < 2:
        warnings.warn("Too few core cells for bootstrap; skipping.", UserWarning)
        return {}

    rng = np.random.default_rng(cfg.seed)
    display_paths = []
    compute_paths = []
    n_points = int(max(5, min(n_points, _path_n_points(adata, profile))))
    for i in range(int(n_bootstrap)):
        print(f"  bootstrap ({mode}): {i + 1}/{n_bootstrap}", flush=True)
        s_idx = int(rng.choice(start_core))
        e_idx = int(rng.choice(end_core))
        try:
            if mode == "lap":
                path = analyzer.compute_least_action_path(
                    positions[s_idx],
                    positions[e_idx],
                    n_points=n_points,
                    use_3d=False,
                )
                arr = np.asarray(path["path"], dtype=float)
            else:
                arr = np.linspace(positions[s_idx], positions[e_idx], n_points)
            compute_paths.append(arr)
            if display_key in adata.obsm:
                display_paths.append(
                    project_path_to_display_space(arr, positions, adata.obsm[display_key])
                )
            else:
                display_paths.append(arr[:, :2])
        except Exception as exc:
            warnings.warn(f"Bootstrap path {i + 1} failed: {exc}", UserWarning)
            continue
    return {
        "display_paths": display_paths,
        "bootstrap_paths": compute_paths,
        "n_success": len(display_paths),
        "bootstrap_mode": mode,
    }


# ---------------------------------------------------------------------------
# Path uncertainty Ω(φ)
# ---------------------------------------------------------------------------

def local_uncertainty_along_path(
    canonical_path: np.ndarray,
    bootstrap_paths: Sequence,
    *,
    n_points: int = 100,
) -> np.ndarray:
    """Per-point normalized variance across bootstrap paths (local Ω profile)."""
    can = _interp_path(canonical_path, n_points=n_points)
    boots = []
    for p in bootstrap_paths:
        if isinstance(p, dict):
            p = p.get("path_compute", p.get("path"))
        if p is None:
            continue
        boots.append(_interp_path(p, n_points=n_points))
    if not boots:
        return np.zeros(n_points)
    arr = np.stack(boots, axis=0)
    var = np.sum((arr - arr.mean(axis=0)[None, :, :]) ** 2, axis=2).mean(axis=0)
    scale = float(np.mean(np.sum(np.diff(can, axis=0) ** 2, axis=1)) + 1e-8)
    return np.clip(var / scale, 0.0, None)


def compute_omega_for_path(
    path_result: dict,
    adata,
    analyzer,
    bootstrap_bundle: Optional[dict],
    *,
    use_display_space: bool = True,
) -> Dict[str, object]:
    """
    Ω(φ) in a consistent embedding.

    Bootstrap paths from celltype_analysis are stored in display UMAP space,
    so default Omega uses path + display_paths + X_umap. Manifold/endpoint
    terms then match that space.
    """
    if use_display_space and "X_umap" in adata.obsm:
        path = np.asarray(path_result.get("path", path_result.get("path_compute")), dtype=float)
        if path.ndim == 2 and path.shape[1] > 2:
            path = path[:, :2]
        coords = np.asarray(adata.obsm["X_umap"][:, :2], dtype=float)
    else:
        path = np.asarray(path_result.get("path_compute", path_result["path"]), dtype=float)
        coords = np.asarray(analyzer.cell_positions_2d, dtype=float)

    spacing = mean_neighbor_spacing(coords)
    boots = None
    if bootstrap_bundle:
        boots = (
            bootstrap_bundle.get("display_paths")
            or bootstrap_bundle.get("bootstrap_paths")
            or bootstrap_bundle.get("paths")
        )
    start_state = path_result.get("start_state", "D0")
    end_state = path_result.get("end_state", "D28")
    start_mask = adata.obs["stage"].astype(str).values == str(start_state)
    end_mask = adata.obs["stage"].astype(str).values == str(end_state)
    # Path-aligned τ via nearest cells (full obs pseudotime must not be passed raw).
    path_tau = None
    if "pseudotime" in adata.obs and len(path) >= 2 and len(coords) > 0:
        nbrs = NearestNeighbors(n_neighbors=1).fit(coords)
        _, ix = nbrs.kneighbors(path)
        path_tau = adata.obs["pseudotime"].astype(float).values[ix[:, 0]]
    omega = compute_path_uncertainty(
        path,
        bootstrap_paths=boots,
        cell_coords=coords,
        pseudotime=path_tau,
        start_coords=coords[start_mask] if start_mask.any() else None,
        end_coords=coords[end_mask] if end_mask.any() else None,
        neighbor_spacing=spacing,
        path_degenerate=bool(path_result.get("path_degenerate", False)),
    )
    local = local_uncertainty_along_path(path, boots or [], n_points=100)
    omega["local_omega_profile"] = local
    omega["local_omega_argmax"] = int(np.argmax(local)) if local.size else -1
    omega["local_omega_max"] = float(local.max()) if local.size else float("nan")
    return omega


def plot_omega_profile(local_omega: np.ndarray, out_path: Path, cell_type: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.linspace(0, 1, len(local_omega))
    ax.plot(x, local_omega, color=PALETTE[0], lw=2.2, zorder=3)
    ax.fill_between(x, local_omega, np.nanmin(local_omega), color=PALETTE[0], alpha=0.12, zorder=1)
    imax = int(np.argmax(local_omega))
    xmax = imax / max(len(local_omega) - 1, 1)
    ax.axvline(xmax, color=ACCENT_HI, ls="--", lw=1.8, zorder=2, label="max Ω (bifurcation)")
    ax.scatter([xmax], [local_omega[imax]], s=55, color=ACCENT_HI,
               edgecolors="white", linewidths=1.0, zorder=4)
    ax.set_xlabel("Normalized path arc-length")
    ax.set_ylabel("Local path uncertainty  Ω(φ)")
    ax.set_title(f"{_display_name(cell_type)}: path uncertainty profile", fontsize=12)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Switch genes at high-Ω / derivative peak locus
# ---------------------------------------------------------------------------

def screen_switch_genes(
    adata,
    analyzer,
    path_result: dict,
    omega: dict,
    *,
    potential_key: str = "potential",
    top_n: int = 30,
    window: int = 5,
) -> pd.DataFrame:
    """
    Genes with steep expression change near the fate-decision locus:
    max(local Ω) ∩ high |dU/d(pseudotime)| neighborhood along the LAP.
    """
    path = np.asarray(path_result.get("path_compute", path_result["path"]), dtype=float)
    coords = np.asarray(analyzer.cell_positions_2d, dtype=float)
    if path.shape[1] != coords.shape[1]:
        # fall back to display space consistency
        path = np.asarray(path_result.get("path", path), dtype=float)[:, :2]
        coords = np.asarray(adata.obsm["X_umap"][:, :2], dtype=float)

    nbrs = NearestNeighbors(n_neighbors=1).fit(coords)
    _, idx = nbrs.kneighbors(path)
    cell_idx = idx.reshape(-1)
    expr = _expr_matrix(adata)[cell_idx]

    n_path = expr.shape[0]
    dec_idx = int(path_result.get("transition_state_idx", n_path // 2))
    local = omega.get("local_omega_profile")
    if local is not None and len(local) > 1:
        frac = float(omega["local_omega_argmax"]) / max(len(local) - 1, 1)
        dec_idx = int(np.clip(round(frac * (n_path - 1)), 0, n_path - 1))

    deriv_peak_pt = np.nan
    if "pseudotime" in adata.obs and potential_key in adata.obs:
        try:
            pt_grid, _, deriv, _ = fit_potential_spline(
                adata.obs["pseudotime"].values, adata.obs[potential_key].values
            )
            extrema = find_derivative_extrema(pt_grid, deriv)
            if len(extrema["peak_pt"]):
                j = int(np.argmax(np.abs(extrema["peak_deriv"])))
                deriv_peak_pt = float(extrema["peak_pt"][j])
        except Exception as exc:
            warnings.warn(f"Spline derivative failed: {exc}", UserWarning)

    lo = max(0, dec_idx - window)
    hi = min(n_path - 1, dec_idx + window)
    if hi <= lo:
        return pd.DataFrame()

    t = np.arange(lo, hi + 1, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-8)
    slopes = expr[lo : hi + 1].T @ t / (t @ t)
    mean_expr = expr[lo : hi + 1].mean(axis=0)
    abs_slope = np.abs(slopes)

    pt_along = (
        adata.obs["pseudotime"].astype(float).values[cell_idx]
        if "pseudotime" in adata.obs
        else np.linspace(0, 1, n_path)
    )
    pt_focus = float(pt_along[dec_idx])
    if np.isfinite(deriv_peak_pt):
        prox = float(np.exp(-((pt_focus - deriv_peak_pt) / (np.std(pt_along) + 1e-6)) ** 2))
    else:
        prox = 1.0

    score = abs_slope * (mean_expr + 1e-3) * prox
    order = np.argsort(-score)[:top_n]
    rows = []
    for rank, gidx in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "gene": str(adata.var_names[gidx]),
                "switch_score": float(score[gidx]),
                "abs_slope": float(abs_slope[gidx]),
                "signed_slope": float(slopes[gidx]),
                "mean_expr_window": float(mean_expr[gidx]),
                "decision_path_index": dec_idx,
                "decision_pseudotime": pt_focus,
                "deriv_peak_pseudotime": deriv_peak_pt,
                "local_omega_max": omega.get("local_omega_max", np.nan),
                "path_omega": omega.get("path_uncertainty", np.nan),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fate bifurcation: Krt8 ADI → AT1 vs Fibroblast/Myofibroblast
# ---------------------------------------------------------------------------

FATE_START = "Krt8 ADI"
FATE_END_AT1 = "AT1 cells"
FATE_END_FIBRO = "Fibroblast_pathology"
FATE_FIBRO_SOURCES = ("Fibroblasts", "Myofibroblasts")


def prepare_fate_branch_adata(adata_full, checkpoint: Path):
    """Joint subspace of Krt8 ADI + AT1 + Fibroblasts/Myofibroblasts."""
    ct = adata_full.obs["cell.type"].astype(str)
    keep = ct.isin((FATE_START, FATE_END_AT1) + FATE_FIBRO_SOURCES)
    if int(keep.sum()) < 40:
        raise RuntimeError(f"Too few fate cells: {int(keep.sum())}")
    adata = adata_full[keep].copy()
    fate = ct[keep].astype(object).values.copy()
    fate[np.isin(fate, FATE_FIBRO_SOURCES)] = FATE_END_FIBRO
    adata.obs["fate_label"] = pd.Categorical(fate)
    print("Fate label counts:\n", adata.obs["fate_label"].value_counts())

    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
    adata = adata[:, adata.var.highly_variable].copy()
    sc.tl.pca(adata, n_comps=40)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata, n_components=2)
    try:
        from latent_embeddings import ensure_latent_embeddings

        ensure_latent_embeddings(adata, checkpoint_dir=str(checkpoint), warn=True)
    except Exception as exc:
        warnings.warn(f"Latent embeddings for fate branch: {exc}", UserWarning)
    if "X_latent_pca" in adata.obsm:
        adata.obsm["_lap_compute_slice"] = np.asarray(adata.obsm["X_latent_pca"][:, :2], dtype=float)
    else:
        adata.obsm["_lap_compute_slice"] = np.asarray(adata.obsm["X_pca"][:, :2], dtype=float)

    pot_key = "potential_stationary" if "potential_stationary" in adata.obs else "potential"
    adata.obs["potential"] = adata.obs[pot_key].astype(float)
    return adata


def analyze_fate_branch(
    adata_full,
    *,
    checkpoint: Path,
    out_root: Path,
    end_label: str,
    bootstrap_n: int,
    n_permutations: int,
    top_n_switch: int,
    top_n_pioneer: int,
    seed: int,
) -> dict:
    """Krt8 ADI → AT1 or → Fibroblast_pathology path with Ω / switch / pioneer."""
    tag = f"Krt8_to_{'AT1' if end_label == FATE_END_AT1 else 'Fibro'}"
    out_dir = out_root  # flat layout
    report: Dict[str, object] = {
        "branch": tag,
        "start": FATE_START,
        "end": end_label,
    }
    print(f"\n======== Fate branch {FATE_START} → {end_label} ========")
    adata = prepare_fate_branch_adata(adata_full, checkpoint)
    if FATE_START not in set(adata.obs["fate_label"].astype(str)):
        raise RuntimeError(f"Missing start label {FATE_START}")
    if end_label not in set(adata.obs["fate_label"].astype(str)):
        raise RuntimeError(f"Missing end label {end_label}")

    # Plasticity-filtered transitional pool for pioneer later
    plas = (
        adata.obs["plasticity_score"].astype(float).values
        if "plasticity_score" in adata.obs
        else np.zeros(adata.n_obs)
    )
    pot = adata.obs["potential"].astype(float).values
    positions = np.asarray(adata.obsm["_lap_compute_slice"], dtype=float)
    labels = adata.obs["fate_label"].astype(str).values

    start_core = select_fate_core_indices(
        positions,
        labels,
        FATE_START,
        potential=pot,
        plasticity=plas,
        prefer_high_potential=True,
        prefer_high_plasticity=True,
        core_fraction=0.25,
        min_cells=8,
    )
    end_core = select_fate_core_indices(
        positions,
        labels,
        end_label,
        potential=pot,
        prefer_high_potential=False,
        core_fraction=0.25,
        min_cells=5,
    )
    report["n_start_core"] = int(len(start_core))
    report["n_end_core"] = int(len(end_core))
    if len(start_core) < 2 or len(end_core) < 1:
        raise RuntimeError(f"Insufficient cores for {tag}: start={len(start_core)} end={len(end_core)}")

    start_pos = positions[start_core].mean(axis=0)
    end_pos = positions[end_core].mean(axis=0)

    # Prefer full LAP when analyzer available; else geodesic
    profile = replace(GSE141259_PROFILE, lap_n_pcs=2, max_path_points=12, bootstrap_n=bootstrap_n)
    path_result = None
    analyzer = None
    try:
        analyzer = _make_analyzer(adata, profile)
        n_points = _path_n_points(adata, profile)
        path_result = analyzer.compute_least_action_path(
            start_pos, end_pos, n_points=n_points, use_3d=False
        )
        path_result["start_state"] = FATE_START
        path_result["end_state"] = end_label
        path_result["path_compute"] = np.asarray(path_result.get("path"), dtype=float)
        if "X_umap" in adata.obsm:
            path_result["path"] = project_path_to_display_space(
                path_result["path_compute"],
                positions,
                adata.obsm["X_umap"],
            )
        report["total_action"] = float(path_result.get("total_action", np.nan))
        report["transition_state_idx"] = int(path_result.get("transition_state_idx", -1))
        report["path_method"] = "lap"
    except Exception as exc:
        warnings.warn(f"LAP failed for {tag}, using geodesic: {exc}", UserWarning)
        arr = geodesic_path(start_pos, end_pos, n_points=12)
        nbrs = NearestNeighbors(n_neighbors=1).fit(positions)
        _, nn = nbrs.kneighbors(arr)
        u_along = pot[nn[:, 0]]
        path_result = {
            "path": arr,
            "path_compute": arr,
            "start_state": FATE_START,
            "end_state": end_label,
            "transition_state_idx": int(np.argmax(u_along)),
            "total_action": float(np.linalg.norm(end_pos - start_pos)),
            "path_degenerate": False,
        }
        report["path_method"] = "geodesic"
        report["total_action"] = path_result["total_action"]
        report["transition_state_idx"] = path_result["transition_state_idx"]
        from CellFateLandscape import NonEquilibriumCellFateLandscape

        analyzer = NonEquilibriumCellFateLandscape(
            adata,
            potential_key="potential",
            embedding_2d_key="_lap_compute_slice",
            potential_transform="none",
        )

    boots = bootstrap_geodesic_paths(
        positions, start_core, end_core, n_bootstrap=bootstrap_n, n_points=10, seed=seed
    )
    boot_bundle = {"display_paths": boots, "bootstrap_paths": boots, "n_success": len(boots)}

    omega = compute_omega_for_path(path_result, adata, analyzer, boot_bundle)
    comps = omega.get("uncertainty_components", {}) or {}

    def _comp_val(key: str):
        v = comps.get(key)
        return v.get("value", np.nan) if isinstance(v, dict) else v

    omega_row = {
        "branch": tag,
        "path_uncertainty_Omega": omega.get("path_uncertainty"),
        "path_reliability": omega.get("path_reliability"),
        "uncertainty_status": omega.get("uncertainty_status"),
        "local_omega_max": omega.get("local_omega_max"),
        "local_omega_argmax": omega.get("local_omega_argmax"),
        "bootstrap": _comp_val("bootstrap"),
        "manifold": _comp_val("manifold"),
        "drift": _comp_val("drift"),
        "endpoint": _comp_val("endpoint"),
        "n_bootstrap_paths": len(boots),
    }
    pd.DataFrame([omega_row]).to_csv(result_path(out_dir, f"{_name_stem(tag)}_path_uncertainty.csv"), index=False)
    if omega.get("local_omega_profile") is not None:
        local = np.asarray(omega["local_omega_profile"], dtype=float)
        plot_omega_profile(
            local,
            fig_path(out_dir, f"{_name_stem(tag)}_path_uncertainty_Omega_profile.png"),
            tag,
        )
        np.save(result_path(out_dir, f"{_name_stem(tag)}_local_omega.npy"), local)
        # (φ, Ω) curve for polished Fig3B redraw
        phi = np.linspace(0.0, 1.0, len(local))
        np.save(
            result_path(out_dir, f"{_name_stem(tag)}_local_omega_digitized.npy"),
            np.column_stack([phi, local]),
        )
    report["omega"] = omega_row
    print(f"[Ω] {tag}: Ω={omega_row['path_uncertainty_Omega']:.4f} status={omega_row['uncertainty_status']}")

    # Switch genes at decision locus
    switch_df = screen_switch_genes(
        adata, analyzer, path_result, omega, potential_key="potential", top_n=top_n_switch
    )
    switch_csv = result_path(out_dir, f"{_name_stem(tag)}_switch_genes.csv")
    switch_df.to_csv(switch_csv, index=False)
    report["switch_genes_top"] = switch_df.head(10).to_dict(orient="records")
    report["switch_genes_csv"] = str(switch_csv)

    # Derivative plot
    try:
        t_pseudo = transition_pseudotime_from_path_result(
            adata, analyzer, path_result, clustering_key="fate_label"
        )
    except Exception:
        t_pseudo = None
    deriv_path = str(fig_path(out_dir, f"{_name_stem(tag)}_potential_derivative_extrema.png"))
    try:
        plot_potential_derivative_extrema_figure(
            adata,
            cell_type_label=tag,
            transition_pseudotime=t_pseudo,
            interpretation_label="fate decision / high-Ω",
            potential_key="potential",
            save_path=deriv_path,
        )
        report["derivative_figure"] = deriv_path
    except Exception as exc:
        report["derivative_error"] = str(exc)

    # Pioneer on high-plasticity ∩ high-U transitional cells along path neighborhood
    plas_thr = np.nanquantile(plas[labels == FATE_START], 0.6) if (labels == FATE_START).any() else 0.5
    pot_thr = np.nanquantile(pot[labels == FATE_START], 0.6) if (labels == FATE_START).any() else np.nanmedian(pot)
    high_plas_mask = (labels == FATE_START) & (plas >= plas_thr) & (pot >= pot_thr)
    report["n_high_plasticity_transitional"] = int(high_plas_mask.sum())

    tf_path = str(TF_LIST if TF_LIST.exists() else PROJECT_ROOT / "allTFs_mm.txt")
    identifier = PioneerGeneIdentifier(analyzer, tf_gene_paths=(tf_path,), n_permutations=n_permutations)
    pioneer_out = identifier.identify_pioneer_genes_along_path(
        path_result, top_n_genes=top_n_pioneer, n_permutations=n_permutations
    )
    ranked = pioneer_out.get("pioneer_genes", {})
    transition_idx = int(pioneer_out.get("transition_idx", path_result.get("transition_state_idx", 0)))
    rows = []
    for gene, info in ranked.items():
        feats = info.get("features", {})
        max_change = feats.get("max_change_idx", np.nan)
        rows.append(
            {
                "gene": gene,
                "rank": info.get("rank"),
                "score": info.get("score"),
                "empirical_p_value": info.get("empirical_p_value", np.nan),
                "is_tf": bool(info.get("is_transcription_factor", identifier._is_transcription_factor(gene))),
                "max_change_idx": max_change,
                "transition_idx": transition_idx,
                "climb_phase": (
                    "potential_climb"
                    if (
                        isinstance(max_change, (int, float, np.integer, np.floating))
                        and np.isfinite(max_change)
                        and max_change <= transition_idx
                    )
                    else "post_barrier"
                ),
            }
        )
    pioneer_df = pd.DataFrame(rows)
    if not pioneer_df.empty:
        fam = annotate_tf_families(pioneer_df["gene"].tolist())
        pioneer_df = pioneer_df.merge(fam, on="gene", how="left")
        pioneer_df.to_csv(result_path(out_dir, f"{_name_stem(tag)}_pioneer_genes.csv"), index=False)
        climb = pioneer_df[(pioneer_df["is_tf"]) & (pioneer_df["climb_phase"] == "potential_climb")]
        climb.to_csv(result_path(out_dir, f"{_name_stem(tag)}_pioneer_TFs_climb_phase.csv"), index=False)
        report["pioneer_climb_top"] = climb.head(10).to_dict(orient="records")
        report["pioneer_literature_hits"] = pioneer_df.loc[
            pioneer_df["literature_family"].astype(str).str.len() > 0, ["gene", "literature_family"]
        ].to_dict(orient="records")

    write_json(report, result_path(out_dir, f"{_name_stem(tag)}_summary.json"))
    return report


# ---------------------------------------------------------------------------
# Per-cell-type orchestration
# ---------------------------------------------------------------------------

def analyze_cell_type(
    adata_full,
    cell_type: str,
    *,
    checkpoint: Path,
    out_root: Path,
    start: str,
    end: str,
    bootstrap_n: int,
    n_permutations: int,
    top_n_switch: int,
    top_n_pioneer: int,
    seed: int,
    potential_key: str | None = None,
) -> dict:
    base_profile = GSE141259_PROFILE
    profile = replace(
        base_profile,
        bootstrap_n=int(bootstrap_n),
        lap_n_pcs=int(DEFAULT_LAP_N_PCS),
        max_path_points=15,
    )
    cfg = CellTypeLAPConfig(
        profile=profile,
        cell_type=cell_type,
        start_state=start,
        end_state=end,
        save_dir=str(checkpoint),
        run_bootstrap=bootstrap_n > 0,
        run_go=False,
        run_de=False,
        seed=seed,
        top_n_pioneer=top_n_pioneer,
    )

    out_dir = out_root  # flat layout
    report: Dict[str, object] = {"cell_type": cell_type, "start": start, "end": end}

    print(f"\n======== {cell_type}: {start} → {end} ========")
    adata = subset_and_preprocess(adata_full, cfg)

    if potential_key is None:
        pot_key = "potential_stationary" if "potential_stationary" in adata.obs else "potential"
    else:
        pot_key = str(potential_key)
        if pot_key not in adata.obs.columns:
            raise KeyError(f"Requested potential_key={pot_key!r} missing from adata.obs")
    # Keep LAP analyzer on 'potential'; map requested field into that column.
    adata.obs["potential"] = adata.obs[pot_key].astype(float)
    report["potential_key"] = pot_key

    krt8 = attach_marker_expression(adata, adata_full, KRT8_ALIASES, "Krt8_expr")
    report["krt8_gene"] = krt8

    # 1) Vector field
    vf = plot_vector_field_streamlines(
        adata,
        cell_type_label=cell_type,
        out_dir=out_dir,
        krt8_obs_key="Krt8_expr" if krt8 else None,
        krt8_gene=krt8,
        potential_key="potential",
    )
    report["vector_field"] = vf

    # 2) LAP (+ optional bootstrap)
    analyzer = _make_analyzer(adata, profile)
    n_points = _path_n_points(adata, profile)
    print(f"Computing LAP {start} → {end} (n_points={n_points})...")
    path_result = _compute_path_between_states(
        analyzer,
        adata,
        profile,
        cfg,
        start,
        end,
        endpoint_mode="min_potential",
        n_points=n_points,
        max_iter=25,
    )
    if path_result.get("status") == "endpoint_not_separable":
        raise RuntimeError(f"Endpoint not separable for {cell_type}: {path_result.get('endpoint_selection')}")

    report["endpoint_meta"] = path_result.get("endpoint_selection")
    report["transition_state_idx"] = int(path_result.get("transition_state_idx", -1))
    report["total_action"] = float(path_result.get("total_action", np.nan))
    # Preserve the actual LAP potential profile for a separate parent-level
    # D0→D28 plot.  Keeping it in the summary avoids recomputing the expensive
    # continuous LAP merely to redraw the profile.
    lap_u = np.asarray(path_result.get("potential", []), dtype=float).reshape(-1)
    if lap_u.size:
        report["lap_U_path"] = lap_u.tolist()
        report["lap_path_progress"] = np.linspace(0.0, 1.0, lap_u.size).tolist()

    boot_bundle = None
    if bootstrap_n > 0:
        print(f"Bootstrap LAPs (fast, n={bootstrap_n})...", flush=True)
        try:
            boot_bundle = bootstrap_paths_fast(
                analyzer,
                adata,
                profile,
                cfg,
                start,
                end,
                n_bootstrap=bootstrap_n,
                n_points=min(10, n_points),
                mode="endpoint_geodesic",
            )
        except Exception as exc:
            warnings.warn(f"Bootstrap failed for {cell_type}: {exc}", UserWarning)
            boot_bundle = None

    # 3) Path uncertainty Ω(φ)
    omega = compute_omega_for_path(path_result, adata, analyzer, boot_bundle)
    comps = omega.get("uncertainty_components", {}) or {}

    def _comp_val(key: str):
        v = comps.get(key)
        if isinstance(v, dict):
            return v.get("value", np.nan)
        return v

    omega_row = {
        "cell_type": cell_type,
        "path_uncertainty_Omega": omega.get("path_uncertainty"),
        "path_reliability": omega.get("path_reliability"),
        "uncertainty_status": omega.get("uncertainty_status"),
        "local_omega_max": omega.get("local_omega_max"),
        "local_omega_argmax": omega.get("local_omega_argmax"),
        "bootstrap": _comp_val("bootstrap"),
        "manifold": _comp_val("manifold"),
        "drift": _comp_val("drift"),
        "endpoint": _comp_val("endpoint"),
        "n_bootstrap_paths": len(boot_bundle.get("display_paths", []))
        if isinstance(boot_bundle, dict)
        else 0,
    }
    pd.DataFrame([omega_row]).to_csv(result_path(out_dir, f"{_name_stem(cell_type)}_path_uncertainty.csv"), index=False)
    if omega.get("local_omega_profile") is not None:
        plot_omega_profile(
            np.asarray(omega["local_omega_profile"], dtype=float),
            fig_path(out_dir, f"{_name_stem(cell_type)}_path_uncertainty_Omega_profile.png"),
            cell_type,
        )
        np.save(result_path(out_dir, f"{_name_stem(cell_type)}_local_omega.npy"), omega["local_omega_profile"])
    report["omega"] = omega_row
    print(
        f"[Ω] path_uncertainty={omega_row['path_uncertainty_Omega']:.4f} "
        f"status={omega_row['uncertainty_status']} "
        f"local_max={omega_row['local_omega_max']:.4f}"
    )

    # 4) Potential derivative + switch genes
    try:
        t_pseudo = transition_pseudotime_from_path_result(
            adata, analyzer, path_result, clustering_key="stage"
        )
    except Exception:
        t_pseudo = None
        if omega.get("local_omega_argmax", -1) >= 0 and "pseudotime" in adata.obs:
            # Approximate decision pseudotime from Ω peak along path
            path_c = np.asarray(path_result.get("path_compute", path_result["path"]), dtype=float)
            coords_c = np.asarray(analyzer.cell_positions_2d, dtype=float)
            if path_c.shape[1] == coords_c.shape[1]:
                _, iidx = NearestNeighbors(n_neighbors=1).fit(coords_c).kneighbors(path_c)
                frac = omega["local_omega_argmax"] / max(len(omega["local_omega_profile"]) - 1, 1)
                pidx = int(round(frac * (len(path_c) - 1)))
                t_pseudo = float(adata.obs["pseudotime"].astype(float).values[int(iidx[pidx, 0])])

    deriv_path = str(fig_path(out_dir, f"{_name_stem(cell_type)}_potential_derivative_extrema.png"))
    try:
        deriv_out = plot_potential_derivative_extrema_figure(
            adata,
            cell_type_label=cell_type,
            transition_pseudotime=t_pseudo,
            interpretation_label="LAP transition / high-Ω locus",
            potential_key="potential",
            save_path=deriv_path,
        )
        report["derivative"] = {
            "figure": deriv_path,
            "n_peaks": int(len(deriv_out["peak_pt"])),
            "transition_pseudotime": t_pseudo,
        }
    except Exception as exc:
        warnings.warn(f"Derivative plot failed: {exc}", UserWarning)
        report["derivative"] = {"error": str(exc)}

    switch_df = screen_switch_genes(
        adata,
        analyzer,
        path_result,
        omega,
        potential_key="potential",
        top_n=top_n_switch,
    )
    switch_csv = result_path(out_dir, f"{_name_stem(cell_type)}_switch_genes.csv")
    switch_df.to_csv(switch_csv, index=False)
    report["switch_genes_csv"] = str(switch_csv)
    report["switch_genes_top"] = switch_df.head(10).to_dict(orient="records")
    print(f"[switch] wrote {switch_csv} ({len(switch_df)} genes)")

    # 5) Pioneer TFs (mouse TF list)
    tf_path = str(TF_LIST if TF_LIST.exists() else PROJECT_ROOT / "allTFs_mm.txt")
    identifier = PioneerGeneIdentifier(
        analyzer,
        tf_gene_paths=(tf_path,),
        n_permutations=n_permutations,
    )
    pioneer_out = identifier.identify_pioneer_genes_along_path(
        path_result, top_n_genes=top_n_pioneer, n_permutations=n_permutations
    )
    ranked = pioneer_out.get("pioneer_genes", {})
    transition_idx = int(pioneer_out.get("transition_idx", path_result.get("transition_state_idx", 0)))
    pioneer_rows = []
    for gene, info in ranked.items():
        feats = info.get("features", {})
        max_change = feats.get("max_change_idx", np.nan)
        pioneer_rows.append(
            {
                "gene": gene,
                "rank": info.get("rank"),
                "score": info.get("score"),
                "empirical_p_value": info.get("empirical_p_value", np.nan),
                "is_tf": bool(
                    info.get("is_transcription_factor", identifier._is_transcription_factor(gene))
                ),
                "mean_expression": feats.get("mean_expression", np.nan),
                "max_change_idx": max_change,
                "transition_idx": transition_idx,
                "climb_phase": (
                    "potential_climb"
                    if (isinstance(max_change, (int, float, np.integer, np.floating))
                        and np.isfinite(max_change)
                        and max_change <= transition_idx)
                    else "post_barrier"
                ),
            }
        )
    pioneer_df = pd.DataFrame(pioneer_rows)
    climb_tf_n = 0
    if not pioneer_df.empty:
        pioneer_tf = pioneer_df[pioneer_df["is_tf"]].copy()
        # Prefer TFs active during potential climb (before LAP barrier).
        if "climb_phase" in pioneer_tf.columns:
            climb = pioneer_tf[pioneer_tf["climb_phase"] == "potential_climb"].copy()
            climb_tf_n = int(len(climb))
            if climb_tf_n:
                climb = climb.sort_values("score", ascending=False)
                climb.to_csv(
                    result_path(out_dir, f"{_name_stem(cell_type)}_pioneer_TFs_climb_phase.csv"),
                    index=False,
                )
                report["pioneer_climb_top"] = climb.head(10).to_dict(orient="records")
        pioneer_tf.to_csv(result_path(out_dir, f"{_name_stem(cell_type)}_pioneer_TFs.csv"), index=False)
        pioneer_df.to_csv(result_path(out_dir, f"{_name_stem(cell_type)}_pioneer_genes_all.csv"), index=False)
        try:
            pioneer_fig = fig_path(out_dir, f"{_name_stem(cell_type)}_pioneer_TF_dynamics.png")
            fig, _axes = identifier.plot_gene_expression_dynamics(
                pioneer_out, save_path=str(pioneer_fig)
            )
            if fig is not None:
                plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Pioneer dynamics plot failed: {exc}", UserWarning)

    report["pioneer_tf_n"] = int(pioneer_df["is_tf"].sum()) if len(pioneer_df) else 0
    report["pioneer_climb_tf_n"] = climb_tf_n
    report["pioneer_top"] = pioneer_df.head(10).to_dict(orient="records")
    print(
        f"[pioneer] {report['pioneer_tf_n']} TFs in top-{top_n_pioneer}; "
        f"{climb_tf_n} during potential-climb (allTFs_mm.txt)"
    )

    with open(result_path(out_dir, f"{_name_stem(cell_type)}_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GSE141259 Club/AT2 landscape analysis")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Trained checkpoint directory")
    p.add_argument(
        "--cell-types",
        nargs="+",
        default=list(FOCUS_CELL_TYPES),
        help="metacelltype labels (default: club_cells alv_epithelium)",
    )
    p.add_argument("--start", type=str, default="D0")
    p.add_argument("--end", type=str, default="D28")
    p.add_argument("--bootstrap-n", type=int, default=8)
    p.add_argument("--n-permutations", type=int, default=20)
    p.add_argument("--top-n-switch", type=int, default=30)
    p.add_argument("--top-n-pioneer", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-combined-vectorfield",
        action="store_true",
        help="Skip joint Club+AT2 streamline figure",
    )
    p.add_argument(
        "--skip-remodeling",
        action="store_true",
        help="Skip within-type D0→D28 remodeling paths (Club/AT2)",
    )
    p.add_argument(
        "--skip-fate-branches",
        action="store_true",
        help="Skip Krt8→AT1 / Krt8→Fibro fate bifurcation analyses",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    checkpoint = _resolve_ckpt(args.checkpoint_dir)
    print(f"Checkpoint: {checkpoint}")
    out_root = init_protocol_outdir(checkpoint / "analysis_protocol_GSE141259")

    print("Loading annotated AnnData + checkpoint obs...")
    adata_full = load_annotated_adata(GSE141259_PROFILE, str(checkpoint))
    print(adata_full)
    print(
        "metacelltype counts:\n",
        adata_full.obs[GSE141259_PROFILE.cell_type_column].value_counts().head(20),
    )
    if "cell.type" in adata_full.obs:
        print(
            "cell.type (fate-relevant):\n",
            adata_full.obs["cell.type"]
            .astype(str)
            .value_counts()
            .loc[
                lambda s: s.index.isin(
                    ["Krt8 ADI", "AT1 cells", "Fibroblasts", "Myofibroblasts", "Club cells", "AT2 cells"]
                )
            ],
        )

    try:
        from latent_embeddings import ensure_latent_embeddings

        ensure_latent_embeddings(adata_full, checkpoint_dir=str(checkpoint), warn=True)
    except Exception as exc:
        warnings.warn(f"Could not load latent embeddings: {exc}", UserWarning)

    if not args.skip_combined_vectorfield:
        print("\n=== Combined Club + AT2 vector field ===")
        if "X_umap" not in adata_full.obsm and "X_latent_pca" not in adata_full.obsm:
            if "X_pca" not in adata_full.obsm:
                sc.pp.pca(adata_full, n_comps=min(50, max(2, adata_full.n_vars - 1)))
            sc.pp.neighbors(
                adata_full,
                n_neighbors=15,
                n_pcs=min(40, adata_full.obsm["X_pca"].shape[1]),
            )
            sc.tl.umap(adata_full)
        plot_combined_epithelial_vectorfield(
            adata_full,
            out_dir=out_root,
            cell_types=tuple(args.cell_types),
        )

    summaries: List[dict] = []
    omega_rows = []
    if not args.skip_remodeling:
        for ct in args.cell_types:
            try:
                rep = analyze_cell_type(
                    adata_full,
                    ct,
                    checkpoint=checkpoint,
                    out_root=out_root,
                    start=args.start,
                    end=args.end,
                    bootstrap_n=args.bootstrap_n,
                    n_permutations=args.n_permutations,
                    top_n_switch=args.top_n_switch,
                    top_n_pioneer=args.top_n_pioneer,
                    seed=args.seed,
                )
                summaries.append(rep)
                if "omega" in rep:
                    omega_rows.append(rep["omega"])
            except Exception as exc:
                warnings.warn(f"Analysis failed for {ct}: {exc}", UserWarning)
                import traceback

                traceback.print_exc()
                summaries.append({"cell_type": ct, "error": str(exc)})

    fate_reports: List[dict] = []
    if not args.skip_fate_branches and "cell.type" in adata_full.obs.columns:
        for end in (FATE_END_AT1, FATE_END_FIBRO):
            try:
                fr = analyze_fate_branch(
                    adata_full,
                    checkpoint=checkpoint,
                    out_root=out_root,
                    end_label=end,
                    bootstrap_n=args.bootstrap_n,
                    n_permutations=args.n_permutations,
                    top_n_switch=args.top_n_switch,
                    top_n_pioneer=args.top_n_pioneer,
                    seed=args.seed,
                )
                fate_reports.append(fr)
                if "omega" in fr:
                    omega_rows.append(fr["omega"])
            except Exception as exc:
                warnings.warn(f"Fate branch failed for {end}: {exc}", UserWarning)
                import traceback

                traceback.print_exc()
                fate_reports.append({"end": end, "error": str(exc)})

    if omega_rows:
        pd.DataFrame(omega_rows).to_csv(result_path(out_root, "path_uncertainty_all_paths_summary.csv"), index=False)
    write_json(
        {
            "checkpoint": str(checkpoint),
            "cell_types": list(args.cell_types),
            "start": args.start,
            "end": args.end,
            "tf_list": str(TF_LIST),
            "remodeling_reports": summaries,
            "fate_branch_reports": fate_reports,
        },
        result_path(out_root, "analysis_summary.json"),
    )
    write_output_file_index(
        out_root,
        [
            ("figures/combined_Club_AT2_vectorfield_streamlines.png",
             "Joint Club+AT2 non-conservative vector-field streamlines toward high-U Krt8+"),
            ("figures/combined_Club_AT2_embedding_overview.png",
             "Club + AT2 cells colored by type on shared embedding"),
            ("figures/remodeling_club_cells_vectorfield_streamlines.png",
             "Club_cells remodeling: streamlines toward high-U Krt8+"),
            ("figures/remodeling_club_cells_path_uncertainty_Omega_profile.png",
             "Club_cells D0→D28 local path-uncertainty Ω(φ) profile"),
            ("figures/remodeling_club_cells_potential_derivative_extrema.png",
             "Club_cells potential U and dU/dt with transition / peak markers"),
            ("figures/remodeling_club_cells_pioneer_TF_dynamics.png",
             "Club_cells pioneer TF expression dynamics along the remodeling path"),
            ("figures/remodeling_alv_epithelium_vectorfield_streamlines.png",
             "AT2 (alv_epithelium) remodeling: streamlines toward high-U Krt8+"),
            ("figures/remodeling_alv_epithelium_path_uncertainty_Omega_profile.png",
             "AT2 D0→D28 local path-uncertainty Ω(φ) profile"),
            ("figures/remodeling_alv_epithelium_potential_derivative_extrema.png",
             "AT2 potential U and dU/dt with transition / peak markers"),
            ("figures/remodeling_alv_epithelium_pioneer_TF_dynamics.png",
             "AT2 pioneer TF expression dynamics along the remodeling path"),
            ("figures/fate_Krt8_to_AT1_path_uncertainty_Omega_profile.png",
             "Fate branch Krt8 ADI→AT1: Ω(φ) profile"),
            ("figures/fate_Krt8_to_AT1_potential_derivative_extrema.png",
             "Fate branch Krt8 ADI→AT1: potential derivative extrema"),
            ("figures/fate_Krt8_to_Fibro_path_uncertainty_Omega_profile.png",
             "Fate branch Krt8 ADI→Fibroblast: Ω(φ) profile"),
            ("figures/fate_Krt8_to_Fibro_potential_derivative_extrema.png",
             "Fate branch Krt8 ADI→Fibroblast: potential derivative extrema"),
            ("path_uncertainty_all_paths_summary.csv",
             "Aggregated Ω / status for remodeling paths and fate branches"),
            ("remodeling_club_cells_*.csv/.json/.npy",
             "Club remodeling tables: path uncertainty, switch genes, pioneer TFs, summary JSON"),
            ("remodeling_alv_epithelium_*.csv/.json/.npy",
             "AT2 remodeling tables: path uncertainty, switch genes, pioneer TFs, summary JSON"),
            ("fate_Krt8_to_AT1_*.csv/.json",
             "Krt8→AT1 fate tables: uncertainty, switch genes, pioneer TFs, summary"),
            ("fate_Krt8_to_Fibro_*.csv/.json",
             "Krt8→Fibroblast fate tables: uncertainty, switch genes, pioneer TFs, summary"),
            ("analysis_summary.json",
             "Top-level summary of the full GSE141259 protocol run"),
            ("OUTPUT_FILE_INDEX.md",
             "This file: human-readable description of every output"),
        ],
    )
    print(f"\nDone. Results under: {out_root}")


if __name__ == "__main__":
    main()
