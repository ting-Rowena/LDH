"""Shared plotting and path utilities for cell-type LAP workflows."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from plot_utils import TECH_BLUE_CMAP
from scipy.signal import savgol_filter
from sklearn.neighbors import NearestNeighbors


def interpolate_path_arclength(path: np.ndarray, n_points: int) -> np.ndarray:
    """Resample path to fixed number of points by arc length."""
    path = np.asarray(path, dtype=float)
    if len(path) <= 1:
        return np.repeat(path[:1], n_points, axis=0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-12:
        return np.repeat(path[:1], n_points, axis=0)
    targets = np.linspace(0.0, total, n_points)
    out = np.zeros((n_points, path.shape[1]), dtype=float)
    for dim in range(path.shape[1]):
        out[:, dim] = np.interp(targets, cum, path[:, dim])
    return out


def project_path_to_display_space(
    path_compute: np.ndarray,
    compute_coords: np.ndarray,
    display_coords: np.ndarray,
) -> np.ndarray:
    """Map LAP points from compute space (e.g. PCA) to display space (e.g. UMAP) via NN."""
    path_compute = np.asarray(path_compute, dtype=float)
    compute_coords = np.asarray(compute_coords, dtype=float)
    display_coords = np.asarray(display_coords, dtype=float)
    nbrs = NearestNeighbors(n_neighbors=1).fit(compute_coords)
    _, idx = nbrs.kneighbors(path_compute)
    return display_coords[idx.flatten()]


def clip_path_index(idx: int, n_points: int) -> int:
    """Clamp path index to [0, n_points - 1]."""
    if n_points <= 0:
        return 0
    return max(0, min(int(idx), n_points - 1))


def path_window_index_groups(
    n_path: int,
    ts_idx: int,
    window_size: int,
) -> Dict[str, List[int]]:
    """Path-point index ranges for start / transition / end windows."""
    return {
        "start_window": list(range(0, min(window_size, n_path))),
        "transition_window": list(
            range(max(0, ts_idx - window_size), min(n_path, ts_idx + window_size + 1))
        ),
        "end_window": list(range(max(0, n_path - window_size), n_path)),
    }


def collect_knn_cells_for_path_windows(
    analyzer,
    path_result: dict,
    *,
    window_size: int = 5,
    neighbors_per_path_point: int = 10,
    interpolate_n: int = 100,
) -> Dict[str, np.ndarray]:
    """Collect unique cells near each path point using k-NN in compute space."""
    path_compute = np.asarray(path_result.get("path_compute", path_result["path"]), dtype=float)
    if interpolate_n and interpolate_n > len(path_compute):
        path = interpolate_path_arclength(path_compute, interpolate_n)
        scale = (len(path) - 1) / max(len(path_compute) - 1, 1)
        ts_idx = clip_path_index(
            int(round(int(path_result["transition_state_idx"]) * scale)),
            len(path),
        )
        win = max(window_size, int(round(window_size * scale)))
    else:
        path = path_compute
        ts_raw = int(path_result["transition_state_idx"])
        pot = path_result.get("potential")
        n_align = min(len(path), len(np.asarray(pot))) if pot is not None else len(path)
        ts_idx = clip_path_index(ts_raw, n_align)
        win = window_size

    positions = np.asarray(analyzer.cell_positions_2d, dtype=float)
    k = min(neighbors_per_path_point, len(positions))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(positions)
    regions = path_window_index_groups(len(path), ts_idx, win)

    region_cells: Dict[str, np.ndarray] = {}
    for region, idxs in regions.items():
        cell_set = set()
        for pi in idxs:
            _, nbr_idx = nbrs.kneighbors([path[int(pi)]])
            cell_set.update(int(i) for i in nbr_idx[0])
        region_cells[region] = np.array(sorted(cell_set), dtype=int)
    return region_cells


def path_result_for_display(
    path_result: dict,
    adata,
    compute_key: str = "X_pca",
    display_key: str = "X_umap",
) -> dict:
    """Copy path result with path coordinates projected to UMAP for plotting."""
    projected = project_path_to_display_space(
        path_result["path"],
        adata.obsm[compute_key],
        adata.obsm[display_key],
    )
    out = dict(path_result)
    out["path_compute"] = np.asarray(path_result["path"], dtype=float)
    out["path"] = projected
    out["compute_space"] = compute_key
    out["display_space"] = display_key
    return out


def median_bootstrap_path(
    paths: Sequence[np.ndarray],
    n_interp: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median path and 10/90 percentile envelopes in display space."""
    if not paths:
        raise ValueError("No bootstrap paths provided")
    interp = [interpolate_path_arclength(p, n_interp) for p in paths]
    stack = np.stack(interp, axis=0)
    return (
        np.median(stack, axis=0),
        np.percentile(stack, 10, axis=0),
        np.percentile(stack, 90, axis=0),
    )


def smooth_path(path: np.ndarray, smoothing_factor: float = 0.2) -> np.ndarray:
    smoothed = np.zeros_like(path)
    window = min(15, len(path))
    if window % 2 == 0:
        window = max(3, window - 1)
    for i in range(path.shape[1]):
        smoothed[:, i] = savgol_filter(path[:, i], window_length=window, polyorder=min(3, window - 1), mode="interp")
    return smoothed


def unique_ordered(arr: np.ndarray) -> np.ndarray:
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def find_unique_neighbors(query_points, cell_positions, k: int = 5) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1).fit(cell_positions)
    distances, indices = nbrs.kneighbors(query_points)
    assigned = set()
    result = np.zeros(len(query_points), dtype=int)
    order = np.argsort(distances[:, 0])
    for i in order:
        for idx in indices[i]:
            if idx not in assigned:
                result[i] = idx
                assigned.add(idx)
                break
    return result


def get_ordered_unique_indices_by_stage(
    indices,
    adata,
    stage_key: str = "stage",
    stage_order: Optional[Sequence[str]] = None,
) -> np.ndarray:
    idx_seq = np.asarray(indices).flatten()
    n = len(idx_seq)
    if stage_order is None:
        stage_order = sorted(adata.obs[stage_key].unique())
    stage2num = {s: i for i, s in enumerate(stage_order)}
    pos_stage = [stage2num.get(adata.obs.iloc[cell][stage_key], -1) for cell in idx_seq]
    cell_pos = defaultdict(list)
    for idx, cell in enumerate(idx_seq):
        cell_pos[cell].append(idx)
    final_seq = []
    seen = set()
    for curr_idx, cell in enumerate(idx_seq):
        if cell in seen:
            continue
        seen.add(cell)
        cand_pos = cell_pos[cell]
        curr_neigh_stage = pos_stage[curr_idx]
        best_p = curr_idx
        min_diff = float("inf")
        for p in cand_pos:
            left_stg = pos_stage[p - 1] if p > 0 else -1
            right_stg = pos_stage[p + 1] if p < n - 1 else 999
            fit_score = abs(left_stg - curr_neigh_stage) + abs(right_stg - curr_neigh_stage)
            if fit_score < min_diff:
                min_diff = fit_score
                best_p = p
        final_seq.append(idx_seq[best_p])
    return np.array(final_seq)


def resolve_umap_coords(analyzer_or_coords, adata=None) -> np.ndarray:
    if isinstance(analyzer_or_coords, np.ndarray):
        return analyzer_or_coords
    return analyzer_or_coords.cell_positions_2d


from lap_label_config import (
    endpoint_styles_for_figure,
    load_label_overrides,
    resolve_text_position,
)

UMAP_CELL_SIZE = 5
PATH_LANDMARK_SIZE = UMAP_CELL_SIZE * 2
PATH_LANDMARK_EDGE_LW = 1.0
PANEL_TRANSITION_EDGE_LW = 2.0
BOOTSTRAP_PATH_COLOR = "red"
BOOTSTRAP_ENVELOPE_ALPHA = 0.2
LAP_UMAP_FIGSIZE = (3.5, 3.0)
LAP_UMAP_PATH_LINEWIDTH = 1.5


def _stage_palette_color(palette: Optional[dict], stage: str, fallback: str = "#999999") -> str:
    if not palette:
        return fallback
    return palette.get(str(stage), fallback)


def _stage_at_umap_point(
    umap_coords: np.ndarray,
    adata,
    point: Sequence[float],
    stage_key: str = "stage",
) -> str:
    coords = np.asarray(umap_coords, dtype=float)
    pt = np.asarray(point, dtype=float).reshape(1, -1)
    nbrs = NearestNeighbors(n_neighbors=1).fit(coords)
    idx = int(nbrs.kneighbors(pt)[1][0, 0])
    return str(adata.obs[stage_key].iloc[idx])


def _draw_path_landmark(
    ax,
    xy: Sequence[float],
    facecolor: str,
    edgecolor: str,
    *,
    size: float = PATH_LANDMARK_SIZE,
    linewidth: float = PATH_LANDMARK_EDGE_LW,
):
    ax.scatter(
        float(xy[0]),
        float(xy[1]),
        s=size,
        c=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=7,
        alpha=0.95,
    )


def _scatter_umap_background(ax, umap_coords: np.ndarray, adata, palette: Optional[dict], stage_key: str = "stage"):
    stages = adata.obs[stage_key].values
    for stage in np.unique(stages):
        mask = stages == stage
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            s=UMAP_CELL_SIZE,
            alpha=0.4,
            c=_stage_palette_color(palette, str(stage)) if palette else None,
        )


def _mark_path_endpoints(
    ax,
    path: np.ndarray,
    start_state: str,
    end_state: str,
    palette: Optional[dict],
):
    path = np.asarray(path, dtype=float)
    _draw_path_landmark(ax, path[0], _stage_palette_color(palette, start_state), "black")
    _draw_path_landmark(ax, path[-1], _stage_palette_color(palette, end_state), "black")


def _mark_path_transition(
    ax,
    path: np.ndarray,
    path_result: dict,
    umap_coords: np.ndarray,
    adata,
    palette: Optional[dict],
    stage_key: str = "stage",
):
    if "transition_state_idx" not in path_result:
        return
    path = np.asarray(path, dtype=float)
    ts_idx = clip_path_index(int(path_result["transition_state_idx"]), len(path))
    ts_pt = path[ts_idx]
    ts_stage = _stage_at_umap_point(umap_coords, adata, ts_pt, stage_key=stage_key)
    _draw_path_landmark(ax, ts_pt, _stage_palette_color(palette, ts_stage), "red")


def _mark_path_landmarks(
    ax,
    path: np.ndarray,
    path_result: Optional[dict],
    start_state: str,
    end_state: str,
    umap_coords: np.ndarray,
    adata,
    palette: Optional[dict],
    *,
    stage_key: str = "stage",
    annotate_endpoints: bool = False,
    mark_transition: bool = True,
    figure_key: Optional[str] = None,
    label_overrides: Optional[dict] = None,
):
    _mark_path_endpoints(ax, path, start_state, end_state, palette)
    if mark_transition and path_result is not None:
        _mark_path_transition(ax, path, path_result, umap_coords, adata, palette, stage_key=stage_key)
    if annotate_endpoints:
        _annotate_path_endpoint_labels(
            ax,
            path,
            start_state,
            end_state,
            figure_key=figure_key,
            label_overrides=label_overrides,
        )


def _annotate_path_endpoint_labels(
    ax,
    path: np.ndarray,
    start_label: str,
    end_label: str,
    *,
    figure_key: Optional[str] = None,
    label_overrides: Optional[dict] = None,
    fontsize: float = 7,
    color: str = "k",
    zorder: int = 11,
):
    """Place start/end stage text near path endpoints (black text)."""
    path = np.asarray(path, dtype=float)
    overrides = label_overrides or {}
    if figure_key and figure_key in overrides:
        start_style, end_style = endpoint_styles_for_figure(
            overrides, figure_key, path, start_label, end_label
        )
    else:
        start_style = {
            "label": start_label,
            "dx": 0.0,
            "dy": 0.0,
            "ha": "center",
            "va": "bottom",
            "fontsize": fontsize,
            "color": color,
        }
        end_style = {
            "label": end_label,
            "dx": 0.0,
            "dy": 0.0,
            "ha": "center",
            "va": "top",
            "fontsize": fontsize,
            "color": color,
        }

    for pt, style in ((path[0], start_style), (path[-1], end_style)):
        x, y = resolve_text_position(pt, style)
        ax.text(
            x,
            y,
            str(style.get("label", "")),
            fontsize=float(style.get("fontsize", fontsize)),
            ha=str(style.get("ha", "center")),
            va=str(style.get("va", "center")),
            color=str(style.get("color", color)),
            zorder=zorder,
        )


def _annotate_path_endpoints(
    ax,
    path: np.ndarray,
    start_label: str,
    end_label: str,
    *,
    palette: Optional[dict] = None,
    figure_key: Optional[str] = None,
    label_overrides: Optional[dict] = None,
    fontsize: float = 7,
    color: str = "k",
    zorder: int = 11,
    mark_points: bool = True,
    show_labels: bool = True,
):
    """Backward-compatible wrapper: endpoint markers + optional labels."""
    path = np.asarray(path, dtype=float)
    if mark_points:
        _mark_path_endpoints(ax, path, start_label, end_label, palette)
    if show_labels:
        _annotate_path_endpoint_labels(
            ax,
            path,
            start_label,
            end_label,
            figure_key=figure_key,
            label_overrides=label_overrides,
            fontsize=fontsize,
            color=color,
            zorder=zorder,
        )


def plot_two_paths_on_umap(
    analyzer_or_coords,
    path1_result,
    path2_result,
    states_to_plot,
    adata,
    palette=None,
    ax=None,
    title: Optional[str] = None,
    *,
    stage_key: str = "stage",
    figure_key: Optional[str] = None,
    label_overrides: Optional[dict] = None,
):
    umap_2d = resolve_umap_coords(analyzer_or_coords, adata)
    if ax is None:
        _, ax = plt.subplots(figsize=LAP_UMAP_FIGSIZE)
    _scatter_umap_background(ax, umap_2d, adata, palette, stage_key=stage_key)
    path1 = smooth_path(path1_result["path"].copy(), smoothing_factor=0.1)
    path2 = smooth_path(path2_result["path"].copy(), smoothing_factor=0.1)
    ax.plot(path1[:, 0], path1[:, 1], color="k", linestyle="-", linewidth=LAP_UMAP_PATH_LINEWIDTH, zorder=5)
    ax.plot(path2[:, 0], path2[:, 1], color="k", linestyle="--", linewidth=LAP_UMAP_PATH_LINEWIDTH, zorder=5)
    if states_to_plot:
        for path, path_result in ((path1, path1_result), (path2, path2_result)):
            _mark_path_endpoints(ax, path, states_to_plot[0], states_to_plot[1], palette)
            _mark_path_transition(ax, path, path_result, umap_2d, adata, palette, stage_key=stage_key)
    ax.set_xlabel("UMAP 1", fontsize=7)
    ax.set_ylabel("UMAP 2", fontsize=7)
    ax.set_title(title or "Least-Action Path on Cell UMAP", fontsize=9, weight="bold")
    ax.tick_params(axis="both", labelsize=7)
    return ax


def plot_path_on_umap(
    umap_coords: np.ndarray,
    path_result: dict,
    adata,
    palette=None,
    ax=None,
    *,
    line_style: str = "-",
    line_color: str = "k",
    line_width: float = LAP_UMAP_PATH_LINEWIDTH,
    mark_transition: bool = True,
    states_to_plot: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    smooth: bool = True,
    stage_key: str = "stage",
    figure_key: Optional[str] = None,
    label_overrides: Optional[dict] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=LAP_UMAP_FIGSIZE)
    umap_coords = np.asarray(umap_coords, dtype=float)
    _scatter_umap_background(ax, umap_coords, adata, palette, stage_key=stage_key)
    path = path_result["path"].copy()
    if smooth:
        path = smooth_path(path, smoothing_factor=0.1)
    ax.plot(path[:, 0], path[:, 1], color=line_color, linestyle=line_style, linewidth=line_width, zorder=5)
    if states_to_plot:
        _mark_path_landmarks(
            ax,
            path,
            path_result if mark_transition else None,
            states_to_plot[0],
            states_to_plot[1],
            umap_coords,
            adata,
            palette,
            stage_key=stage_key,
            annotate_endpoints=True,
            mark_transition=mark_transition,
            figure_key=figure_key,
            label_overrides=label_overrides,
        )
    ax.set_xlabel("UMAP 1", fontsize=7)
    ax.set_ylabel("UMAP 2", fontsize=7)
    if title:
        ax.set_title(title, fontsize=9, weight="bold")
    ax.tick_params(axis="both", labelsize=7)
    return ax


def overlay_canonical_medoid_path_on_umap(
    ax,
    umap_coords: np.ndarray,
    path_result: Optional[dict],
    adata,
    palette=None,
    *,
    line_color: str = "red",
    stage_key: str = "stage",
    mark_transition: bool = True,
) -> None:
    """Overlay smoothed canonical medoid path and landmarks (same style as umap_canonical_medoid)."""
    if path_result is None or "path" not in path_result:
        return
    umap_coords = np.asarray(umap_coords, dtype=float)
    path = smooth_path(np.asarray(path_result["path"], dtype=float).copy(), smoothing_factor=0.1)
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=line_color,
        linestyle="-",
        linewidth=LAP_UMAP_PATH_LINEWIDTH,
        zorder=5,
    )
    start_state = path_result.get("start_state")
    end_state = path_result.get("end_state")
    if start_state and end_state:
        _mark_path_landmarks(
            ax,
            path,
            path_result if mark_transition else None,
            str(start_state),
            str(end_state),
            umap_coords,
            adata,
            palette,
            stage_key=stage_key,
            annotate_endpoints=True,
            mark_transition=mark_transition,
        )


def plot_all_lap_paths_on_umap(
    umap_coords: np.ndarray,
    paths: Dict[str, dict],
    adata,
    palette=None,
    ax=None,
    *,
    states_to_plot: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    stage_key: str = "stage",
    figure_key: Optional[str] = None,
    label_overrides: Optional[dict] = None,
):
    """Overlay canonical / min / max paths on one UMAP."""
    styles = {
        "canonical_medoid": {"color": "red", "ls": "-", "lw": LAP_UMAP_PATH_LINEWIDTH},
        "canonical_pseudotime": {"color": "red", "ls": "-.", "lw": LAP_UMAP_PATH_LINEWIDTH},
        "min": {"color": "k", "ls": "-", "lw": LAP_UMAP_PATH_LINEWIDTH},
        "max": {"color": "k", "ls": "--", "lw": LAP_UMAP_PATH_LINEWIDTH},
    }
    if ax is None:
        _, ax = plt.subplots(figsize=LAP_UMAP_FIGSIZE)
    umap_coords = np.asarray(umap_coords, dtype=float)
    _scatter_umap_background(ax, umap_coords, adata, palette, stage_key=stage_key)

    labeled = set()
    draw_order = ("min", "max", "canonical_medoid", "canonical_pseudotime")
    smoothed_paths: Dict[str, np.ndarray] = {}
    for key in draw_order:
        if key not in paths:
            continue
        style = styles[key]
        path = smooth_path(paths[key]["path"].copy(), smoothing_factor=0.1)
        smoothed_paths[key] = path
        label = key.replace("_", " ")
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            zorder=5 if key.startswith("canonical") else 4,
            label=label if label not in labeled else None,
        )
        labeled.add(label)

    if states_to_plot:
        for key in ("min", "max", "canonical_medoid"):
            if key not in paths or key not in smoothed_paths:
                continue
            path = smoothed_paths[key]
            _mark_path_endpoints(ax, path, states_to_plot[0], states_to_plot[1], palette)
            _mark_path_transition(
                ax, path, paths[key], umap_coords, adata, palette, stage_key=stage_key
            )

    ax.set_xlabel("UMAP 1", fontsize=7)
    ax.set_ylabel("UMAP 2", fontsize=7)
    ax.set_title(title or "LAP paths on UMAP", fontsize=9, weight="bold")
    ax.tick_params(axis="both", labelsize=7)
    return ax


def plot_bootstrap_path_envelope(
    ax,
    median_path: np.ndarray,
    lower_path: np.ndarray,
    upper_path: np.ndarray,
    *,
    path_color: str = BOOTSTRAP_PATH_COLOR,
    envelope_alpha: float = BOOTSTRAP_ENVELOPE_ALPHA,
    linewidth: float = LAP_UMAP_PATH_LINEWIDTH,
    label_median: bool = True,
    label_envelope: bool = True,
):
    """Draw bootstrap median path with 10–90% envelope band."""
    median_path = np.asarray(median_path, dtype=float)
    lower_path = np.asarray(lower_path, dtype=float)
    upper_path = np.asarray(upper_path, dtype=float)
    ax.plot(
        median_path[:, 0],
        median_path[:, 1],
        color=path_color,
        linewidth=linewidth,
        label="median" if label_median else None,
        zorder=5,
    )
    ax.fill_between(
        median_path[:, 0],
        lower_path[:, 1],
        upper_path[:, 1],
        color=path_color,
        alpha=envelope_alpha,
        label="10–90%" if label_envelope else None,
        zorder=4,
        linewidth=0,
    )


def plot_bootstrap_path_on_umap(
    umap_coords: np.ndarray,
    median_path: np.ndarray,
    lower_path: np.ndarray,
    upper_path: np.ndarray,
    adata,
    palette=None,
    ax=None,
    *,
    central_path_result: Optional[dict] = None,
    states_to_plot: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    stage_key: str = "stage",
    figure_key: Optional[str] = None,
    label_overrides: Optional[dict] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=LAP_UMAP_FIGSIZE)
    umap_coords = np.asarray(umap_coords, dtype=float)
    _scatter_umap_background(ax, umap_coords, adata, palette, stage_key=stage_key)

    median = smooth_path(median_path, smoothing_factor=0.05)
    lower = smooth_path(lower_path, smoothing_factor=0.05)
    upper = smooth_path(upper_path, smoothing_factor=0.05)
    plot_bootstrap_path_envelope(ax, median, lower, upper, label_median=False, label_envelope=False)

    if states_to_plot:
        _mark_path_landmarks(
            ax,
            median,
            central_path_result,
            states_to_plot[0],
            states_to_plot[1],
            umap_coords,
            adata,
            palette,
            stage_key=stage_key,
            annotate_endpoints=True,
            mark_transition=central_path_result is not None,
            figure_key=figure_key,
            label_overrides=label_overrides,
        )

    ax.set_xlabel("UMAP 1", fontsize=7)
    ax.set_ylabel("UMAP 2", fontsize=7)
    ax.set_title(title or "Bootstrap canonical path (10–90% band)", fontsize=9, weight="bold")
    ax.tick_params(axis="both", labelsize=7)
    return ax


def plot_one_path_on_umap(
    analyzer,
    path1_result,
    states_to_plot,
    adata,
    palette=None,
    ax=None,
    *,
    stage_key: str = "stage",
    figure_key: Optional[str] = None,
    label_overrides: Optional[dict] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=LAP_UMAP_FIGSIZE)
    umap_2d = np.asarray(analyzer.cell_positions_2d, dtype=float)
    _scatter_umap_background(ax, umap_2d, adata, palette, stage_key=stage_key)
    path1 = smooth_path(path1_result["path"].copy(), smoothing_factor=0.1)
    ax.plot(path1[:, 0], path1[:, 1], color="k", linestyle="-", linewidth=LAP_UMAP_PATH_LINEWIDTH, zorder=5)
    if states_to_plot:
        _mark_path_landmarks(
            ax,
            path1,
            path1_result,
            states_to_plot[0],
            states_to_plot[1],
            umap_2d,
            adata,
            palette,
            stage_key=stage_key,
            annotate_endpoints=True,
            mark_transition=True,
            figure_key=figure_key,
            label_overrides=label_overrides,
        )
    ax.set_xlabel("UMAP 1", fontsize=7)
    ax.set_ylabel("UMAP 2", fontsize=7)
    ax.set_title("Least-Action Path on Cell UMAP", fontsize=9, weight="bold")
    return ax


def plot_pioneer_gene_heatmap(
    adata,
    indices,
    pioneer_gene,
    transition_state_indice_cell,
    save_path: str,
    figsize=(8, 5),
    cmap=None,
    gene_fontsize=8,
    title_fontsize=10,
    dpi=300,
):
    indices_cell = adata[indices, pioneer_gene].copy()
    indices_cell.obs["original_cell_id"] = indices_cell.obs.index
    indices_cell_sorted = indices_cell.obs.sort_values(by=["stage", "pseudotime"], ascending=[True, True])
    sorted_global_indices = indices_cell_sorted["original_cell_id"].values
    expr_matrix = indices_cell[sorted_global_indices, :].X
    expr_matrix_T = expr_matrix.T
    expr_df = pd.DataFrame(
        expr_matrix_T.toarray() if hasattr(expr_matrix_T, "toarray") else expr_matrix_T,
        index=pioneer_gene,
        columns=sorted_global_indices,
    )
    expr_df_normalized = expr_df.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0,
        axis=1,
    )
    if cmap is None:
        cmap = TECH_BLUE_CMAP
    g = sns.clustermap(
        expr_df_normalized,
        cmap=cmap,
        row_cluster=True,
        col_cluster=False,
        figsize=figsize,
        cbar_kws={"label": "Normalized Gene Expression", "orientation": "horizontal", "shrink": 0.8, "pad": 0.1},
        dendrogram_ratio=(0.1, 0.1),
        yticklabels=True,
        xticklabels=False,
        cbar_pos=(0.2, 0.02, 0.6, 0.01),
    )
    cell_id_to_col_idx = {cell_id: idx for idx, cell_id in enumerate(expr_df.columns)}
    valid_key_cells = [cid for cid in transition_state_indice_cell if cid in cell_id_to_col_idx]
    for cid in valid_key_cells:
        pos = cell_id_to_col_idx[cid]
        g.ax_heatmap.axvline(x=pos + 0.5, color="red", linestyle="-", linewidth=1, zorder=5)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=gene_fontsize, rotation=0)
    g.fig.suptitle(
        "Pioneer Gene Expression in Transition Path Cells (Sorted by Stage → Pseudotime)",
        fontsize=title_fontsize,
        y=1.03,
    )
    plt.subplots_adjust(top=0.96, bottom=0.08, left=0.09, right=0.87, hspace=0.23)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    return g, expr_df_normalized


def path_cell_indices_for_panels(
    analyzer,
    path_result,
    *,
    mode: str = "legacy",
    interpolate_n: int = 100,
    unique_k: int = 5,
) -> np.ndarray:
    """Map LAP path points to ordered unique cell indices for panel figures."""
    cell_positions = np.asarray(analyzer.cell_positions_2d, dtype=float)
    path_compute = np.asarray(path_result.get("path_compute", path_result["path"]), dtype=float)
    if mode == "enhanced":
        path_interp = interpolate_path_arclength(path_compute, interpolate_n)
        raw = find_unique_neighbors(path_interp, cell_positions, k=unique_k)
        return unique_ordered(raw)
    nbrs = NearestNeighbors(n_neighbors=1).fit(cell_positions)
    _, indices = nbrs.kneighbors(path_compute)
    return unique_ordered(indices.flatten())


def _canonical_display_path(path_result: dict, *, smooth: bool = True) -> np.ndarray:
    """UMAP path curve used by umap_canonical_medoid figures."""
    path = np.asarray(path_result["path"], dtype=float)
    if smooth:
        path = smooth_path(path, smoothing_factor=0.1)
    return path


def _plot_path_cell_panel_axes(
    ax,
    adata,
    path_cells,
    transition_idx: int,
    color_key: str,
    title: str,
    palette: dict,
    *,
    show_background: bool = False,
    background_size: float = 3.0,
    path_cell_size: float = 80.0,
    path_display: Optional[np.ndarray] = None,
    transition_display: Optional[np.ndarray] = None,
):
    bg_umap = np.asarray(adata.obsm["X_umap"], dtype=float)
    path_umap = np.asarray(path_cells.obsm["X_umap"], dtype=float)
    if transition_display is not None:
        transition_coord = np.asarray(transition_display, dtype=float)
    else:
        transition_coord = path_umap[transition_idx]

    if show_background:
        ax.scatter(
            bg_umap[:, 0],
            bg_umap[:, 1],
            c="#d8d8d8",
            s=background_size,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
            zorder=0,
        )

    if color_key == "stage":
        for stage, color in palette.items():
            mask = path_cells.obs[color_key].astype(str).values == str(stage)
            if not np.any(mask):
                continue
            ax.scatter(
                path_umap[mask, 0],
                path_umap[mask, 1],
                c=color,
                s=path_cell_size,
                alpha=0.92,
                edgecolors="black",
                linewidths=0.35,
                zorder=2,
            )
    else:
        vals = np.asarray(path_cells.obs[color_key].values, dtype=float)
        sc_obj = ax.scatter(
            path_umap[:, 0],
            path_umap[:, 1],
            c=vals,
            cmap="coolwarm",
            s=path_cell_size,
            alpha=0.92,
            edgecolors="black",
            linewidths=0.35,
            zorder=2,
        )
        plt.colorbar(sc_obj, ax=ax, fraction=0.046, pad=0.04)

    if path_display is not None:
        path_line = np.asarray(path_display, dtype=float)
    else:
        path_line = path_umap
    ax.plot(path_line[:, 0], path_line[:, 1], color="black", linewidth=1.2, linestyle="-", zorder=1)
    ax.scatter(
        transition_coord[0],
        transition_coord[1],
        facecolor="none",
        edgecolors="red",
        linewidth=1.5,
        s=120,
        zorder=10,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.set_aspect("equal", adjustable="box")


def plot_path_cell_panels(adata, path_cells, transition_idx, palette, save_path: str):
    umap_coords = path_cells.obsm["X_umap"]
    transition_coord = umap_coords[transition_idx]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, left=0.05, right=0.95, top=0.88, bottom=0.1)
    for ax, color_key, title, kwargs in [
        (axs[0], "stage", "Stage", {"palette": palette}),
        (axs[1], "pseudotime", "Pseudotime", {"color_map": "coolwarm"}),
        (axs[2], "potential", "Potential", {"color_map": "coolwarm"}),
    ]:
        sc.pl.umap(path_cells, color=color_key, title=title, ax=ax, size=200, alpha=0.7, show=False, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.plot(umap_coords[:, 0], umap_coords[:, 1], color="black", linewidth=1, linestyle="-", zorder=0)
        ax.scatter(
            transition_coord[0],
            transition_coord[1],
            facecolor="none",
            edgecolors="red",
            linewidth=1.5,
            s=100,
            zorder=999,
        )
    plt.savefig(save_path)
    plt.close("all")


def plot_path_cell_panels_enhanced(
    adata,
    path_cells,
    transition_idx,
    palette,
    save_path: str,
    *,
    background_size: float = 3.0,
    path_cell_size: float = 80.0,
):
    """Gray background + path anchor cells connected by their UMAP polyline."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=False)
    plt.subplots_adjust(wspace=0.35, hspace=0.2, left=0.05, right=0.97, top=0.88, bottom=0.12)
    for ax, color_key, title in [
        (axs[0], "stage", "Stage"),
        (axs[1], "pseudotime", "Pseudotime"),
        (axs[2], "potential", "Potential"),
    ]:
        _plot_path_cell_panel_axes(
            ax,
            adata,
            path_cells,
            transition_idx,
            color_key,
            title,
            palette,
            show_background=True,
            background_size=background_size,
            path_cell_size=path_cell_size,
        )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_path_cell_panels_enhanced_canonical(
    adata,
    path_cells,
    transition_idx,
    palette,
    path_result: dict,
    save_path: str,
    *,
    background_size: float = 3.0,
    path_cell_size: float = 80.0,
):
    """Gray background + canonical smoothed UMAP path (same as umap_canonical_medoid) + anchor cells."""
    path_display = _canonical_display_path(path_result)
    ts_idx = clip_path_index(int(path_result["transition_state_idx"]), len(path_display))
    transition_display = path_display[ts_idx]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=False)
    plt.subplots_adjust(wspace=0.35, hspace=0.2, left=0.05, right=0.97, top=0.88, bottom=0.12)
    for ax, color_key, title in [
        (axs[0], "stage", "Stage"),
        (axs[1], "pseudotime", "Pseudotime"),
        (axs[2], "potential", "Potential"),
    ]:
        _plot_path_cell_panel_axes(
            ax,
            adata,
            path_cells,
            transition_idx,
            color_key,
            title,
            palette,
            show_background=True,
            background_size=background_size,
            path_cell_size=path_cell_size,
            path_display=path_display,
            transition_display=transition_display,
        )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_de_transition_heatmap(
    adata_de,
    top_genes: Sequence[str],
    ts_pseudotime: float,
    save_path: str,
    vmax: float = 2.5,
):
    sorted_adata = adata_de[adata_de.obs["pseudotime"].argsort()].copy()
    exp_mat = sorted_adata[:, top_genes].X
    if hasattr(exp_mat, "toarray"):
        exp_mat = exp_mat.toarray()
    sorted_pt = sorted_adata.obs["pseudotime"].values
    n_cells, n_genes = exp_mat.shape
    if n_cells < 1 or n_genes < 1:
        warnings.warn("Skipping DE heatmap: empty expression matrix.", UserWarning)
        return None
    row_cluster = n_genes >= 2
    try:
        g = sns.clustermap(
            exp_mat.T,
            cmap=TECH_BLUE_CMAP,
            vmin=0,
            vmax=vmax,
            xticklabels=False,
            yticklabels=list(top_genes),
            cbar_kws={"orientation": "horizontal", "shrink": 0.8, "pad": 0.1},
            row_cluster=row_cluster,
            col_cluster=False,
            metric="euclidean",
            figsize=(6, 3.5),
            dendrogram_ratio=(0.06, 0.02) if row_cluster else (0.0, 0.02),
            cbar_pos=(0.2, 0.02, 1, 0.03),
        )
        g.ax_cbar.set_title("Gene Expression", fontsize=6)
        ts_pos = np.searchsorted(sorted_pt, ts_pseudotime) - 1
        g.ax_heatmap.axvline(x=ts_pos + 0.5, color="red", linewidth=1, zorder=10)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8, rotation=0)
        g.fig.suptitle("Before vs After Transition State", fontsize=10, y=0.97, x=0.56)
        plt.subplots_adjust(top=0.93, bottom=0.23, left=0.27, right=0.73, hspace=0.22)
        plt.savefig(save_path)
        plt.close("all")
        return g
    except ValueError as exc:
        warnings.warn(f"Skipping DE heatmap clustermap: {exc}", UserWarning)
        fig, ax = plt.subplots(figsize=(6, max(1.5, 0.35 * n_genes)))
        im = ax.imshow(exp_mat.T, aspect="auto", cmap=TECH_BLUE_CMAP, vmin=0, vmax=vmax)
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(list(top_genes), fontsize=8)
        ax.set_xticks([])
        ts_pos = np.searchsorted(sorted_pt, ts_pseudotime) - 1
        ax.axvline(x=ts_pos + 0.5, color="red", linewidth=1, zorder=10)
        fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.12)
        fig.suptitle("Before vs After Transition State", fontsize=10)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return None


def path_nearest_cell_indices(analyzer, path_result, start_state: str, end_state: str, clustering_key: str = "stage"):
    cell_states = analyzer.identify_cell_states(clustering_key=clustering_key, max_pot=False, use_3d=False)
    start_pos = cell_states[start_state]["position"]
    end_pos = cell_states[end_state]["position"]
    cell_positions = analyzer.cell_positions_2d
    nbrs = NearestNeighbors(n_neighbors=1).fit(cell_positions)
    path = path_result.get("path_compute", path_result["path"])
    _, indices = nbrs.kneighbors(path)
    return indices, start_pos, end_pos, cell_states
