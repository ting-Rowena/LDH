"""Headless plotting helpers for server runs (save figures, no display)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

TECH_BLUE_COLORS = ["#3A6798", "#E8D5CE", "#B86A78"]


def tech_blue_cmap() -> LinearSegmentedColormap:
    """Custom colormap for validation support scores and heatmaps."""
    return LinearSegmentedColormap.from_list("tech_blue", TECH_BLUE_COLORS)


TECH_BLUE_CMAP = tech_blue_cmap()

# Validation pseudotime distribution figures (bootstrap TS hist + region violins)
PSEUDOTIME_DISTRIBUTION_COLOR = "#FFE9DD"
PSEUDOTIME_DISTRIBUTION_ALPHA = 0.9


def configure_headless(show_figures: bool = False):
    """Use non-interactive backend; optionally allow plt.show() for local debugging."""
    if not show_figures and os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg", force=True)
    apply_publication_style()
    return show_figures


# ---------------------------------------------------------------------------
# Cohesive publication palette (shared across all protocol figures)
# ---------------------------------------------------------------------------
INK = "#1f2933"            # near-black text / spines
MUTED = "#7b8794"          # secondary text, ticks
GRID = "#e3e8ee"           # subtle gridlines
PANEL_BG = "#ffffff"       # clean white panels

# Qualitative palette (color-blind friendly, muted-modern)
PALETTE = [
    "#3a6ea5",  # blue
    "#e07a5f",  # coral
    "#3d9970",  # green
    "#c9a227",  # gold
    "#8367c7",  # violet
    "#d1495b",  # rose
    "#2a9d8f",  # teal
    "#6d597a",  # plum
]

# Named accents for recurring semantics
ACCENT_HI = "#d1495b"      # highlight / peak / max
ACCENT_UP = "#c1121f"      # up-regulated / stress
ACCENT_DN = "#3a6ea5"      # down-regulated / stable


def apply_publication_style():
    """Shared matplotlib style for training / analysis figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 11.5,
            "axes.labelcolor": INK,
            "axes.labelpad": 5,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.9,
            "text.color": INK,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.borderaxespad": 0.4,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": PANEL_BG,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )
    try:
        import cycler

        plt.rcParams["axes.prop_cycle"] = cycler.cycler(color=PALETTE)
    except Exception:
        pass


def style_axis(ax, *, grid_axis: str = "y"):
    """Apply the shared clean look to a single axis (spines, grid, ticks)."""
    ax.set_facecolor(PANEL_BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK, length=3.5, width=0.9)
    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=1.0)
        ax.set_axisbelow(True)
    return ax


def polished_colorbar(mappable, ax, label: str = "", **kwargs):
    """Slim, well-labelled colorbar with muted outline."""
    kwargs.setdefault("fraction", 0.046)
    kwargs.setdefault("pad", 0.03)
    cbar = plt.colorbar(mappable, ax=ax, **kwargs)
    cbar.outline.set_edgecolor(MUTED)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=9, colors=INK, length=2.5, width=0.7)
    if label:
        cbar.set_label(label, fontsize=10, color=INK)
    return cbar


def gradient_barh(ax, labels, values, *, cmap="viridis", reverse_cmap=False,
                  highlight=None, highlight_color=ACCENT_HI):
    """Horizontal bar chart with a value-mapped color gradient (top value on top)."""
    import numpy as _np

    labels = list(labels)
    values = _np.asarray(values, dtype=float)
    y = _np.arange(len(labels))
    vmin, vmax = float(_np.nanmin(values)), float(_np.nanmax(values))
    span = (vmax - vmin) or 1.0
    cm = plt.get_cmap(cmap)
    norm = (values - vmin) / span
    if reverse_cmap:
        norm = 1.0 - norm
    colors = [cm(0.15 + 0.75 * n) for n in norm]
    if highlight is not None:
        for i, lab in enumerate(labels):
            if lab in highlight:
                colors[i] = highlight_color
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    style_axis(ax, grid_axis="x")
    return ax


def ensure_figure_dir(save_dir: str, subdir: str = "figures") -> str:
    root = os.path.abspath(save_dir)
    if os.path.basename(root) == subdir:
        os.makedirs(root, exist_ok=True)
        return root
    fig_dir = os.path.join(root, subdir)
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def figure_path(save_dir: str, filename: str, subdir: str = "figures") -> str:
    if not filename.lower().endswith((".png", ".pdf", ".svg")):
        filename = f"{filename}.png"
    return os.path.join(ensure_figure_dir(save_dir, subdir), filename)


def _adaptive_dpi(n_obs: int, default: int = 300) -> int:
    if n_obs > 40000:
        return 120
    if n_obs > 15000:
        return 150
    return default


def subsample_for_plot(
    adata,
    max_cells: int = 15000,
    label_key: Optional[str] = None,
    random_state: int = 112,
):
    """Return a smaller AnnData copy for visualization only."""
    if adata.n_obs <= max_cells:
        return adata
    rng = np.random.RandomState(random_state)
    if label_key and label_key in adata.obs:
        groups = adata.obs[label_key]
        idx = []
        frac = max_cells / adata.n_obs
        for cat in groups.unique():
            inds = np.where(groups == cat)[0]
            n_take = max(1, int(len(inds) * frac))
            idx.extend(rng.choice(inds, size=min(n_take, len(inds)), replace=False))
        if len(idx) > max_cells:
            idx = rng.choice(idx, max_cells, replace=False)
        return adata[idx].copy()
    import scanpy as sc

    return sc.pp.subsample(adata, n_obs=max_cells, random_state=random_state, copy=True)


def rasterize_figure(fig):
    """Rasterize scatter collections so large UMAP panels stay small on disk."""
    for ax in fig.get_axes():
        for col in ax.collections:
            col.set_rasterized(True)


def save_figure(
    fig,
    save_dir: str,
    filename: str,
    *,
    subdir: str = "figures",
    dpi: Optional[int] = None,
    bbox_inches: str = "tight",
    close: bool = True,
    rasterize: bool = True,
) -> str:
    path = figure_path(save_dir, filename, subdir=subdir)
    if rasterize:
        rasterize_figure(fig)
    fig.savefig(path, dpi=dpi or 150, bbox_inches=bbox_inches, facecolor="white")
    print(f"Saved figure: {path}")
    if close:
        plt.close(fig)
    return path


def save_current_figure(save_dir: str, filename: str, **kwargs) -> str:
    return save_figure(plt.gcf(), save_dir, filename, **kwargs)


def finish_figure(save_dir, filename, fig=None, show_figures=False, **kwargs):
    """Save to save_dir/figures by default; show only when explicitly enabled."""
    f = fig if fig is not None else plt.gcf()
    if save_dir and not show_figures:
        return save_figure(f, save_dir, filename, **kwargs)
    if show_figures:
        plt.show()
        return None
    plt.close(f)
    return None


def setup_scanpy_figdir(save_dir: str, subdir: str = "figures"):
    """Route scanpy `save=` outputs into save_dir/figures."""
    import scanpy as sc

    fig_dir = ensure_figure_dir(save_dir, subdir)
    sc.settings.figdir = fig_dir
    setter = getattr(sc.settings, "_set_figure_params", sc.settings.set_figure_params)
    setter(dpi=300, frameon=False, figsize=(5, 5), fontsize=11)
    return fig_dir


def resolve_violin_groupby_key(adata, config) -> str:
    """Categorical obs column for scanpy violin plots (time may be float for training)."""
    import pandas as pd

    style = getattr(config, "plot_style", None) if config is not None else None
    candidates = [
        getattr(config, "temporal_group_key", None) if config else None,
        style.stage_key if style else None,
        "stage",
        "condition",
        "treatment",
        getattr(config, "time_key", "time") if config else "time",
    ]
    for key in candidates:
        if not key or key not in adata.obs:
            continue
        series = adata.obs[key]
        if hasattr(series, "cat") or not pd.api.types.is_numeric_dtype(series):
            return key
    time_key = getattr(config, "time_key", "time") if config else "time"
    cat_key = f"{time_key}_violin"
    adata.obs[cat_key] = adata.obs[time_key].astype(str).astype("category")
    return cat_key


def resolve_label_key(adata, config=None, preferred: Optional[str] = None) -> Optional[str]:
    """Best human-readable cell-type column for legends / UMAP labels."""
    cell_type_key = getattr(config, "cell_type_key", None) if config is not None else None
    candidates = [
        preferred,
        "annotation",
        "cell_type",
        "celltype",
        "cell_subtype",
        cell_type_key,
    ]
    for key in candidates:
        if not key or key not in adata.obs:
            continue
        series = adata.obs[key]
        if hasattr(series, "cat"):
            if len(series.cat.categories) > 1:
                return key
        elif series.nunique() > 1:
            return key
    return None


def ensure_categorical_obs(adata, key: str):
    if key not in adata.obs:
        return adata
    if not hasattr(adata.obs[key], "cat"):
        adata.obs[key] = adata.obs[key].astype(str).astype("category")
    return adata


CONTINUOUS_OBS_KEYS = frozenset(
    {"pseudotime", "potential", "diffusion_eff", "hjb_residual"}
)


def _resolve_plot_style(config=None, plot_style=None):
    if plot_style is not None:
        return plot_style
    if config is not None:
        return getattr(config, "plot_style", None)
    return None


def _is_discrete_obs(adata, key: str) -> bool:
    if key in CONTINUOUS_OBS_KEYS:
        return False
    if key not in adata.obs:
        return False
    series = adata.obs[key]
    if hasattr(series, "cat"):
        return True
    import pandas as pd

    return not pd.api.types.is_numeric_dtype(series)


def _palette_for_obs_key(plot_style, key: str) -> Optional[Dict[str, str]]:
    if plot_style is None:
        return None
    if key == plot_style.celltype_key:
        return plot_style.celltype_palette
    if key == plot_style.stage_key:
        return plot_style.stage_palette
    return None


def _complete_palette(adata, key: str, palette: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Ensure every observed category has a color (avoids scanpy KeyError)."""
    if not isinstance(palette, dict) or key not in adata.obs:
        return palette
    categories = [str(c) for c in adata.obs[key].astype(str).unique()]
    missing = [c for c in categories if c not in palette]
    if not missing:
        return palette
    import matplotlib.pyplot as plt

    tab = plt.get_cmap("tab20").colors
    out = dict(palette)
    for i, cat in enumerate(missing):
        out[cat] = plt.matplotlib.colors.to_hex(tab[i % len(tab)])
    return out


def get_dataset_plot_style(dataset_key: str):
    from dataset_pipeline import PLOT_STYLES

    return PLOT_STYLES.get(dataset_key)


def dataset_stage_key(dataset_key: str, default: str = "stage") -> str:
    style = get_dataset_plot_style(dataset_key)
    return style.stage_key if style is not None else default


def dataset_stage_palette(dataset_key: str, profile=None) -> Dict[str, str]:
    style = get_dataset_plot_style(dataset_key)
    if style is not None:
        return dict(style.stage_palette)
    if profile is not None and getattr(profile, "stage_palette", None):
        return dict(profile.stage_palette)
    return {}


def dataset_celltype_palette(dataset_key: str) -> Dict[str, str]:
    style = get_dataset_plot_style(dataset_key)
    return dict(style.celltype_palette) if style is not None else {}


def ordered_stage_labels(stages, stage_order: Sequence[str]) -> List[str]:
    unique = [str(s) for s in stages]
    ordered = [s for s in stage_order if s in unique]
    rest = sorted(s for s in unique if s not in ordered)
    return ordered + rest


def colors_for_labels(labels: Sequence[str], palette: Dict[str, str], *, fallback: str = "#999999") -> List[str]:
    return [palette.get(str(label), fallback) for label in labels]


def stages_in_path_window(
    start_state,
    end_state,
    stage_order: Sequence[str],
    *,
    available_stages: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return ordered stage labels from start_state through end_state (inclusive)."""
    order = [str(s) for s in stage_order]
    try:
        i0 = order.index(str(start_state))
        i1 = order.index(str(end_state))
    except ValueError:
        if available_stages is None:
            return []
        return ordered_stage_labels(available_stages, stage_order)
    if i1 < i0:
        i0, i1 = i1, i0
    window = order[i0 : i1 + 1]
    if available_stages is None:
        return window
    avail = {str(s) for s in available_stages}
    return [s for s in window if s in avail]


def midpoint_stage_between(start_state, end_state, stage_order: Sequence[str]) -> Optional[str]:
    order = [str(s) for s in stage_order]
    try:
        i0 = order.index(str(start_state))
        i1 = order.index(str(end_state))
    except ValueError:
        return None
    if i1 <= i0:
        return None
    if i1 - i0 >= 2:
        return order[i0 + 1]
    return order[(i0 + i1) // 2]


def region_window_colors(
    start_state,
    end_state,
    stage_palette: Dict[str, str],
    stage_order: Sequence[str],
) -> Dict[str, str]:
    mid = midpoint_stage_between(start_state, end_state, stage_order)
    return {
        "start_window": stage_palette.get(str(start_state), "#cccccc"),
        "transition_window": stage_palette.get(str(mid), "#999999") if mid else "#999999",
        "end_window": stage_palette.get(str(end_state), "#666666"),
    }


def scatter_embedding_by_stage(
    ax,
    coords: np.ndarray,
    stage_labels,
    stage_palette: Dict[str, str],
    stage_order: Sequence[str],
    *,
    size: float = 2.0,
    alpha: float = 0.15,
) -> None:
    """Scatter embedding colored by stage/treatment using dataset stage_palette."""
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(stage_labels)
    for stage in ordered_stage_labels(np.unique(labels), stage_order):
        mask = labels.astype(str) == str(stage)
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            alpha=alpha,
            c=stage_palette.get(str(stage), "#999999"),
            label=str(stage),
        )


def _ordered_celltype_labels(labels, celltype_palette: Dict[str, str]) -> List[str]:
    unique = [str(x) for x in labels]
    ordered = [k for k in celltype_palette if k in unique]
    rest = sorted(s for s in unique if s not in ordered)
    return ordered + rest


def scatter_embedding_by_celltype(
    ax,
    coords: np.ndarray,
    celltype_labels,
    celltype_palette: Dict[str, str],
    *,
    size: float = 2.0,
    alpha: float = 0.55,
    fallback: str = "#999999",
) -> None:
    """Scatter embedding colored by cell type using dataset celltype_palette."""
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(celltype_labels).astype(str)
    for ct in _ordered_celltype_labels(np.unique(labels), celltype_palette):
        mask = labels == ct
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            alpha=alpha,
            c=celltype_palette.get(ct, fallback),
            label=ct,
            rasterized=True,
            linewidths=0,
        )


def _resolve_stage_obs_key(adata, style, stage_order: Sequence[str]) -> str:
    candidates = []
    if style is not None:
        candidates.append(style.stage_key)
    candidates.extend(["stage", "condition", "treatment"])
    order_set = {str(s) for s in stage_order}
    for key in candidates:
        if key not in adata.obs:
            continue
        avail = {str(v) for v in adata.obs[key].astype(str).unique()}
        if avail & order_set:
            return key
    if style is not None and style.stage_key in adata.obs:
        return style.stage_key
    if "stage" in adata.obs:
        return "stage"
    raise KeyError(f"No stage column found for stages {list(stage_order)}")


def plot_stage_umap_panels_by_celltype(
    adata,
    save_dir: str,
    dataset_key: str,
    stage_order: Sequence[str],
    *,
    umap_key: str = "X_umap",
    subdir: str = "figures",
    filename: Optional[str] = None,
    max_cells_per_panel: int = 15000,
) -> str:
    """
    One-row UMAP panels: one subplot per stage, cells colored by celltype_palette.

    Saves to ``{save_dir}/{subdir}/{dataset_key}_stage_umap_panels_by_celltype.png``.
    """
    from dataset_pipeline import compute_training_umap

    style = get_dataset_plot_style(dataset_key)
    if style is None:
        raise ValueError(f"No PlotStyle registered for dataset {dataset_key!r}")

    if umap_key not in adata.obsm:
        compute_training_umap(adata, plot_style=style)

    celltype_key = style.celltype_key
    if celltype_key not in adata.obs and "celltype" in adata.obs:
        adata.obs[celltype_key] = adata.obs["celltype"].astype("category")
    if celltype_key not in adata.obs:
        raise KeyError(f"Cell type column {celltype_key!r} missing in adata.obs")

    stage_key = _resolve_stage_obs_key(adata, style, stage_order)
    stages = [str(s) for s in stage_order if str(s) in set(adata.obs[stage_key].astype(str))]
    if not stages:
        raise ValueError(f"No stages from {stage_order} found in adata.obs[{stage_key!r}]")

    coords = np.asarray(adata.obsm[umap_key], dtype=float)
    n = len(stages)
    fig_w = max(3.2 * n, 8.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.6), squeeze=False)
    axes = axes[0]

    pt_size = 3.0 if adata.n_obs > 20000 else 5.0
    legend_handles = []
    legend_labels = []

    for ax, stage in zip(axes, stages):
        mask = adata.obs[stage_key].astype(str) == stage
        n_stage = int(mask.sum())
        if n_stage == 0:
            ax.set_title(f"{stage}\n(n=0)", fontsize=9)
            ax.axis("off")
            continue

        idx = np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask))
        if len(idx) > max_cells_per_panel:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, size=max_cells_per_panel, replace=False)

        stage_coords = coords[idx]
        stage_labels = adata.obs[celltype_key].iloc[idx]
        scatter_embedding_by_celltype(
            ax,
            stage_coords,
            stage_labels,
            style.celltype_palette,
            size=pt_size,
            alpha=0.6,
        )
        ax.set_title(f"{stage}\n(n={n_stage:,})", fontsize=9)
        ax.set_xlabel("UMAP1", fontsize=8)
        ax.set_ylabel("UMAP2", fontsize=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)

        if not legend_handles:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(legend_labels), 8),
            fontsize=7,
            frameon=False,
        )

    fig.suptitle(f"{dataset_key}: UMAP by stage (cell type colors)", fontsize=10, y=1.02)
    plt.tight_layout()

    out_name = filename or f"{dataset_key}_stage_umap_panels_by_celltype.png"
    return save_figure(fig, save_dir, out_name, subdir=subdir, dpi=_adaptive_dpi(adata.n_obs))

def _legend_loc_for_categories(n_categories: int, n_obs: int) -> str:
    return "right margin"


def _category_count(adata, key: str) -> int:
    series = adata.obs[key]
    if hasattr(series, "cat"):
        return len(series.cat.categories)
    return int(series.nunique())


def embedding_kwargs(
    *,
    size: float = 8.0,
    alpha: float = 0.85,
    legend_loc: Optional[str] = None,
    n_categories: Optional[int] = None,
    n_obs: Optional[int] = None,
    frameon: bool = False,
) -> dict:
    if legend_loc is None and n_categories is not None:
        legend_loc = _legend_loc_for_categories(n_categories, n_obs or 0)
    elif legend_loc is None:
        legend_loc = "right margin"
    return dict(
        size=size,
        alpha=alpha,
        legend_loc=legend_loc,
        legend_fontsize=8 if (n_categories or 0) > 12 else 9,
        legend_fontoutline=2,
        frameon=frameon,
    )


def plot_embedding_panel(
    adata,
    ax,
    color: str,
    *,
    basis: str = "umap",
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    palette: Optional[Union[str, Dict[str, str]]] = None,
    legend_loc: Optional[str] = None,
    size: float = 8.0,
    discrete: Optional[bool] = None,
    plot_style=None,
    config=None,
):
    """Styled scanpy embedding: discrete → palette + right legend; continuous → colorbar."""
    import scanpy as sc

    style = _resolve_plot_style(config, plot_style)
    is_discrete = _is_discrete_obs(adata, color) if discrete is None else discrete

    kw: dict[str, Any] = dict(size=size, alpha=0.85, frameon=False)

    if is_discrete:
        ensure_categorical_obs(adata, color)
        n_cat = _category_count(adata, color)
        pal = palette if isinstance(palette, dict) else _palette_for_obs_key(style, color)
        if pal is None and isinstance(palette, str):
            pal = palette
        kw["legend_loc"] = legend_loc or "right margin"
        kw["legend_fontsize"] = 8 if n_cat > 12 else 9
        kw["legend_fontoutline"] = 2
        if isinstance(pal, dict):
            kw["palette"] = _complete_palette(adata, color, pal)
        elif isinstance(pal, str):
            kw["palette"] = pal
        elif n_cat is not None and n_cat <= 20:
            kw["palette"] = "tab20"
    else:
        kw["color_map"] = cmap or "viridis"
        kw["colorbar_loc"] = "right"

    sc.pl.embedding(adata, basis=basis, color=color, ax=ax, show=False, **kw)
    if title:
        ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")


def plot_training_umap_overview(adata, save_dir: str, config=None, plot_style=None, subdir: str = "figures"):
    """2×2 UMAP overview: cell type, stage, pseudotime, potential."""
    from dataset_pipeline import compute_training_umap

    if "X_umap" not in adata.obsm:
        compute_training_umap(adata, plot_style=plot_style, config=config)

    style = _resolve_plot_style(config, plot_style)
    celltype_key = style.celltype_key if style else resolve_label_key(adata, config)
    stage_key = style.stage_key if style else getattr(config, "temporal_group_key", "stage")
    if stage_key not in adata.obs and style is None:
        stage_key = "stage"

    label_key = celltype_key or resolve_label_key(adata, config)
    plot_adata = subsample_for_plot(adata, max_cells=15000, label_key=label_key)

    pseudotime_cmap = style.pseudotime_cmap if style else "magma"
    potential_cmap = style.potential_cmap if style else "RdYlBu_r"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panels = []

    if celltype_key and celltype_key in plot_adata.obs:
        panels.append((celltype_key, None, "Cell type", True))
    if stage_key and stage_key in plot_adata.obs:
        panels.append((stage_key, None, "Stage", True))
    if "pseudotime" in plot_adata.obs:
        panels.append(("pseudotime", pseudotime_cmap, "Pseudotime", False))
    if "potential" in plot_adata.obs:
        panels.append(("potential", potential_cmap, "Potential U(z,t)", False))

    pt_size = 8 if plot_adata.n_obs > 10000 else 12
    for ax, (color, cmap, title, is_disc) in zip(axes.flat, panels):
        plot_embedding_panel(
            plot_adata,
            ax,
            color,
            title=title,
            cmap=cmap,
            size=pt_size,
            discrete=is_disc,
            plot_style=style,
            config=config,
        )

    for ax in axes.flat[len(panels) :]:
        ax.axis("off")

    n_show = plot_adata.n_obs
    if n_show < adata.n_obs:
        fig.suptitle(
            f"Training overview (UMAP) — {n_show:,} / {adata.n_obs:,} cells shown",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
    else:
        fig.suptitle("Training overview (UMAP)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    save_figure(
        fig,
        save_dir,
        "umap_training_overview.png",
        subdir=subdir,
        dpi=_adaptive_dpi(adata.n_obs),
    )


def plot_pseudotime_relationships(
    adata,
    save_dir: str,
    *,
    gene_name: Optional[str] = None,
    subdir: str = "figures",
):
    """Beautified pseudotime scatter panels."""
    pt = np.asarray(adata.obs["pseudotime"], dtype=float)

    def _styled_scatter(x, y, xlabel, ylabel, title, filename, c=None, cmap="viridis"):
        fig, ax = plt.subplots(figsize=(7.5, 6))
        if c is not None:
            sc = ax.scatter(x, y, c=c, cmap=cmap, s=8, alpha=0.65, linewidths=0, rasterized=True)
            cb = fig.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label(ylabel if c is y else "value")
        else:
            ax.scatter(x, y, s=8, alpha=0.55, c="#4C72B0", linewidths=0, rasterized=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.25)
        save_figure(fig, save_dir, filename, subdir=subdir, dpi=_adaptive_dpi(len(x)))

    if gene_name and gene_name in adata.var_names:
        idx = adata.var_names.get_loc(gene_name)
        x = adata.X[:, idx]
        if hasattr(x, "toarray"):
            x = x.toarray().flatten()
        else:
            x = np.asarray(x).flatten()
        _styled_scatter(
            pt,
            x,
            "Pseudotime",
            gene_name,
            f"Pseudotime vs {gene_name}",
            "pseudotime_vs_gene.png",
        )

    if "time" in adata.obs or getattr(adata.obs, "columns", None) is not None:
        time_key = "time"
        if time_key in adata.obs:
            time_vals = adata.obs[time_key].astype(float).values
            _styled_scatter(
                pt,
                time_vals,
                "Pseudotime",
                "Time",
                "Pseudotime vs time",
                "pseudotime_vs_time.png",
            )

    if "potential" in adata.obs:
        pot = np.asarray(adata.obs["potential"], dtype=float)
        _styled_scatter(
            pt,
            pot,
            "Pseudotime",
            "Potential",
            "Pseudotime vs potential",
            "pseudotime_vs_potential.png",
            c=pot,
            cmap="RdYlBu_r",
        )


def run_standard_training_figures(
    adata,
    save_dir: str,
    config=None,
    *,
    top_gene: Optional[str] = None,
    subdir: str = "figures",
):
    """Shared post-training UMAP + pseudotime figures for dataset scripts."""
    plot_training_umap_overview(adata, save_dir, config=config, subdir=subdir)
    plot_pseudotime_relationships(adata, save_dir, gene_name=top_gene, subdir=subdir)
