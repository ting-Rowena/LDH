#!/usr/bin/env python3
"""Beautify PNG figures under each dataset checkpoint's ``figures/`` folder.

Regenerates publication-style UMAP / landscape / pseudotime panels from
checkpoint ``obs.csv`` + ``latent_embeddings.npz`` (+ AnnData for gene plots).
Originals are copied to ``figures/_original_backup/`` before overwrite.
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D

from plot_utils import (
    INK,
    MUTED,
    PALETTE,
    apply_publication_style,
    configure_headless,
    get_dataset_plot_style,
    polished_colorbar,
    save_figure,
    style_axis,
    subsample_for_plot,
)

warnings.filterwarnings("ignore", category=UserWarning)
configure_headless(show_figures=False)
apply_publication_style()

BASE = Path(__file__).resolve().parent

CKPTS = {
    "GSE155622": BASE
    / "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1",
    "GSE141259": BASE
    / "GSE141259_checkpoints_5000_5000_512_0.06_recon0.01_valD28_timeX_lossnorm_qp_d0p01_z0p5_k0p2_ld1",
    "HGSOC": BASE
    / "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1",
}

# Formal display names for GSE141259 metacelltype (user-edited mapping)
GSE141259_FORMAL_MAP_PATH = (
    CKPTS["GSE141259"] / "figures" / "GSE141259_metacelltype_formal_label_mapping.csv"
)


def _backup_pngs(fig_dir: Path) -> None:
    bak = fig_dir / "_original_backup"
    bak.mkdir(parents=True, exist_ok=True)
    for p in fig_dir.glob("*.png"):
        dest = bak / p.name
        if not dest.exists():
            shutil.copy2(p, dest)


def _load_formal_map(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["metacelltype"].astype(str), df["formal_label"].astype(str)))


def _ensure_umap_from_latent(adata, *, n_neighbors: int = 15, min_dist: float = 0.35):
    """Build latent-space UMAP into ``X_umap_latent`` on ``X_latent_pca`` / ``X_latent``."""
    if "X_umap_latent" in adata.obsm:
        return adata

    if "X_latent_pca" in adata.obsm:
        use_rep = "X_latent_pca"
        n_pcs = min(20, adata.obsm[use_rep].shape[1])
    elif "X_latent" in adata.obsm:
        use_rep = "X_latent"
        n_pcs = min(40, adata.obsm[use_rep].shape[1])
    else:
        print("  [warn] no latent embedding; skip X_umap_latent", flush=True)
        return adata

    X = np.asarray(adata.obsm[use_rep][:, :n_pcs], dtype=float)
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X).all(axis=1)).sum())
        print(f"  [warn] {use_rep}: nan_to_num on {n_bad} non-finite cells", flush=True)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  computing latent-space UMAP on {use_rep} ...", flush=True)
    from umap import UMAP as _UMAP
    adata.obsm["X_umap_latent"] = _UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
    ).fit_transform(X)
    return adata


def _subset_to_training_genes(adata, ckpt: Path):
    """Restrict to checkpoint HVG panel so expression UMAP matches training figures."""
    import json

    path = Path(ckpt) / "training_var_names.json"
    if not path.is_file():
        print(f"  [warn] no {path.name}; expression UMAP uses all {adata.n_vars} genes", flush=True)
        return adata
    genes = json.loads(path.read_text())
    if not isinstance(genes, list) or not genes:
        return adata
    keep = [g for g in genes if g in adata.var_names]
    if len(keep) < 100:
        print(f"  [warn] only {len(keep)} training genes found in AnnData; skip subset", flush=True)
        return adata
    print(f"  subset to training panel: {adata.n_vars} → {len(keep)} genes", flush=True)
    return adata[:, keep].copy()


def _looks_like_counts(adata) -> bool:
    """Heuristic: raw/integer-like counts vs already log-normalized expression."""
    import scipy.sparse as sp

    X = adata.X
    try:
        n = adata.n_obs
        # sample many rows so empty leading blocks don't fool the heuristic
        step = max(1, n // 500)
        rows = np.arange(0, n, step)[:500]
        if sp.issparse(X):
            sample = np.asarray(X[rows].data, dtype=float)
            if sample.size == 0:
                sample = np.asarray(X[rows].toarray(), dtype=float).ravel()
        else:
            sample = np.asarray(X[rows], dtype=float).ravel()
        sample = sample[:20000]
    except Exception:
        return False
    if sample.size == 0:
        return False
    mx = float(np.nanmax(sample))
    mean = float(np.nanmean(sample))
    # log1p-normalized matrices rarely exceed ~10–15; counts often >>20
    return mx > 20.0 and mean < 5.0


def _ensure_log_normalized(adata):
    """Match training prep: total-count normalize + log1p when X looks like raw counts."""
    if not _looks_like_counts(adata):
        return adata
    print("  normalize_total + log1p (raw counts detected) ...", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _training_umap_path(ckpt: Path) -> Path:
    return Path(ckpt) / "training_umap.npz"


def _ensure_training_umap(adata, dataset_key: str, ckpt: Path):
    """
    Build / load the same expression UMAP used at training plot time.

    Recipe: log-normalize if needed → training HVG panel → PlotStyle PCA/neighbors/UMAP.
    Persists ``training_umap.npz`` (index + X_umap) so all figures share one coordinate set.
    """
    from dataset_pipeline import PLOT_STYLES, compute_training_umap

    ckpt = Path(ckpt)
    out = _training_umap_path(ckpt)
    obs_names = adata.obs_names.astype(str)

    if out.is_file():
        z = np.load(out, allow_pickle=True)
        idx = pd.Index(z["index"].astype(str))
        coords = np.asarray(z["X_umap"], dtype=float)
        if len(idx) == len(coords):
            # Align to current adata order; missing cells → NaN then drop later if needed
            aligned = np.full((adata.n_obs, coords.shape[1]), np.nan, dtype=float)
            mapper = {b: i for i, b in enumerate(idx)}
            rows = []
            src = []
            for i, b in enumerate(obs_names):
                j = mapper.get(b)
                if j is not None:
                    rows.append(i)
                    src.append(j)
            if len(rows) >= max(100, int(0.5 * adata.n_obs)):
                aligned[rows] = coords[src]
                finite = np.isfinite(aligned).all(axis=1)
                if finite.mean() >= 0.5:
                    adata.obsm["X_umap"] = aligned
                    print(
                        f"  loaded training X_umap from {out.name} "
                        f"({finite.sum()}/{adata.n_obs} cells aligned)",
                        flush=True,
                    )
                    return adata
            print(f"  [warn] {out.name} barcode overlap too low; recomputing", flush=True)

    style = PLOT_STYLES[dataset_key]
    work = adata.copy()
    work = _ensure_log_normalized(work)
    work = _subset_to_training_genes(work, ckpt)
    print(
        f"  computing training X_umap ({dataset_key}): "
        f"n_pcs={style.n_pcs}, n_neighbors={style.n_neighbors}, "
        f"min_dist={style.umap_min_dist}, n_genes={work.n_vars} ...",
        flush=True,
    )
    compute_training_umap(work, plot_style=style, force=True)
    umap = np.asarray(work.obsm["X_umap"], dtype=float)
    # work may be gene-subset only; same cells/order as adata
    if work.n_obs != adata.n_obs or not np.array_equal(
        work.obs_names.astype(str).values, obs_names.values
    ):
        raise RuntimeError("training UMAP cell identity drifted from adata")
    adata.obsm["X_umap"] = umap
    np.savez_compressed(
        out,
        index=np.asarray(obs_names, dtype=object),
        X_umap=umap,
        dataset=np.asarray(dataset_key),
        n_pcs=np.asarray(style.n_pcs),
        n_neighbors=np.asarray(style.n_neighbors),
        umap_min_dist=np.asarray(style.umap_min_dist),
    )
    print(f"  saved training X_umap → {out}", flush=True)
    return adata


def _expression_umap_style(dataset_key: str):
    """Deprecated path retained for --umap-space expression override."""
    from dataset_pipeline import PLOT_STYLES

    return PLOT_STYLES[dataset_key]


def _ensure_umap_expression(adata, dataset_key: str, ckpt: Optional[Path] = None):
    """Legacy alias: training-recipe expression UMAP stored as X_umap / X_umap_expr."""
    if ckpt is None:
        raise ValueError("ckpt is required to build training expression UMAP")
    adata = _ensure_training_umap(adata, dataset_key, ckpt)
    adata.obsm["X_umap_expr"] = np.asarray(adata.obsm["X_umap"], dtype=float)
    return adata


def _coords(adata, umap_key: str = "X_umap"):
    if umap_key in adata.obsm:
        return np.asarray(adata.obsm[umap_key], dtype=float)
    if "X_umap" in adata.obsm:
        return np.asarray(adata.obsm["X_umap"], dtype=float)
    raise KeyError(f"No embedding key {umap_key!r} or X_umap in adata.obsm")


def _clean_umap_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=10, color=MUTED)
    ax.set_ylabel("UMAP2", fontsize=10, color=MUTED)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")


def _pt_size(n: int) -> float:
    if n > 20000:
        return 2.8
    if n > 10000:
        return 4.0
    return 8.0


def _palette_colors(n: int) -> Sequence[str]:
    if n <= len(PALETTE):
        return PALETTE[:n]
    if n <= 20:
        return sc.pl.palettes.default_20[:n]
    if n <= 28:
        return sc.pl.palettes.default_28[:n]
    base = list(sc.pl.palettes.default_102)
    while len(base) < n:
        base += base
    return base[:n]


def _resolve_keys(adata, dataset_key: str):
    style = get_dataset_plot_style(dataset_key)
    ct = style.celltype_key if style else None
    stage = style.stage_key if style else "stage"
    if ct not in adata.obs:
        for cand in ("annotation", "metacelltype", "celltype", "cell_type"):
            if cand in adata.obs:
                ct = cand
                break
    if stage not in adata.obs:
        for cand in ("stage", "condition", "treatment"):
            if cand in adata.obs:
                stage = cand
                break
    return ct, stage, style


def _display_labels(series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    s = series.astype(str)
    if not mapping:
        return s
    return s.map(lambda x: mapping.get(x, x))


def _scatter_discrete(ax, coords, labels, *, palette: Optional[Dict[str, str]] = None,
                      size: float = 4.0, title: str = "", legend_title: str = ""):
    labels = pd.Series(labels).astype(str)
    counts = labels.value_counts()
    cats = counts.index.tolist()
    if palette:
        colors = [palette.get(c, "#999999") for c in cats]
        # fill missing with fallback cycle
        miss = [i for i, c in enumerate(cats) if c not in palette]
        if miss:
            fb = _palette_colors(len(miss))
            for j, i in enumerate(miss):
                colors[i] = fb[j]
    else:
        colors = _palette_colors(len(cats))
    cmap = {c: colors[i] for i, c in enumerate(cats)}

    # rare types on top
    order = np.argsort([counts[l] for l in labels])[::-1]  # abundant first (underneath)
    order = order[::-1]  # rare last (on top)
    xy = coords[order]
    labs = labels.to_numpy()[order]
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=[cmap[l] for l in labs],
        s=size, alpha=0.88, linewidths=0, rasterized=True,
    )
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[c],
               markersize=7, label=f"{c} ({counts[c]:,})")
        for c in cats
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=7.5, frameon=False, title=legend_title or None, title_fontsize=9,
        borderaxespad=0, handletextpad=0.35, labelspacing=0.32,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    _clean_umap_ax(ax)


def _scatter_continuous(ax, coords, values, *, cmap: str, size: float, title: str, cbar_label: str):
    vals = np.asarray(values, dtype=float)
    sc_map = ax.scatter(
        coords[:, 0], coords[:, 1], c=vals, cmap=cmap,
        s=size, alpha=0.88, linewidths=0, rasterized=True,
    )
    polished_colorbar(sc_map, ax, label=cbar_label)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    _clean_umap_ax(ax)


def _umap_basis_label(umap_key: str) -> str:
    if "latent" in umap_key:
        return "latent PCA→UMAP"
    if umap_key == "X_umap":
        return "training X_umap"
    if "expr" in umap_key:
        return "expression PCA→UMAP"
    return "UMAP"


def plot_umap_overview(adata, save_dir: Path, dataset_key: str, display_map: Dict[str, str],
                       *, umap_key: str = "X_umap"):
    """Cell-type overview on shared training X_umap (default)."""
    ct, stage, style = _resolve_keys(adata, dataset_key)
    plot = subsample_for_plot(adata, max_cells=15000, label_key=ct)
    coords = _coords(plot, umap_key)
    size = _pt_size(plot.n_obs)
    ct_palette = dict(style.celltype_palette) if style else {}
    stage_palette = dict(style.stage_palette) if style else {}

    # Remap palette keys to display labels when using formal names
    ct_labels = _display_labels(plot.obs[ct], display_map) if ct else None
    if display_map and ct_palette:
        ct_palette = {display_map.get(k, k): v for k, v in ct_palette.items()}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.2))
    fig.patch.set_facecolor("white")

    if ct_labels is not None:
        _scatter_discrete(
            axes[0, 0], coords, ct_labels, palette=ct_palette, size=size,
            title="Cell type", legend_title="Cell type",
        )
    else:
        axes[0, 0].axis("off")

    if stage in plot.obs:
        _scatter_discrete(
            axes[0, 1], coords, plot.obs[stage].astype(str),
            palette=stage_palette, size=size, title="Stage", legend_title="Stage",
        )
    else:
        axes[0, 1].axis("off")

    if "pseudotime" in plot.obs:
        cmap = style.pseudotime_cmap if style else "magma"
        _scatter_continuous(
            axes[1, 0], coords, plot.obs["pseudotime"], cmap=cmap, size=size,
            title="Pseudotime", cbar_label="Pseudotime",
        )
    else:
        axes[1, 0].axis("off")

    pot_key = "potential_stationary" if "potential_stationary" in plot.obs else "potential"
    if pot_key in plot.obs:
        cmap = style.potential_cmap if style else "RdYlBu_r"
        _scatter_continuous(
            axes[1, 1], coords, plot.obs[pot_key], cmap=cmap, size=size,
            title=r"Potential $U(z,t)$", cbar_label=r"$U(z,t)$",
        )
    else:
        axes[1, 1].axis("off")

    n_show, n_tot = plot.n_obs, adata.n_obs
    basis = _umap_basis_label(umap_key)
    subtitle = f"{dataset_key} training overview ({basis})"
    if n_show < n_tot:
        subtitle += f"  ·  {n_show:,} / {n_tot:,} cells shown"
    fig.suptitle(subtitle, fontsize=14, fontweight="bold", color=INK, y=0.995)
    fig.subplots_adjust(wspace=0.55, hspace=0.28, left=0.04, right=0.88, top=0.94, bottom=0.04)
    save_figure(fig, str(save_dir), "umap_training_overview.png", subdir=".", dpi=300)


def plot_potential_landscape(adata, save_dir: Path, dataset_key: str, display_map: Dict[str, str],
                             *, umap_key: str = "X_umap"):
    """Potential landscape on shared training X_umap (default)."""
    ct, _, style = _resolve_keys(adata, dataset_key)
    plot = subsample_for_plot(adata, max_cells=15000, label_key=ct)
    coords = _coords(plot, umap_key)
    size = _pt_size(plot.n_obs)
    pot_key = "potential_stationary" if "potential_stationary" in plot.obs else "potential"
    cmap = style.potential_cmap if style else "RdYlBu_r"
    ct_palette = dict(style.celltype_palette) if style else {}
    ct_labels = _display_labels(plot.obs[ct], display_map) if ct else None
    if display_map and ct_palette:
        ct_palette = {display_map.get(k, k): v for k, v in ct_palette.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    fig.patch.set_facecolor("white")
    _scatter_continuous(
        axes[0], coords, plot.obs[pot_key], cmap=cmap, size=size,
        title="Learned potential on UMAP", cbar_label=r"$U(z,t)$",
    )
    if ct_labels is not None:
        _scatter_discrete(
            axes[1], coords, ct_labels, palette=ct_palette, size=size,
            title="Cell type labels", legend_title="Cell type",
        )
    else:
        axes[1].axis("off")
    fig.suptitle(
        f"{dataset_key} potential landscape ({_umap_basis_label(umap_key)})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(wspace=0.55, left=0.04, right=0.88, top=0.88, bottom=0.08)
    save_figure(fig, str(save_dir), "potential_landscape_umap.png", subdir=".", dpi=300)


def plot_kinetics_overview(adata, save_dir: Path, dataset_key: str, display_map: Dict[str, str],
                           *, umap_key: str = "X_umap"):
    """Kinetics overview on shared training X_umap (default)."""
    ct, stage, style = _resolve_keys(adata, dataset_key)
    plot = subsample_for_plot(adata, max_cells=15000, label_key=ct)
    coords = _coords(plot, umap_key)
    size = _pt_size(plot.n_obs)
    ct_palette = dict(style.celltype_palette) if style else {}
    stage_palette = dict(style.stage_palette) if style else {}
    ct_labels = _display_labels(plot.obs[ct], display_map) if ct else None
    if display_map and ct_palette:
        ct_palette = {display_map.get(k, k): v for k, v in ct_palette.items()}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.2))
    fig.patch.set_facecolor("white")

    if ct_labels is not None:
        _scatter_discrete(axes[0, 0], coords, ct_labels, palette=ct_palette, size=size,
                          title="Cell type", legend_title="Cell type")
    if "pseudotime" in plot.obs:
        cmap = style.pseudotime_cmap if style else "magma"
        _scatter_continuous(axes[0, 1], coords, plot.obs["pseudotime"], cmap=cmap, size=size,
                            title="Pseudotime", cbar_label="Pseudotime")
    if stage in plot.obs:
        _scatter_discrete(axes[1, 0], coords, plot.obs[stage].astype(str),
                          palette=stage_palette, size=size, title="Stage", legend_title="Stage")
    else:
        axes[1, 0].axis("off")

    # Bottom-right: cell type + optional momentum arrows from latent PCA gradient of potential
    if ct_labels is not None:
        _scatter_discrete(axes[1, 1], coords, ct_labels, palette=ct_palette, size=size,
                          title="Cell type + potential gradient", legend_title="Cell type")
        if "potential" in plot.obs or "potential_stationary" in plot.obs:
            pot = plot.obs.get("potential_stationary", plot.obs.get("potential")).astype(float).values
            # local finite-diff proxy: toward lower potential among kNN in UMAP
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(16, max(5, plot.n_obs // 500))).fit(coords)
                _, idx = nn.kneighbors(coords)
                # vector from cell to neighbor with lowest potential
                neigh_p = pot[idx]
                j = np.argmin(neigh_p, axis=1)
                targets = idx[np.arange(len(idx)), j]
                v = coords[targets] - coords
                # normalize
                nrm = np.linalg.norm(v, axis=1, keepdims=True)
                nrm[nrm < 1e-8] = 1.0
                v = v / nrm
                step = max(1, plot.n_obs // 1800)
                sel = np.arange(0, plot.n_obs, step)
                axes[1, 1].quiver(
                    coords[sel, 0], coords[sel, 1], v[sel, 0], v[sel, 1],
                    angles="xy", scale_units="xy", scale=0.9,
                    width=0.0022, alpha=0.45, color="#1f2933", zorder=3,
                )
            except Exception as exc:
                print(f"  [warn] gradient overlay skipped: {exc}")

    fig.suptitle(
        f"{dataset_key} kinetics overview ({_umap_basis_label(umap_key)})",
        fontsize=14, fontweight="bold", color=INK, y=0.995,
    )
    fig.subplots_adjust(wspace=0.55, hspace=0.28, left=0.04, right=0.88, top=0.94, bottom=0.04)
    save_figure(fig, str(save_dir), "kinetics_overview.png", subdir=".", dpi=300)


def plot_pseudotime_panels(adata, save_dir: Path, dataset_key: str, gene: Optional[str] = None):
    if "pseudotime" not in adata.obs:
        return
    pt = np.asarray(adata.obs["pseudotime"], dtype=float)

    def _one(x, y, xlabel, ylabel, title, filename, c=None, cmap="viridis"):
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        fig.patch.set_facecolor("white")
        if c is not None:
            sc_map = ax.scatter(x, y, c=c, cmap=cmap, s=6, alpha=0.55, linewidths=0, rasterized=True)
            polished_colorbar(sc_map, ax, label=ylabel)
        else:
            ax.scatter(x, y, s=6, alpha=0.45, c="#3a6ea5", linewidths=0, rasterized=True)
        # LOWESS trend when enough points
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            if len(x) > 200:
                sm = lowess(y, x, frac=0.2, return_sorted=True)
                ax.plot(sm[:, 0], sm[:, 1], color="#e07a5f", lw=2.4, zorder=3, label="LOWESS")
                ax.legend(frameon=False, loc="best", fontsize=9)
        except Exception:
            pass
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{dataset_key}: {title}", fontweight="bold", color=INK)
        style_axis(ax, grid_axis="both")
        ax.grid(True, color="#e3e8ee", linewidth=0.8, alpha=1.0)
        save_figure(fig, str(save_dir), filename, subdir=".", dpi=300)

    if "time" in adata.obs:
        _one(pt, adata.obs["time"].astype(float).values, "Pseudotime", "Biological time",
             "Pseudotime vs time", "pseudotime_vs_time.png")

    pot_key = "potential_stationary" if "potential_stationary" in adata.obs else "potential"
    if pot_key in adata.obs:
        pot = adata.obs[pot_key].astype(float).values
        _one(pt, pot, "Pseudotime", "Potential", "Pseudotime vs potential",
             "pseudotime_vs_potential.png", c=pot, cmap="RdYlBu_r")

    if gene and gene in adata.var_names:
        x = adata[:, gene].X
        if hasattr(x, "toarray"):
            x = x.toarray().ravel()
        else:
            x = np.asarray(x).ravel()
        _one(pt, x, "Pseudotime", gene, f"Pseudotime vs {gene}", "pseudotime_vs_gene.png")
        _one(pt, x, "Pseudotime", gene, f"Predicted / observed {gene}", "predicted_genes_vs_pseudotime.png")


def plot_gene_trends(adata, save_dir: Path, dataset_key: str, genes: Sequence[str]):
    from statsmodels.nonparametric.smoothers_lowess import lowess

    genes = [g for g in genes if g in adata.var_names][:3]
    if not genes:
        return
    ct, stage, style = _resolve_keys(adata, dataset_key)
    group = stage if stage in adata.obs else ct
    fig, axes = plt.subplots(len(genes), 2, figsize=(14.5, 4.6 * len(genes)))
    if len(genes) == 1:
        axes = np.array([axes])
    fig.patch.set_facecolor("white")
    plot = subsample_for_plot(adata, max_cells=12000, label_key=ct)
    coords = _coords(plot, "X_umap")
    size = _pt_size(plot.n_obs)

    for i, gene in enumerate(genes):
        # violin by stage
        try:
            sc.pl.violin(adata, gene, groupby=group, ax=axes[i, 0], show=False, rotation=40,
                         stripplot=False, inner="box")
            axes[i, 0].set_title(f"{gene} by {group}", fontweight="bold")
            style_axis(axes[i, 0], grid_axis="y")
        except Exception as exc:
            axes[i, 0].text(0.5, 0.5, f"violin failed:\n{exc}", ha="center", va="center")
            axes[i, 0].axis("off")

        expr = plot[:, gene].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray().ravel()
        else:
            expr = np.asarray(expr).ravel()
        sc_map = axes[i, 1].scatter(
            coords[:, 0], coords[:, 1], c=expr, cmap="viridis",
            s=size, alpha=0.85, linewidths=0, rasterized=True,
        )
        polished_colorbar(sc_map, axes[i, 1], label=gene)
        axes[i, 1].set_title(f"{gene} on UMAP", fontweight="bold")
        _clean_umap_ax(axes[i, 1])
        if "pseudotime" in plot.obs:
            inset = axes[i, 1].inset_axes([0.58, 0.58, 0.38, 0.38])
            x = plot.obs["pseudotime"].astype(float).values
            inset.scatter(x, expr, s=3, alpha=0.25, c="#3a6ea5", linewidths=0)
            sm = lowess(expr, x, frac=0.2)
            inset.plot(sm[:, 0], sm[:, 1], c="#e07a5f", lw=2)
            inset.set_title("Trend", fontsize=8)
            inset.tick_params(labelsize=7)
            inset.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"{dataset_key} gene trends", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, str(save_dir), "gene_trends.png", subdir=".", dpi=300)


def plot_gene_phase(adata, save_dir: Path, dataset_key: str, g1: str, g2: str, color_by: str):
    if g1 not in adata.var_names or g2 not in adata.var_names:
        return
    x = adata[:, g1].X
    y = adata[:, g2].X
    if hasattr(x, "toarray"):
        x = x.toarray().ravel()
        y = y.toarray().ravel()
    else:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.patch.set_facecolor("white")
    if color_by in adata.obs:
        labs = adata.obs[color_by].astype(str)
        cats = labs.value_counts().index.tolist()
        cols = _palette_colors(len(cats))
        cmap = {c: cols[i] for i, c in enumerate(cats)}
        ax.scatter(x, y, c=[cmap[l] for l in labs], s=8, alpha=0.55, linewidths=0, rasterized=True)
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[c],
                          markersize=7, label=c) for c in cats[:20]]
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=False, title=color_by)
    else:
        ax.scatter(x, y, s=8, alpha=0.45, c="#3a6ea5", linewidths=0, rasterized=True)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(y, x, frac=0.25)
        ax.plot(sm[:, 0], sm[:, 1], color="#e07a5f", lw=2.4, zorder=3)
    except Exception:
        pass
    ax.set_xlabel(g1)
    ax.set_ylabel(g2)
    ax.set_title(f"{dataset_key}: {g1} vs {g2}", fontweight="bold")
    style_axis(ax, grid_axis="both")
    save_figure(fig, str(save_dir), f"gene_phase_{g1}_vs_{g2}.png", subdir=".", dpi=300)


def plot_gse141259_celltype_umap(adata, save_dir: Path, formal_map: Dict[str, str],
                                 *, umap_key: str = "X_umap"):
    """Two-row metacelltype (formal labels) + cell.type on shared training X_umap."""
    if "metacelltype" not in adata.obs or "cell.type" not in adata.obs:
        return
    plot = adata
    if plot.n_obs > 30000:
        plot = subsample_for_plot(adata, max_cells=25000, label_key="metacelltype")
    coords = _coords(plot, umap_key)
    finite = np.isfinite(coords).all(axis=1)
    if not finite.all():
        plot = plot[finite].copy()
        coords = coords[finite]
    size = _pt_size(plot.n_obs)
    formal = _display_labels(plot.obs["metacelltype"], formal_map)

    # Build palette keyed by formal names from existing palette
    style = get_dataset_plot_style("GSE141259")
    raw_pal = dict(style.celltype_palette) if style else {}
    formal_pal = {formal_map.get(k, k): v for k, v in raw_pal.items()}

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 14))
    fig.patch.set_facecolor("white")
    _scatter_discrete(axes[0], coords, formal, palette=formal_pal, size=size,
                      title="metacelltype (formal labels)", legend_title="Cell type")
    _scatter_discrete(axes[1], coords, plot.obs["cell.type"].astype(str), palette=None, size=size,
                      title="cell.type", legend_title="cell.type")
    fig.suptitle(
        f"GSE141259 cell-type UMAP ({_umap_basis_label(umap_key)})",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.subplots_adjust(hspace=0.22, left=0.06, right=0.70, top=0.95, bottom=0.04)
    save_figure(fig, str(save_dir), "GSE141259_metacelltype_formal_celltype_umap.png",
                subdir=".", dpi=300)


def load_dataset(dataset_key: str, ckpt: Path, *, need_latent_umap: bool = False):
    from celltype_analysis import DATASET_REGISTRY, load_annotated_adata
    from latent_embeddings import ensure_latent_embeddings

    profile = DATASET_REGISTRY[dataset_key]
    print(f"[{dataset_key}] loading AnnData + checkpoint obs ...", flush=True)
    adata = load_annotated_adata(profile, str(ckpt))
    # Match training prep: drop unassigned metacelltype (GSE141259)
    if dataset_key == "GSE141259" and "metacelltype" in adata.obs:
        n0 = adata.n_obs
        adata = adata[adata.obs["metacelltype"].astype(str) != "unassigned"].copy()
        print(f"  filtered unassigned: {n0} → {adata.n_obs}", flush=True)
    try:
        ensure_latent_embeddings(adata, checkpoint_dir=str(ckpt), warn=True)
    except Exception as exc:
        print(f"  [warn] latent embeddings: {exc}")
    # Shared training-recipe X_umap for all UMAP figures
    adata = _ensure_training_umap(adata, dataset_key, Path(ckpt))
    # Drop cells without finite UMAP (e.g. barcode mismatch leftovers)
    coords = np.asarray(adata.obsm["X_umap"], dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    if not finite.all():
        print(f"  drop {(~finite).sum()} cells without finite X_umap", flush=True)
        adata = adata[finite].copy()
    if need_latent_umap:
        adata = _ensure_umap_from_latent(adata)
    return adata


def pick_genes(adata, n: int = 3) -> Sequence[str]:
    # Prefer highly variable / high-variance genes present in panel
    if "highly_variable" in adata.var.columns and adata.var["highly_variable"].any():
        genes = adata.var_names[adata.var["highly_variable"]].tolist()
    else:
        genes = list(adata.var_names)
    # variance ranking on dense subsample
    X = adata.X
    if hasattr(X, "toarray"):
        # sparse: mean of squares approx via sample
        idx = np.linspace(0, adata.n_obs - 1, num=min(4000, adata.n_obs), dtype=int)
        Xs = X[idx].toarray()
    else:
        Xs = np.asarray(X)
        if Xs.shape[0] > 4000:
            idx = np.linspace(0, Xs.shape[0] - 1, num=4000, dtype=int)
            Xs = Xs[idx]
    var = Xs.var(axis=0)
    order = np.argsort(var)[::-1]
    picked = [str(adata.var_names[i]) for i in order[: max(n + 5, 10)]]
    # de-duplicate / filter mitochondrial-ish if many
    out = []
    for g in picked:
        if g.startswith(("mt-", "MT-", "Rpl", "Rps", "RPL", "RPS")):
            continue
        out.append(g)
        if len(out) >= n:
            break
    if len(out) < n:
        out = [str(adata.var_names[i]) for i in order[:n]]
    return out


def beautify_one(
    dataset_key: str,
    ckpt: Path,
    *,
    skip_genes: bool = False,
    umap_only: bool = False,
    umap_key: str = "X_umap",
):
    fig_dir = ckpt / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n======== {dataset_key} → {fig_dir} (UMAP={umap_key}) ========", flush=True)
    _backup_pngs(fig_dir)

    # Formal full names only for GSE141259 metacelltype; other datasets keep original labels.
    display_map: Dict[str, str] = {}
    if dataset_key == "GSE141259":
        display_map = _load_formal_map(GSE141259_FORMAL_MAP_PATH)

    adata = load_dataset(dataset_key, ckpt, need_latent_umap=(umap_key == "X_umap_latent"))

    # Apply formal display column without breaking analysis keys
    ct, _, _ = _resolve_keys(adata, dataset_key)
    if dataset_key == "GSE141259" and display_map and "metacelltype" in adata.obs:
        adata.obs["metacelltype_formal"] = _display_labels(adata.obs["metacelltype"], display_map)

    plot_umap_overview(adata, fig_dir, dataset_key, display_map, umap_key=umap_key)
    plot_potential_landscape(adata, fig_dir, dataset_key, display_map, umap_key=umap_key)
    plot_kinetics_overview(adata, fig_dir, dataset_key, display_map, umap_key=umap_key)

    if dataset_key == "GSE141259":
        plot_gse141259_celltype_umap(adata, fig_dir, display_map, umap_key=umap_key)

    if not umap_only:
        genes = pick_genes(adata, n=3)
        print(f"  top genes for trends: {genes}", flush=True)
        plot_pseudotime_panels(adata, fig_dir, dataset_key, gene=genes[0] if genes else None)

        if not skip_genes:
            plot_gene_trends(adata, fig_dir, dataset_key, genes)
            if len(genes) >= 2:
                color_by = "annotation" if "annotation" in adata.obs else (ct or "stage")
                plot_gene_phase(adata, fig_dir, dataset_key, genes[0], genes[1], color_by=color_by)

    print(f"[{dataset_key}] done. Outputs in {fig_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Beautify checkpoint figures/")
    p.add_argument("--datasets", nargs="+", default=list(CKPTS.keys()),
                   choices=list(CKPTS.keys()))
    p.add_argument("--skip-genes", action="store_true")
    p.add_argument("--umap-only", action="store_true",
                   help="Only regenerate UMAP panels (overview / landscape / kinetics / celltype)")
    p.add_argument(
        "--umap-space",
        choices=["training", "expression", "latent"],
        default="training",
        help="Embedding for UMAP figures (default: shared training X_umap)",
    )
    p.add_argument(
        "--recompute-umap",
        action="store_true",
        help="Ignore cached training_umap.npz and recompute training X_umap",
    )
    args = p.parse_args()
    if args.umap_space == "training":
        umap_key = "X_umap"
    elif args.umap_space == "expression":
        umap_key = "X_umap_expr"
    else:
        umap_key = "X_umap_latent"
    if args.recompute_umap:
        for key in args.datasets:
            pth = _training_umap_path(CKPTS[key])
            if pth.exists():
                pth.unlink()
                print(f"removed cached {pth}", flush=True)
    for key in args.datasets:
        beautify_one(
            key,
            CKPTS[key],
            skip_genes=args.skip_genes,
            umap_only=args.umap_only,
            umap_key=umap_key,
        )


if __name__ == "__main__":
    main()
