#!/usr/bin/env python3
"""Supplementary Figure 1: cell-type UMAPs for three datasets.

Layout
------
  Row 1: (a) GSE155622  |  (c) HGSOC
  Row 2: (b) GSE141259  (= checkpoint formal metacelltype + cell.type UMAP)

Colors follow each dataset's adopted checkpoint cell-type UMAP palette:
  - GSE155622 / HGSOC: ``dataset_pipeline`` celltype palettes (same as
    ``umap_training_overview.png`` cell-type panel)
  - GSE141259: hierarchical palette used by
    ``GSE141259_metacelltype_formal_celltype_umap.png``

Default output:
  output_file/Supplementary_figure1.png

Usage:
  python output_file/Supplementary_figure1.py
  python output_file/Supplementary_figure1.py /path/to/out.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_pipeline import (  # noqa: E402
    GSE141259_CELLTYPE_PALETTE,
    GSE155622_CELLTYPE_PALETTE,
    HGSOC_CELLTYPE_PALETTE,
)

# ---------------------------------------------------------------------------
# Paths (adopted checkpoints)
# ---------------------------------------------------------------------------
CK_PAIN = ROOT / (
    "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
CK_LUNG = ROOT / (
    "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
CK_HG = ROOT / (
    "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)

LUNG_PALETTE = CK_LUNG / "figures" / "GSE141259_umap_hierarchical_palette.csv"
LUNG_MAP = CK_LUNG / "figures" / "GSE141259_metacelltype_formal_label_mapping.csv"

DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_figure1.png"

INK = "#1f2933"
MUTED = "#7b8794"
MAX_CELLS = 15000


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "center",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _pt_size(n: int) -> float:
    if n > 20000:
        return 2.8
    if n > 10000:
        return 4.0
    return 6.5


def _clean_umap_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=8.5, color=MUTED)
    ax.set_ylabel("UMAP2", fontsize=8.5, color=MUTED)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")


def _panel_letter(ax, letter: str, *, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        str(letter).upper(),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
        clip_on=False,
    )


def _load_umap_frame(ck: Path, label_col: str) -> tuple[np.ndarray, pd.Series]:
    """Align training_umap.npz with obs.csv and return (xy, labels)."""
    obs = pd.read_csv(ck / "obs.csv", low_memory=False)
    if "cell" in obs.columns:
        obs.index = obs["cell"].astype(str)
    elif "barcode" in obs.columns:
        obs.index = obs["barcode"].astype(str)
    else:
        obs.index = obs.index.astype(str)
        if "Unnamed: 0" in obs.columns:
            obs.index = obs["Unnamed: 0"].astype(str)

    z = np.load(ck / "training_umap.npz", allow_pickle=True)
    umap_idx = pd.Index(np.asarray(z["index"]).astype(str))
    xy_all = np.asarray(z["X_umap"], dtype=float)

    common = umap_idx.intersection(obs.index)
    if len(common) == 0:
        raise RuntimeError(f"no overlapping barcodes between UMAP and obs in {ck.name}")
    obs = obs.loc[common]
    mapper = {b: i for i, b in enumerate(umap_idx)}
    xy = xy_all[[mapper[b] for b in common]]
    labels = obs[label_col].astype(str)

    finite = np.isfinite(xy).all(axis=1) & labels.notna().to_numpy()
    xy = xy[finite]
    labels = labels.loc[finite]

    # Drop unassigned for lung metacelltype if present
    if label_col == "metacelltype":
        keep = labels.to_numpy() != "unassigned"
        xy = xy[keep]
        labels = labels.loc[keep]

    return xy, labels


def _subsample(xy: np.ndarray, labels: pd.Series, *, max_cells: int = MAX_CELLS, seed: int = 0):
    labels = pd.Series(np.asarray(labels).astype(str))
    n = len(labels)
    if n <= max_cells:
        return xy, labels
    rng = np.random.default_rng(seed)
    idx_parts = []
    lab_arr = labels.to_numpy()
    for lab in pd.unique(lab_arr):
        pos = np.flatnonzero(lab_arr == lab)
        k = max(1, int(round(max_cells * (len(pos) / n))))
        k = min(k, len(pos))
        idx_parts.append(rng.choice(pos, size=k, replace=False))
    idx = np.concatenate(idx_parts)
    if len(idx) > max_cells:
        idx = rng.choice(idx, size=max_cells, replace=False)
    return xy[idx], labels.iloc[idx].reset_index(drop=True)


def _scatter_umap_only(
    ax,
    xy: np.ndarray,
    labels: pd.Series,
    palette: dict[str, str],
    *,
    title: str | None = None,
    letter: str | None = None,
) -> pd.Series:
    """Draw colored UMAP; return counts for an external legend axis."""
    xy, labels = _subsample(xy, labels)
    size = _pt_size(len(labels))
    counts = labels.value_counts()
    for lab in counts.index[::-1]:
        m = labels.to_numpy() == lab
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            c=palette.get(str(lab), "#999999"),
            s=size,
            alpha=0.88,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    _clean_umap_ax(ax)
    if title:
        ax.set_title(title, loc="center", fontsize=10, fontweight="bold", color=INK, pad=4)
    if letter:
        _panel_letter(ax, letter, x=-0.12, y=1.08)
    return counts


def _legend_simple(
    ax,
    counts: pd.Series,
    palette: dict[str, str],
    *,
    title: str | None = None,
    order: list[str] | None = None,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="center", fontsize=8.5, fontweight="bold", color=INK, pad=2)
    if order is None:
        ordered = [k for k in palette if k in counts.index]
        ordered += [k for k in counts.index if k not in ordered]
    else:
        ordered = [k for k in order if k in counts.index]
        ordered += [k for k in counts.index if k not in ordered]
    ys = np.linspace(0.92, 0.08, max(len(ordered), 2))
    for y, lab in zip(ys, ordered):
        ax.scatter(
            [0.08],
            [y],
            s=36,
            c=[palette.get(str(lab), "#999999")],
            transform=ax.transAxes,
            edgecolors="none",
            zorder=3,
        )
        ax.text(
            0.20,
            y,
            f"{lab}  ({int(counts[lab]):,})",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=6.6,
            color=INK,
        )


def _load_gse141259() -> tuple[np.ndarray, pd.Series, pd.Series, dict, dict, pd.DataFrame, pd.DataFrame]:
    pal = pd.read_csv(LUNG_PALETTE)
    fmap = pd.read_csv(LUNG_MAP)
    key_to_formal = dict(
        zip(fmap["metacelltype"].astype(str), fmap["formal_label"].astype(str))
    )
    major_pal = {}
    for _, r in pal.drop_duplicates("formal_label").iterrows():
        major_pal[str(r["formal_label"])] = str(r["parent_color"])
    for k, formal in key_to_formal.items():
        major_pal.setdefault(formal, GSE141259_CELLTYPE_PALETTE.get(k, "#999999"))
    sub_pal = dict(zip(pal["cell.type"].astype(str), pal["subtype_color"].astype(str)))

    obs = pd.read_csv(CK_LUNG / "obs.csv", low_memory=False)
    if "Unnamed: 0" in obs.columns:
        obs.index = obs["Unnamed: 0"].astype(str)
    else:
        obs.index = obs.index.astype(str)
    z = np.load(CK_LUNG / "training_umap.npz", allow_pickle=True)
    umap_idx = pd.Index(np.asarray(z["index"]).astype(str))
    xy_all = np.asarray(z["X_umap"], dtype=float)
    common = umap_idx.intersection(obs.index)
    obs = obs.loc[common]
    mapper = {b: i for i, b in enumerate(umap_idx)}
    xy = xy_all[[mapper[b] for b in common]]
    keep = (obs["metacelltype"].astype(str) != "unassigned").to_numpy() & np.isfinite(xy).all(
        axis=1
    )
    xy = xy[keep]
    obs = obs.loc[keep]
    formal = obs["metacelltype"].astype(str).map(lambda x: key_to_formal.get(x, x))
    subtypes = obs["cell.type"].astype(str)
    return xy, formal, subtypes, major_pal, sub_pal, pal, fmap


def _legend_subtype_grouped(ax, pal: pd.DataFrame, fmap: pd.DataFrame, major_pal: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Subtypes", loc="center", fontsize=8.5, fontweight="bold", color=INK, pad=2)
    lines = []
    for _, r in fmap.iterrows():
        key = str(r["metacelltype"])
        formal_lab = str(r["formal_label"])
        lines.append(("header", formal_lab, major_pal.get(formal_lab, "#999999"), 0))
        sub = pal.loc[pal["metacelltype"].astype(str) == key]
        for _, row in sub.iterrows():
            lines.append(
                ("item", str(row["cell.type"]), str(row["subtype_color"]), int(row["n"]))
            )
    ys = np.linspace(0.985, 0.015, max(len(lines), 2))
    for y, (kind, name, color, n) in zip(ys, lines):
        if kind == "header":
            ax.scatter(
                [0.05],
                [y],
                s=28,
                c=[color],
                transform=ax.transAxes,
                marker="s",
                edgecolors="none",
                zorder=3,
            )
            ax.text(
                0.14,
                y,
                name,
                transform=ax.transAxes,
                va="center",
                fontsize=6.0,
                fontweight="bold",
                color=INK,
            )
        else:
            ax.scatter(
                [0.05],
                [y],
                s=16,
                c=[color],
                transform=ax.transAxes,
                edgecolors="none",
                zorder=3,
            )
            ax.text(
                0.14,
                y,
                f"{name}  ({n:,})",
                transform=ax.transAxes,
                va="center",
                fontsize=5.4,
                color="#3a3a3a",
            )


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    _apply_style()

    xy_pain, lab_pain = _load_umap_frame(CK_PAIN, "annotation")
    xy_hg, lab_hg = _load_umap_frame(CK_HG, "annotation")
    xy_lung, formal, subtypes, major_pal, sub_pal, pal, fmap = _load_gse141259()

    # Shared 2×4 grid so row1 (a|c) columns align with row2 (major|subtype)
    fig = plt.figure(figsize=(15.5, 10.8), facecolor="white")
    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.35, 0.78, 1.35, 1.15],
        wspace=0.14,
        hspace=0.38,
        left=0.045,
        right=0.99,
        top=0.94,
        bottom=0.05,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_la = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_lc = fig.add_subplot(gs[0, 3])
    ax_maj = fig.add_subplot(gs[1, 0])
    ax_lm = fig.add_subplot(gs[1, 1])
    ax_sub = fig.add_subplot(gs[1, 2])
    ax_ls = fig.add_subplot(gs[1, 3])

    # Row 1
    counts_a = _scatter_umap_only(
        ax_a, xy_pain, lab_pain, GSE155622_CELLTYPE_PALETTE, title="GSE155622 · cell types", letter="A"
    )
    _legend_simple(ax_la, counts_a, GSE155622_CELLTYPE_PALETTE)
    counts_c = _scatter_umap_only(
        ax_c, xy_hg, lab_hg, HGSOC_CELLTYPE_PALETTE, title="HGSOC · cell types", letter="C"
    )
    _legend_simple(ax_lc, counts_c, HGSOC_CELLTYPE_PALETTE)

    # Row 2
    counts_m = _scatter_umap_only(
        ax_maj, xy_lung, formal, major_pal, title="Major cell types", letter="B"
    )
    formal_order = [str(r.formal_label) for _, r in fmap.iterrows()]
    _legend_simple(ax_lm, counts_m, major_pal, title="Major cell types", order=formal_order)
    _scatter_umap_only(ax_sub, xy_lung, subtypes, sub_pal, title="Subtypes")
    _legend_subtype_grouped(ax_ls, pal, fmap, major_pal)

    # Panel-b block title centered in the gap between the two UMAP axes
    fig.canvas.draw()
    p0 = ax_maj.get_position()
    p1 = ax_sub.get_position()
    mid_x = 0.5 * (p0.x1 + p1.x0)
    title_y = max(p0.y1, p1.y1) + 0.022
    fig.text(
        mid_x,
        title_y,
        "GSE141259 · cell types",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=INK,
        transform=fig.transFigure,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep grid alignment; do not let tight bbox reflow column edges.
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(out, dpi=300, facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else DEFAULT_OUT
    compose(out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
