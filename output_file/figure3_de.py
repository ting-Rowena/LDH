#!/usr/bin/env python3
"""Figure 3: SNIIC module heatmap (neurons ordered by −U_rel) + Atf3-KO 1×3 rollout tracks.

Computes SNIIC gene z-scores on the adopted GSE155622 checkpoint and hybrid
Atf3 knockout tracks, then composes the published two-block layout.

Default output:
  output_file/figure3_de.png

Usage:
  python output_file/figure3_de.py
  python output_file/figure3_de.py /path/to/out.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import (  # noqa: E402
    CACHE,
    SNIIC_HEATMAP_GENES,
    load_pain_neuron_expression,
    pain_atf3_ko_track,
)
from _supp_compose import compose_side_by_side  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "figure3_de.png"
HEAT_PATH = CACHE / "figure3_de_heatmap.png"
KO_PATH = CACHE / "figure3_de_ko_tracks.png"
# Backup figure3_de.png geometry (heatmap | 1×3 KO).
PUBLISHED_SIZE = (4397, 887)
PUBLISHED_GAP = 40
# Right block matches archived Fig2C_Atf3_KO.png aspect after height-normalization.
KO_REF = (3071, 905)
KO_WIDTH = int(round(KO_REF[0] * PUBLISHED_SIZE[1] / KO_REF[1]))  # 3010
LEFT_WIDTH = PUBLISHED_SIZE[0] - PUBLISHED_GAP - KO_WIDTH  # 1347

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e3e8ee"
PANEL_BG = "#ffffff"
# Archived Fig2C / backup figure3_de track colors (pale teal / rose / cream),
# not the saturated knockout-script hexes (#5F8D4E / #6E2C4B / #B38B6D).
MODULE_COLORS = [("SNIIC1", "#9EC1C0"), ("SNIIC2", "#E0BFB8"), ("SNIIC3", "#F0E4D2")]
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "journal_zscore",
    [
        "#3E6F9F",
        "#7FA8C9",
        "#C9DCEB",
        "#F7F7F7",
        "#F2D0C6",
        "#D88B7C",
        "#B04A42",
    ],
)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "center",
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.9,
            "text.color": INK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": PANEL_BG,
            "axes.axisbelow": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _style_axis(ax, *, grid_axis: str = "y") -> None:
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


def _heatmap_matrix() -> tuple[np.ndarray, list[str]]:
    X, neu, genes = load_pain_neuron_expression(SNIIC_HEATMAP_GENES)
    rel = neu["potential_relative_type"].astype(float).to_numpy()
    injury = -rel
    order = np.argsort(injury)
    Xz = np.column_stack(
        [(X[:, i] - np.nanmean(X[:, i])) / (np.nanstd(X[:, i]) + 1e-12) for i in range(X.shape[1])]
    ).T
    Xz = Xz[:, order]
    n_bins = 200
    edges = np.linspace(0, Xz.shape[1], n_bins + 1).astype(int)
    mat = np.zeros((len(genes), n_bins))
    for b in range(n_bins):
        sl = slice(edges[b], edges[b + 1])
        mat[:, b] = np.nanmean(Xz[:, sl], axis=1) if edges[b + 1] > edges[b] else np.nan
    return mat, list(genes)


def _draw_heatmap(out: Path) -> Path:
    mat, genes = _heatmap_matrix()
    vmax = float(max(1.0, np.nanpercentile(np.abs(mat), 98)))
    vmax = float(np.ceil(vmax * 10.0) / 10.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(6.6, 3.35))
    im = ax.imshow(mat, aspect="auto", cmap=HEAT_CMAP, norm=norm, interpolation="nearest")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels([rf"$\mathit{{{g}}}$" for g in genes])
    n_bins = mat.shape[1]
    ax.set_xticks([0, n_bins // 2, n_bins - 1])
    ax.set_xticklabels(
        [
            r"normal" + "\n" + r"(high $U_{\mathrm{rel}}$)",
            "→",
            r"injury" + "\n" + r"(low $U_{\mathrm{rel}}$)",
        ]
    )
    ax.set_xlabel(r"Neurons ordered by $-U_{\mathrm{rel}}$")
    ax.set_title("SNIIC genes along injury axis", fontweight="bold", loc="center", pad=4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Row z-score")
    cbar.set_ticks([-vmax, 0.0, vmax])
    cbar.set_ticklabels([f"{-vmax:g}", "0", f"{vmax:g}"])
    _style_axis(ax, grid_axis="none")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _draw_ko_tracks(out: Path) -> Path:
    track = pain_atf3_ko_track()
    cond = track["condition"].astype(str)
    wt = track[cond.str.lower().eq("wildtype") | cond.str.lower().eq("wt")].copy()
    kd = track[~track.index.isin(wt.index)].copy()
    if wt.empty:
        names = cond.unique()
        wt = track[cond == names[0]]
        kd = track[cond == names[-1]]
    tcol = "t" if "t" in track.columns else "time"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4), sharey=False)
    for ax, (module, color) in zip(axes, MODULE_COLORS):
        if module not in track.columns:
            ax.set_axis_off()
            continue
        ax.plot(wt[tcol], wt[module], "-o", color=color, lw=2, ms=5.5, alpha=0.9, label="WT")
        ax.plot(
            kd[tcol],
            kd[module],
            "--s",
            color=color,
            lw=2,
            ms=5.5,
            alpha=0.9,
            label=r"$\mathit{Atf3}$-KO",
        )
        ax.set_title(module, fontweight="bold", loc="center")
        ax.set_xlabel("Simulated time (days)")
        ax.set_ylabel("Module score (nearest cells)")
        _style_axis(ax, grid_axis="y")
    axes[2].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.55), frameon=False)
    fig.subplots_adjust(wspace=0.28, top=0.86, bottom=0.18, right=0.88)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    _apply_style()
    print("[figure3_de] computing SNIIC heatmap + Atf3 KO tracks...", flush=True)
    _draw_heatmap(HEAT_PATH)
    _draw_ko_tracks(KO_PATH)
    return compose_side_by_side(
        HEAT_PATH,
        KO_PATH,
        out,
        target_size=PUBLISHED_SIZE,
        gap=PUBLISHED_GAP,
        left_width=LEFT_WIDTH,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    compose(out=Path(argv[0]) if argv else DEFAULT_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
