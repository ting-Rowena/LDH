#!/usr/bin/env python3
"""Figure 5: deep-valley EOC DEG volcano (HGSOC).

Self-contained script consolidated from:
  - scripts/redraw_fig4a_polished.py (draw_deg_volcano)
  - panel_style / plot_utils (minimal helpers inlined)

Default output:
  output_file/figure5_b.png
  (= archived panel Fig4A_deep_valley_DEG.png; manuscript Figure 5)

Usage:
  python output_file/figure5_b.py
  python output_file/figure5_b.py /path/to/out.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import CK_HG, hgsoc_deep_valley_deg  # noqa: E402

CK = CK_HG
DEFAULT_OUT = Path(__file__).resolve().parent / "figure5_b.png"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PANEL_TITLE_SIZE = 10
INK = "#111111"
MUTED = "#555555"
GRID = "#e8e8e8"
UP = "#9c3d2e"
DOWN = "#2f5f8a"
NS = "#b0b0b0"
PDVS = ("BBC3", "SOD2", "ZC3H12A", "WFDC2", "FTL", "CEBPD", "IRF1")


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "axes.titlelocation": "center",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _spine(ax, *, grid: bool = False) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK, length=3, width=0.7)
    if grid:
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def draw_deg_volcano(out: Path) -> Path:
    deg = hgsoc_deep_valley_deg()
    gene = deg["gene"].astype(str)
    lfc = deg["logfoldchange"].astype(float)
    padj = deg["pval_adj"].astype(float).clip(lower=1e-300)
    y = -np.log10(padj)

    x_disp = lfc.clip(-6, 6)
    sig_up = (padj < 0.05) & (lfc > 0.5)
    sig_dn = (padj < 0.05) & (lfc < -0.5)
    ns = ~(sig_up | sig_dn)

    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.scatter(x_disp[ns], y[ns], s=7, c=NS, alpha=0.35, linewidths=0, rasterized=True, zorder=1, label="n.s.")
    ax.scatter(
        x_disp[sig_dn], y[sig_dn], s=11, c=DOWN, alpha=0.75, linewidths=0, rasterized=True, zorder=2, label="Down"
    )
    ax.scatter(
        x_disp[sig_up],
        y[sig_up],
        s=11,
        c=UP,
        alpha=0.8,
        linewidths=0,
        rasterized=True,
        zorder=3,
        label="Up in valley",
    )

    ax.axhline(-np.log10(0.05), color=MUTED, ls="--", lw=0.7, zorder=0)
    ax.axvline(0.5, color=MUTED, ls=":", lw=0.7, zorder=0)
    ax.axvline(-0.5, color=MUTED, ls=":", lw=0.7, zorder=0)

    label_genes = list(PDVS)
    extra = (
        deg.loc[sig_up & ~gene.isin(PDVS)]
        .sort_values("pval_adj")
        .head(3)["gene"]
        .astype(str)
        .tolist()
    )
    label_genes.extend(extra)

    labeled = []
    for g in label_genes:
        m = gene == g
        if not m.any():
            continue
        labeled.append((g, float(x_disp[m].iloc[0]), float(y[m].iloc[0]), float(lfc[m].iloc[0])))
    labeled.sort(key=lambda t: t[2], reverse=True)

    y_top = float(np.nanmax(y)) if len(y) else 10.0
    for i, (g, xi, yi, lfci) in enumerate(labeled):
        text_y = y_top * (0.97 - 0.07 * i)
        text_x = 5.35 if lfci >= 0 else -5.35
        ha = "left" if lfci >= 0 else "right"
        weight = "bold" if g in PDVS else "normal"
        color = UP if lfci >= 0 else DOWN
        ax.annotate(
            g,
            xy=(xi, yi),
            xytext=(text_x, text_y),
            textcoords="data",
            fontsize=6.5,
            fontweight=weight,
            color=color,
            ha=ha,
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color="#9a9a9a",
                lw=0.45,
                shrinkA=0,
                shrinkB=2,
                connectionstyle="arc3,rad=0.05",
            ),
            zorder=5,
        )
        ax.scatter([xi], [yi], s=26, facecolors="none", edgecolors=color, linewidths=1.0, zorder=5)

    ax.set_xlim(-6.2, 6.2)
    ax.set_xlabel(r"log$_2$ fold change (deep-valley vs other EOC)")
    ax.set_ylabel(r"$-\log_{10}$ adjusted $p$")
    ax.set_title("Deep-valley EOC DEGs", loc="center", pad=4, fontweight="bold", fontsize=PANEL_TITLE_SIZE)
    ax.legend(loc="upper left", fontsize=6.5, markerscale=1.3, handletextpad=0.3)
    _spine(ax, grid=True)
    ax.text(
        0.98,
        0.02,
        "Bold = PDVS genes",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=MUTED,
        style="italic",
    )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def compose(out: Path | None = None) -> Path:
    _apply_style()
    return draw_deg_volcano(out or DEFAULT_OUT)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else DEFAULT_OUT
    compose(out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
