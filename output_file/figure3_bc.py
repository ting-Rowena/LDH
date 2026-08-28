#!/usr/bin/env python3
"""Generate Figure 3 panels: cell-type landscape + SNI neuron deviation timeline.

Self-contained script consolidated from:
  - scripts/plot_fig2_celltype_relu_combined.py
  - panel_style.py / plot_utils.py (minimal helpers inlined)

Default output:
  output_file/figure3_bc.png
  (= archived panel Fig2A_celltype_landscape_and_relU.png)

Usage:
  python output_file/figure3_bc.py
  python output_file/figure3_bc.py /path/to/out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import CK_PAIN, load_obs, mean_u0_by_type, neuron_deviation_timeline  # noqa: E402

CHECKPOINT = CK_PAIN
DEFAULT_OUT = Path(__file__).resolve().parent / "figure3_bc.png"

# ---------------------------------------------------------------------------
# Style (inlined)
# ---------------------------------------------------------------------------
PANEL_TITLE_SIZE = 10
PANEL_TITLE_WEIGHT = "bold"
PANEL_TITLE_LOC = "center"
PANEL_TITLE_PAD = 4
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 7.5
ANNOT_SIZE = 7

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e3e8ee"
PANEL_BG = "#ffffff"
ACCENT_HI = "#d1495b"

NEURAL = {"Satellite", "Neuron", "Schwann"}
SHALLOW = {"Fibroblast", "VSMC", "RBC"}
NEURAL_COLOR = "#355C8A"
OTHER_COLOR = "#8B929A"
SHALLOW_COLOR = "#B07A52"


def apply_panel_title_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": PANEL_TITLE_WEIGHT,
            "axes.titlelocation": PANEL_TITLE_LOC,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def set_panel_title(ax, title: str, *, pad: float | None = None, color=None, **kwargs) -> None:
    kw = {
        "loc": PANEL_TITLE_LOC,
        "fontweight": PANEL_TITLE_WEIGHT,
        "fontsize": PANEL_TITLE_SIZE,
        "pad": PANEL_TITLE_PAD if pad is None else pad,
    }
    if color is not None:
        kw["color"] = color
    kw.update(kwargs)
    ax.set_title(title, **kw)


def style_axis(ax, *, grid_axis: str = "y"):
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


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def draw_celltype_landscape(ax: plt.Axes) -> None:
    obs = load_obs(CHECKPOINT, usecols=["annotation", "potential_stationary", "potential"])
    df = mean_u0_by_type(obs).sort_values("potential_stationary_mean", ascending=False)
    y = np.arange(len(df))

    ax.axvspan(-0.237, -0.228, color=NEURAL_COLOR, alpha=0.055, lw=0)
    ax.axvspan(-0.220, -0.200, color=SHALLOW_COLOR, alpha=0.045, lw=0)

    for yi, (_, row) in zip(y, df.iterrows()):
        label = str(row["cell_type"])
        color = (
            NEURAL_COLOR
            if label in NEURAL
            else SHALLOW_COLOR
            if label in SHALLOW
            else OTHER_COLOR
        )
        ax.errorbar(
            row["potential_stationary_mean"],
            yi,
            xerr=row["potential_stationary_std"],
            fmt="o",
            ms=7,
            mfc=color,
            mec="white",
            mew=0.75,
            ecolor=color,
            elinewidth=1.3,
            capsize=2.5,
            capthick=1.0,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["cell_type"])
    for tick in ax.get_yticklabels():
        label = tick.get_text()
        if label in NEURAL:
            tick.set_color(NEURAL_COLOR)
            tick.set_fontweight("bold")
        elif label in SHALLOW:
            tick.set_color(SHALLOW_COLOR)

    for yi, n in zip(y, df["n_cells"]):
        ax.text(
            0.99,
            yi,
            f"n={int(n):,}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=ANNOT_SIZE,
            color="0.38",
        )

    ax.set_xlim(-0.238, -0.194)
    ax.set_xlabel(r"Mean stationary potential, $U_0$  (lower = deeper basin)")
    set_panel_title(
        ax,
        "Neural cell populations occupy deeper potential basins",
        y=1.17,
        pad=0,
    )
    ax.xaxis.grid(True, linestyle=":", linewidth=0.65, color="0.84", zorder=0)
    ax.tick_params(axis="y", length=0, labelsize=TICK_LABEL_SIZE)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NEURAL_COLOR,
            markeredgecolor="white",
            markersize=7,
            label="Neural-associated",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OTHER_COLOR,
            markeredgecolor="white",
            markersize=7,
            label="Other",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SHALLOW_COLOR,
            markeredgecolor="white",
            markersize=7,
            label="Shallow-flank examples",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0,
    )


def draw_deviation_timeline(ax: plt.Axes) -> None:
    obs = load_obs(
        CHECKPOINT,
        usecols=["annotation", "condition", "potential_deviation"],
    )
    tab = neuron_deviation_timeline(obs).sort_values("time")

    ax.axvspan(
        0.8,
        2.2,
        color=ACCENT_HI,
        alpha=0.10,
        lw=0,
        label="24h–2d window",
        zorder=0,
    )
    ax.errorbar(
        tab["time"],
        tab["mean_abs_deviation"],
        yerr=tab["se_deviation"],
        fmt="-o",
        color=NEURAL_COLOR,
        ecolor=NEURAL_COLOR,
        elinewidth=1.2,
        capsize=3,
        lw=2.2,
        ms=7,
        markerfacecolor="white",
        markeredgecolor=NEURAL_COLOR,
        markeredgewidth=1.8,
        label=r"mean $|\mathrm{potential\ deviation}|$",
        zorder=3,
    )

    layouts = {
        "Control": ((0, -10), "center", "top"),
        "SNI 6h": ((13, 0), "left", "center"),
        "SNI 24h": ((11, -9), "left", "top"),
        "SNI 2d": ((0, 13), "center", "bottom"),
        "SNI 7d": ((0, 12), "center", "bottom"),
        "SNI 14d": ((0, -13), "center", "top"),
    }
    for _, row in tab.iterrows():
        offset, ha, va = layouts[str(row["condition"])]
        ax.annotate(
            str(row["condition"]),
            xy=(row["time"], row["mean_abs_deviation"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            ha=ha,
            va=va,
            color="#374151",
            annotation_clip=False,
        )

    yspan = float(tab["mean_abs_deviation"].max() - tab["mean_abs_deviation"].min()) or 1.0
    ax.set_xlim(-0.55, 14.5)
    ax.set_ylim(
        tab["mean_abs_deviation"].min() - 0.18 * yspan,
        tab["mean_abs_deviation"].max() + 0.27 * yspan,
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"Mean $|\mathrm{potential\ deviation}|$")
    set_panel_title(ax, "Neuron homeostasis deviation over the SNI time course", y=1.17, pad=0)
    ax.legend(loc="lower right", frameon=False, fontsize=LEGEND_SIZE)
    style_axis(ax, grid_axis="y")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else DEFAULT_OUT

    apply_panel_title_rc()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 4.65),
        gridspec_kw={"width_ratios": [1.0, 1.08]},
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.76, wspace=0.28)

    draw_celltype_landscape(axes[0])
    draw_deviation_timeline(axes[1])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
