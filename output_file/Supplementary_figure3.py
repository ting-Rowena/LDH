#!/usr/bin/env python3
"""Supplementary Figure 3: training Train PCC and Train MSE curves.

One row, three panels from adopted checkpoints' ``Loss_epoch.csv``:
  (a) GSE155622
  (b) GSE141259
  (c) HGSOC

Default output:
  output_file/Supplementary_figure3.png

Usage:
  python output_file/Supplementary_figure3.py
  python output_file/Supplementary_figure3.py /path/to/out.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]

CK_PAIN = ROOT / (
    "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
CK_LUNG = ROOT / (
    "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
CK_HG = ROOT / (
    "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)

DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_figure3.png"

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e8e8e8"
C_PCC = "#1f4e79"
C_MSE = "#c45911"

PANELS = [
    ("A", "GSE155622", CK_PAIN),
    ("B", "GSE141259", CK_LUNG),
    ("C", "HGSOC", CK_HG),
]


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "center",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
        }
    )


def _panel_letter(ax, letter: str, *, x: float = -0.12, y: float = 1.08) -> None:
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


def _style_axis(ax) -> None:
    ax.tick_params(colors=INK, length=3.2, width=0.7)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _plot_train_metrics(ax, ck: Path, *, title: str, letter: str) -> None:
    csv_path = ck / "Loss_epoch.csv"
    if not csv_path.is_file():
        ax.text(0.5, 0.5, f"missing\n{csv_path.name}", ha="center", va="center", color=MUTED)
        ax.set_axis_off()
        ax.set_title(title, loc="center", fontsize=11, fontweight="bold", color=INK, pad=6)
        _panel_letter(ax, letter)
        return

    df = pd.read_csv(csv_path, usecols=["Epoch", "TrainPCC", "TrainMSE"])
    ep = df["Epoch"].to_numpy()
    pcc = df["TrainPCC"].to_numpy(dtype=float)
    mse = df["TrainMSE"].to_numpy(dtype=float)

    ax.plot(ep, pcc, color=C_PCC, lw=1.35, label="Train PCC", zorder=3)
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("Train PCC", color=C_PCC)
    ax.tick_params(axis="y", colors=C_PCC)
    ax.spines["left"].set_color(C_PCC)
    _style_axis(ax)
    ax.spines["left"].set_color(C_PCC)

    ax2 = ax.twinx()
    ax2.plot(ep, mse, color=C_MSE, lw=1.35, label="Train MSE", zorder=3)
    ax2.set_ylabel("Train MSE", color=C_MSE)
    ax2.tick_params(axis="y", colors=C_MSE, length=3.2, width=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["right"].set_color(C_MSE)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.set_yscale("log")

    ax.set_title(title, loc="center", fontsize=11, fontweight="bold", color=INK, pad=6)
    _panel_letter(ax, letter)


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    _apply_style()

    fig = plt.figure(figsize=(14.2, 4.2), facecolor="white")
    gs = GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.22],
        wspace=0.42,
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.16,
    )

    for i, (letter, title, ck) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        _plot_train_metrics(ax, ck, title=title, letter=letter)

    # Shared legend on the far right
    ax_leg = fig.add_subplot(gs[0, 3])
    ax_leg.set_axis_off()
    handles = [
        Line2D([0], [0], color=C_PCC, lw=1.8, label="Train PCC"),
        Line2D([0], [0], color=C_MSE, lw=1.8, label="Train MSE"),
    ]
    ax_leg.legend(
        handles=handles,
        loc="center left",
        fontsize=9,
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.0,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(out, dpi=300, facecolor="white")
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
