#!/usr/bin/env python3
"""Figure 5: HGSOC paracrine CCC (deep-valley EOC ↔ high-U stromal) + PDVS Kaplan–Meier OS.

Computes ligand–receptor scores on the adopted HGSOC checkpoint and draws
overall survival for high vs low PDVS on TCGA-OV.

Default output:
  output_file/figure5_ef.png

Usage:
  python output_file/figure5_ef.py
  python output_file/figure5_ef.py /path/to/out.png
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import CACHE, CK_HG, hgsoc_ccc  # noqa: E402
from _supp_compose import compose_side_by_side  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "figure5_ef.png"
CCC_PATH = CACHE / "figure5_ef_ccc.png"
KM_PATH = CACHE / "figure5_ef_km.png"
PUBLISHED_SIZE = (3997, 1098)
PUBLISHED_GAP = 36

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e3e8ee"
S2E = "#3d9970"
E2S = "#9c3d2e"
PALETTE = ["#3a6ea5", "#e07a5f"]

LR_CLASS = {
    ("COL1A1", "ITGB1"): "ECM",
    ("FN1", "ITGB1"): "ECM",
    ("FN1", "ITGAV"): "ECM",
    ("IL6", "IL6ST"): "cyto",
    ("LIF", "IL6ST"): "cyto",
    ("IL6", "IL6R"): "cyto",
    ("CXCL12", "CXCR4"): "cyto",
    ("HBEGF", "EGFR"): "GF",
    ("GAS6", "AXL"): "GF",
    ("TNF", "TNFRSF1A"): "cyto",
    ("IL1B", "IL1R1"): "cyto",
    ("AREG", "EGFR"): "GF",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "center",
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.axisbelow": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _style_axis(ax, *, grid_axis: str = "both") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, length=3.5, width=0.9, direction="out")
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, lw=0.7)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def _pair_label(lig: str, rec: str) -> str:
    cls = LR_CLASS.get((str(lig).upper(), str(rec).upper()), "")
    core = f"{lig}-{rec}"
    return f"{core} ({cls})" if cls else core


def _draw_ccc(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    s2e = tables.get("Stromal_to_EOC", pd.DataFrame())
    e2s = tables.get("EOC_to_Stromal", pd.DataFrame())
    fig = plt.figure(figsize=(10.8, 4.2), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, wspace=0.55, left=0.22, right=0.97, top=0.78, bottom=0.14)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    def _bars(ax, df: pd.DataFrame, color: str, title: str) -> None:
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No LR pairs scored", ha="center", va="center")
            ax.axis("off")
            return
        top = df.sort_values("lr_score", ascending=False).head(8)
        labels = [_pair_label(r.ligand, r.receptor) for r in top.itertuples()]
        y = np.arange(len(labels))
        vals = top["lr_score"].to_numpy(float)
        ax.barh(y, vals, color=color, edgecolor="white", linewidth=0.4, zorder=3, height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(r"LR score ($\bar{L} \times \bar{R}$)")
        ax.set_title(title, fontweight="bold", loc="center", pad=6)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f"  {v:.2f}", va="center", ha="left", fontsize=7.5, color=INK)
        xmax = float(np.nanmax(vals)) if len(vals) else 1.0
        ax.set_xlim(0, xmax * 1.18)
        _style_axis(ax, grid_axis="both")

    _bars(ax_l, s2e, S2E, r"Stromal high-$U$ $\rightarrow$ deep-valley EOC")
    _bars(ax_r, e2s, E2S, r"Deep-valley EOC $\rightarrow$ stromal high-$U$")
    fig.suptitle(
        r"Deep-valley EOC $\leftrightarrow$ high-$U$ stromal LR scores",
        fontsize=11,
        fontweight="bold",
        y=0.98,
        color=INK,
    )
    fig.legend(
        handles=[
            Patch(facecolor=S2E, label=r"Stromal $\rightarrow$ EOC"),
            Patch(facecolor=E2S, label=r"EOC $\rightarrow$ Stromal"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    return out


def _load_tcga_patient_table() -> pd.DataFrame:
    path = CK_HG / "methods_enhancement" / "pdvs_TCGA_OV_patient_table.csv"
    if not path.is_file():
        from run_clinical_pdvs_validation import run_clinical_pdvs_validation

        print("[figure5_ef] running PDVS clinical validation (TCGA-OV KM)...", flush=True)
        run_clinical_pdvs_validation(CK_HG)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _draw_km(out: Path) -> Path:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from matplotlib.lines import Line2D

    df = _load_tcga_patient_table()
    times = df["os_months"].to_numpy(float)
    events = df["os_event"].to_numpy(int)
    groups = df["PDVS_group"].astype(str).to_numpy()
    m0, m1 = groups == "high_PDVS", groups == "low_PDVS"
    pval = float(logrank_test(times[m0], times[m1], events[m0], events[m1]).p_value)
    n = int(len(df))
    n_ev = int(events.sum())
    hr_hi = hr_lo = hr = float("nan")
    summary_path = CK_HG / "methods_enhancement" / "pdvs_clinical_summary.csv"
    if summary_path.is_file():
        sm = pd.read_csv(summary_path)
        row = sm.loc[sm["cohort"].astype(str) == "TCGA_OV"]
        if len(row):
            r = row.iloc[0]
            hr = float(r["uni_PDVS_high_hr"])
            hr_lo = float(r["uni_PDVS_high_hr_lo"])
            hr_hi = float(r["uni_PDVS_high_hr_hi"])

    # Display window matches the risk table (clinical OS convention).
    xmax = 60.0
    months = np.array([0.0, 12.0, 24.0, 36.0, 60.0])
    series = (
        ("high_PDVS", PALETTE[0], "High PDVS"),
        ("low_PDVS", PALETTE[1], "Low PDVS"),
    )

    fig = plt.figure(figsize=(4.8, 4.3), facecolor="white")
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[3.8, 0.95],
        hspace=0.02,
        left=0.20,
        right=0.97,
        top=0.90,
        bottom=0.12,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0], sharex=ax)

    for lab, color, pretty in series:
        m = groups == lab
        if int(m.sum()) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(times[m], events[m], label=pretty)
        kmf.plot_survival_function(
            ax=ax,
            color=color,
            lw=2.4,
            ci_show=True,
            ci_alpha=0.14,
            show_censors=True,
            censor_styles={"ms": 2.8, "marker": "|", "mew": 0.8},
        )

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(months)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Overall survival", fontsize=9)
    ax.set_xlabel("")
    ax.set_title("TCGA-OV · PDVS", fontweight="bold", loc="center", fontsize=10, pad=6)
    handles = [
        Line2D([0], [0], color=PALETTE[0], lw=2.4, label="High PDVS"),
        Line2D([0], [0], color=PALETTE[1], lw=2.4, label="Low PDVS"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        loc="upper right",
        handlelength=1.8,
        borderaxespad=0.15,
    )
    _style_axis(ax, grid_axis="y")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False, length=0)

    p_str = f"{pval:.3f}" if pval >= 0.001 else f"{pval:.2e}"
    stats = [f"n = {n:,}  ·  {n_ev:,} deaths", f"Log-rank P = {p_str}"]
    if np.isfinite(hr):
        stats.append(f"HR {hr:.2f} ({hr_lo:.2f}–{hr_hi:.2f})")
    ax.text(
        0.03,
        0.05,
        "\n".join(stats),
        transform=ax.transAxes,
        fontsize=7.5,
        color=INK,
        va="bottom",
        ha="left",
        linespacing=1.4,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#d5dde5",
            "linewidth": 0.7,
            "alpha": 0.95,
        },
    )

    # Risk table: Low/High as y-tick labels so they sit flush against the axis.
    risk_order = (("low_PDVS", PALETTE[1], "Low"), ("high_PDVS", PALETTE[0], "High"))
    ax_r.set_xlim(0, xmax)
    ys = list(range(len(risk_order) - 1, -1, -1))  # Low on top
    ax_r.set_ylim(-0.55, len(risk_order) - 0.35)
    ax_r.set_xticks(months)
    ax_r.set_xticklabels([str(int(t)) for t in months], fontsize=8)
    ax_r.set_xlabel("Months", fontsize=9, labelpad=1)
    ax_r.set_yticks(ys)
    ax_r.set_yticklabels([pretty for _, _, pretty in risk_order], fontsize=8, fontweight="bold")
    for tick, (_, color, _) in zip(ax_r.get_yticklabels(), risk_order):
        tick.set_color(color)
    ax_r.tick_params(axis="x", length=3.2, width=0.9, colors=INK)
    ax_r.tick_params(axis="y", length=0, pad=4)
    for side in ("top", "right", "left"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["bottom"].set_color(MUTED)
    ax_r.set_facecolor("white")
    ax_r.set_ylabel("At risk", fontsize=9, color=INK)
    for y, (lab, _color, _pretty) in zip(ys, risk_order):
        m = groups == lab
        for t in months:
            n_at = int((times[m] >= t).sum())
            if t == 0:
                ax_r.text(0.8, y, str(n_at), ha="left", va="center", fontsize=8.2, color=INK)
            else:
                ax_r.text(t, y, str(n_at), ha="center", va="center", fontsize=8.2, color=INK)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    return out


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    _apply_style()
    tables = hgsoc_ccc()
    print("[figure5_ef] drawing CCC + KM ...", flush=True)
    _draw_ccc(tables, CCC_PATH)
    _draw_km(KM_PATH)
    return compose_side_by_side(
        CCC_PATH,
        KM_PATH,
        out,
        target_size=PUBLISHED_SIZE,
        gap=PUBLISHED_GAP,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    compose(out=Path(argv[0]) if argv else DEFAULT_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
