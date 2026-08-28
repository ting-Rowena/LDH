#!/usr/bin/env python3
"""Figure 5: BBC3/SOD2 hybrid-KO histograms + PDVS5 escape/ΔU (2×2).

Self-contained entrypoint consolidated from:
  - scripts/compose_fig4b_bbc3_sod2_pdvs5.py
  - scripts/plot_fig4b_bbc3_valley_eviction.py (model load on --rebuild)

Default output:
  output_file/figure5_cd.png
  (= archived panel Fig4B_BBC3_SOD2_PDVS5_combined.png; manuscript Figure 5)

Usage:
  python output_file/figure5_cd.py              # recompute hybrid KO + redraw
  python output_file/figure5_cd.py /path/to/out.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CK = ROOT / (
    "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
OUT_ME = CK / "methods_enhancement"
DEFAULT_OUT = Path(__file__).resolve().parent / "figure5_cd.png"

PDVS = ["BBC3", "SOD2", "WFDC2", "FTL", "CEBPD"]
INK, UP, DOWN, MUTED, GREY = "#111111", "#9c3d2e", "#2f5f8a", "#555555", "#9a9a9a"
PALETTE = [
    "#3a6ea5",
    "#e07a5f",
    "#3d9970",
    "#c9a227",
    "#8367c7",
    "#d1495b",
    "#2a9d8f",
    "#6d597a",
]
PANEL_TITLE_SIZE = 10

for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "savefig.bbox": None,
            "savefig.pad_inches": 0,
        }
    )


def _set_panel_title(ax, title: str, *, pad: float = 10) -> None:
    ax.set_title(title, loc="center", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=pad)


def _style_axis(ax, *, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#1f2933", length=3.5, width=0.9)
    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, color="#e3e8ee", linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)


def _esc(row: pd.Series) -> float:
    for k in ("frac_escape_published_cutoff", "frac_escape_q15", "frac_escape_valley", "frac_escape_claim"):
        if k in row.index and pd.notna(row[k]):
            return float(row[k])
    return float("nan")


def _ensure_pdvs_summary() -> Path:
    summary = OUT_ME / "PDVS5_valley_eviction_summary.csv"
    if summary.is_file():
        return summary
    alt = OUT_ME / "PDVS7_valley_eviction_summary.csv"
    if alt.is_file():
        return alt
    print("[figure5_cd] computing PDVS valley-eviction summary...", flush=True)
    from analyze_pdvs7_valley_eviction import main as _pdvs_main

    _pdvs_main()
    if summary.is_file():
        return summary
    if alt.is_file():
        return alt
    raise FileNotFoundError("PDVS valley-eviction summary was not written")


def _load_pdvs_tables() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    summary = _ensure_pdvs_summary()
    res = pd.read_csv(summary)
    sub = res[res["mode"].isin(["reencode_only", "hybrid", "hybrid_published"])].copy()
    hy = sub[sub["mode"] == "hybrid"].set_index("gene")
    pub = sub[sub["mode"] == "hybrid_published"].set_index("gene")
    for g in PDVS:
        if g not in hy.index and g in pub.index:
            hy.loc[g] = pub.loc[g]
    re = sub[sub["mode"] == "reencode_only"].set_index("gene")
    null_p = ROOT / "output_file" / "robustness" / "p0_robustness" / "SOD2_random_gene_nulls.csv"
    if not null_p.is_file():
        print("[figure5_cd] computing SOD2 random-gene nulls via P0 robustness...", flush=True)
        from run_p0_robustness import main as _p0

        _p0()
    return hy, re, null_p


def _set_bold_italic_gene_title(ax, gene: str) -> None:
    fp_plain = FontProperties(family="Liberation Sans", size=PANEL_TITLE_SIZE, weight="bold")
    fp_gene = FontProperties(
        family="Liberation Sans", size=PANEL_TITLE_SIZE, weight="bold", style="italic"
    )
    box = HPacker(
        children=[
            TextArea("HGSOC deep-valley EOC: ", textprops=dict(fontproperties=fp_plain, color="k")),
            TextArea(gene, textprops=dict(fontproperties=fp_gene, color="k")),
            TextArea("-KO", textprops=dict(fontproperties=fp_plain, color="k")),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    ax.add_artist(
        AnchoredOffsetbox(
            loc="lower center",
            child=box,
            pad=0,
            frameon=False,
            bbox_to_anchor=(0.5, 1.03),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
    )


def _draw_hist(
    ax,
    gene: str,
    u_wt: np.ndarray,
    u_kd: np.ndarray,
    u_cut: float,
    *,
    xlim: tuple[float, float] | None = None,
) -> None:
    ax.hist(u_wt, bins=30, alpha=0.55, color=PALETTE[0], label=r"WT valley $U_0$", density=True)
    ax.hist(
        u_kd,
        bins=30,
        alpha=0.55,
        color=PALETTE[5],
        label=rf"$\mathit{{{gene}}}$-KO $U_0$",
        density=True,
    )
    ax.axvline(u_cut, color="k", ls="--", lw=1.2, label=f"valley cutoff={u_cut:.3g}")
    ax.set_xlabel(r"Stationary potential $U_0$")
    ax.set_ylabel("Density")
    if xlim is not None:
        ax.set_xlim(*xlim)
    _set_bold_italic_gene_title(ax, gene)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style_axis(ax, grid_axis="y")


def _style_bar_ax(ax) -> None:
    ax.yaxis.grid(True, color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_escape(ax, hy: pd.DataFrame, re: pd.DataFrame, null_p: Path) -> None:
    x = np.arange(len(PDVS))
    w = 0.38
    in_re = [i for i, g in enumerate(PDVS) if g in re.index]
    in_hy = [i for i, g in enumerate(PDVS) if g in hy.index]
    y_re = np.array([_esc(re.loc[PDVS[i]]) for i in in_re])
    y_hy = np.array([_esc(hy.loc[PDVS[i]]) for i in in_hy])
    if in_re:
        ax.bar(np.asarray(in_re, float) - w / 2, y_re, w, color="#6a8fa0", label="reencode only", edgecolor="none")
    if in_hy:
        ax.bar(np.asarray(in_hy, float) + w / 2, y_hy, w, color=UP, label="hybrid", edgecolor="none")
    ax.axhline(0.788, color=MUTED, ls=":", lw=0.9, label="SOD2 protocol 78.8%")
    if null_p.is_file():
        med = float(pd.read_csv(null_p)["frac_escape_q15"].median())
        ax.axhline(med, color=GREY, ls="--", lw=0.8, label=f"random-null median={med:.2f}")
    ax.set_xticks(x, [rf"$\mathit{{{g}}}$-KO" for g in PDVS], rotation=25, ha="right")
    ax.set_xlim(-0.6, len(PDVS) - 0.4)
    ax.set_ylabel("Escape fraction")
    ax.set_ylim(0, 1.05)
    _set_panel_title(ax, "PDVS genes: valley escape", pad=10)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    _style_bar_ax(ax)


def _draw_delta(ax, hy: pd.DataFrame) -> None:
    colors = [
        UP if (g in hy.index and float(hy.loc[g, "mean_delta_U"]) > 0) else (DOWN if g in hy.index else GREY)
        for g in PDVS
    ]
    vals = [float(hy.loc[g, "mean_delta_U"]) if g in hy.index else 0.0 for g in PDVS]
    ax.bar(np.arange(len(PDVS)), vals, color=colors, edgecolor="none", width=0.72)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(np.arange(len(PDVS)), [rf"$\mathit{{{g}}}$-KO" for g in PDVS], rotation=25, ha="right")
    ax.set_ylabel(r"Mean $\Delta U_0$")
    _set_panel_title(ax, "Potential shift (HVG-restricted PDVS)", pad=10)
    _style_bar_ax(ax)


def _hybrid_u_for_gene(gene, model, config, X, ct, stage, valley, pot, gene_list):
    from plot_fig4b_bbc3_valley_eviction import _U, _encode, _unit

    gcol = list(gene_list).index(gene)
    bs = int(getattr(config, "batch_size", 256))
    X_kd = X.copy()
    X_kd[:, gcol] = 0.0
    z_wt = _encode(model, X[valley], ct[valley], "cpu", bs=bs, stage=stage[valley])
    z_kd = _encode(model, X_kd[valley], ct[valley], "cpu", bs=bs, stage=stage[valley])
    z_kd = z_wt + _unit((z_kd - z_wt).mean(0))[None, :]
    u_wt = pot[valley]
    u_kd = _U(model, z_kd, "cpu", bs=bs)
    return u_wt, u_kd


def compose_four(
    hist_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    *,
    out: Path,
) -> Path:
    """Draw BBC3 | SOD2 / escape | ΔU on one 2×2 GridSpec."""
    _apply_style()
    hy, re, null_p = _load_pdvs_tables()

    all_u = np.concatenate(
        [hist_data["BBC3"][0], hist_data["BBC3"][1], hist_data["SOD2"][0], hist_data["SOD2"][1]]
    )
    lo, hi = float(np.nanpercentile(all_u, 0.5)), float(np.nanpercentile(all_u, 99.5))
    pad = 0.05 * (hi - lo + 1e-12)
    xlim = (lo - pad, hi + pad)

    fig = plt.figure(figsize=(11.0, 8.2), facecolor="white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        left=0.10,
        right=0.98,
        top=0.93,
        bottom=0.09,
        wspace=0.18,
        hspace=0.40,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
    )
    ax_bbc3 = fig.add_subplot(gs[0, 0])
    ax_sod2 = fig.add_subplot(gs[0, 1])
    ax_esc = fig.add_subplot(gs[1, 0])
    ax_du = fig.add_subplot(gs[1, 1])

    u_wt, u_kd, u_cut = hist_data["BBC3"]
    _draw_hist(ax_bbc3, "BBC3", u_wt, u_kd, u_cut, xlim=xlim)
    u_wt, u_kd, u_cut = hist_data["SOD2"]
    _draw_hist(ax_sod2, "SOD2", u_wt, u_kd, u_cut, xlim=xlim)
    _draw_escape(ax_esc, hy, re, null_p)
    _draw_delta(ax_du, hy)

    positions = [ax.get_position() for ax in (ax_bbc3, ax_sod2, ax_esc, ax_du)]
    for ax, pos in zip((ax_bbc3, ax_sod2, ax_esc, ax_du), positions):
        ax.set_position(pos)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def _rebuild(out: Path) -> Path:
    from plot_fig4b_bbc3_valley_eviction import load_valley_matrix_and_model

    _apply_style()
    model, config, X, ct, stage, valley, pot, u_cut, gene_list = load_valley_matrix_and_model()
    hist_data = {}
    for gene in ("BBC3", "SOD2"):
        print(f"  hybrid KO {gene} ...", flush=True)
        u_wt, u_kd = _hybrid_u_for_gene(gene, model, config, X, ct, stage, valley, pot, gene_list)
        hist_data[gene] = (u_wt, u_kd, float(u_cut))
    return compose_four(hist_data, out=out)


def compose(*, out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    return _rebuild(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default=None, help="Output PNG path")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    compose(out=Path(args.out) if args.out else DEFAULT_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
