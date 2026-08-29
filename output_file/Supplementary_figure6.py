#!/usr/bin/env python3
"""Supplementary Figure 6: lung gene perturbations + dynamical claims (GSE141259).

Fully redrawn from CSV/JSON (not nested PNG composites) for a journal layout:
  Row1 a–c  ADI→AT1 exit / exit-vs-trapping / AT2→ADI entry
  Row2 d–f  per-cell exit-vs-trapping / exit-component decomposition / Krt8⁺ sink
  Row3 g–i  AT1-directed geometry (weak) / Fibro foil / process type

Biology framing (not AT1 vs Fibro lineage conversion):
  Main: leave ADI / trapping vs a weak AT1-directed geometry (AT1 is rare)
  Aux:  AT2 → Krt8+ ADI entry (the supported epithelial claim)
  Fibro/Myofibro are downstream tissue consequences, not ADI fates.
  Panels g–h: AT1 is the better foil than Fibro; do not claim a certified lineage exit.

Gene panel (U_rel–motivated):
  Candidate KO (×0): Lgals3, Cdkn1a, Spp1 — top co-varying / high-U_rel genes
  Control KO (×0): Cbr2, Hc, Chi3l1       — top anti-varying / low-U_rel genes

Default output:
  output_file/Supplementary_figure6.png

Usage:
  python output_file/Supplementary_figure6.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "output_file"))
from _supp_compose import save_fig  # noqa: E402
from dataset_pipeline import GSE141259_CELLTYPE_PALETTE  # noqa: E402

CK = ROOT / (
    "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
ME = CK / "methods_enhancement"
ARR = ME / "fig3c_refined_arrays"
CSV = ME / "Fig3C_refined_gene_readouts.csv"
PROTO = CK / "analysis_protocol_GSE141259"
P1 = ROOT / "output_file" / "robustness" / "p1_robustness"
AUDIT = ROOT / "output_file" / "mac_landscape_audit"
DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_figure6.png"

# Unified academic palette (aligned with Supplementary_figure4)
INK = "#1F2933"
MUTED = "#6B7280"
GRID = "#E9EEF3"
AT1_C = "#2E7D4F"
ADI_C = "#8B4A6B"
FIBRO_C = "#C45C26"  # used only in dynamical path-quality panel (tissue consequence)
AT2_BAR_C = "#3D7A8C"
RE_C = "#3D7A8C"
HY_C = "#9AABB3"
CLUB_C = GSE141259_CELLTYPE_PALETTE["club_cells"]
AT2_C = GSE141259_CELLTYPE_PALETTE["alv_epithelium"]
MAC_C = GSE141259_CELLTYPE_PALETTE["macrophages"]
REV_C = "#3566A0"
DIR_C = "#8B4A6B"

GENE_ORDER = ["Lgals3", "Cdkn1a", "Spp1", "Cbr2", "Hc", "Chi3l1"]
N_CANDIDATE = 3  # first block: high-U_rel candidates; second block: anti-varying controls


def _bold_title(ax, text: str, *, fontsize: float = 8.6, pad: float = 3.5) -> None:
    ax.set_title(text, fontsize=fontsize, fontweight="bold", color=INK, pad=pad, loc="center")


def _style_axis(ax, *, grid: str = "y") -> None:
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#D0D7DE")
    ax.spines["bottom"].set_color("#D0D7DE")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=7.0, length=2.4, width=0.6)
    if grid == "y":
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    elif grid == "x":
        ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def _letter(ax, letter: str, *, dx: float = -0.012, dy: float = 0.010) -> None:
    """Place panel letter in figure margin above the axes (avoids title overlap)."""
    pos = ax.get_position()
    ax.figure.text(
        pos.x0 + dx,
        pos.y1 + dy,
        str(letter).upper(),
        fontsize=12.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
        clip_on=False,
    )


def _gene_ticklabels(df: pd.DataFrame) -> list[str]:
    return [rf"$\mathit{{{r.gene}}}$" + f"\n{r.perturbation}" for r in df.itertuples()]


def _mark_group_split(ax, n_candidate: int = N_CANDIDATE) -> None:
    """Separate high-U_rel candidate KOs from anti-varying control KOs."""
    ax.axvline(n_candidate - 0.5, color="#CBD2D9", lw=0.9, ls="--", zorder=1)


def _stars(p: float) -> str:
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"


def _load_gene_tables() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = pd.read_csv(CSV)
    re = (
        df[df["mode"] == "reencode_only"]
        .set_index("gene")
        .reindex(GENE_ORDER)
        .reset_index()
    )
    # Prefer new column names; fall back to legacy aliases if present.
    if "mean_relative_AT1_vs_ADI" not in re.columns and "mean_relative_AT1_vs_Fibro" in re.columns:
        re = re.rename(columns={"mean_relative_AT1_vs_Fibro": "mean_relative_AT1_vs_ADI"})
    if "proj_mean_dz_on_exit_axis" not in re.columns and "proj_mean_dz_on_AT1_axis" in re.columns:
        re = re.rename(columns={"proj_mean_dz_on_AT1_axis": "proj_mean_dz_on_exit_axis"})
    arrays: dict[str, np.ndarray] = {}
    for gene, tag in zip(re["gene"], re["perturbation"]):
        path = ARR / f"{gene}_{tag}_reencode_only.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        z = np.load(path)
        key = "relative_AT1_vs_ADI" if "relative_AT1_vs_ADI" in z.files else "relative_closer_at1"
        arrays[gene] = z[key]
    return re, arrays


def _plot_exit_axis(ax, re: pd.DataFrame) -> None:
    x = np.arange(len(re))
    vals = re["proj_mean_dz_on_exit_axis"].to_numpy()
    colors = [AT1_C if i < N_CANDIDATE else RE_C for i in range(len(re))]
    ax.bar(x, vals, color=colors, width=0.62, edgecolor="none", zorder=2)
    ax.axhline(0, color=INK, lw=0.65, zorder=1)
    _mark_group_split(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(_gene_ticklabels(re), fontsize=5.8)
    ax.set_ylabel(r"Proj. mean $\Delta z$ on ADI→AT1", fontsize=7.5, color=INK)
    _bold_title(ax, r"ADI → AT1 exit axis")
    ax.text(
        0.98,
        0.02,
        "pos. = toward AT1 exit",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
        style="italic",
    )
    _style_axis(ax)
    _letter(ax, "A")


def _plot_relative(ax, re: pd.DataFrame) -> None:
    x = np.arange(len(re))
    vals = re["mean_relative_AT1_vs_ADI"].to_numpy()
    colors = [AT1_C if v < 0 else ADI_C for v in vals]
    ax.bar(x, vals, color=colors, width=0.62, edgecolor="none", zorder=2)
    ax.axhline(0, color=INK, lw=0.65, zorder=1)
    ymax = float(np.nanmax(np.abs(vals))) if len(vals) else 0.01
    pad = max(0.0015, 0.08 * ymax)
    y_hi = float(np.nanmax(vals)) if len(vals) else 0.01
    y_lo = float(np.nanmin(vals)) if len(vals) else -0.01
    pcol = (
        "p_relative_prefer_AT1_exit"
        if "p_relative_prefer_AT1_exit" in re.columns
        else "p_relative_prefer_AT1"
    )
    for i, (v, p) in enumerate(zip(vals, re[pcol])):
        mark = _stars(float(p))
        if mark == "ns":
            continue
        ax.text(
            i,
            v + (pad if v >= 0 else -pad),
            mark,
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=7.2,
            color=MUTED,
            fontweight="bold",
        )
    ax.set_ylim(y_lo - 3.2 * pad, y_hi + 4.5 * pad)
    _mark_group_split(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(_gene_ticklabels(re), fontsize=5.8)
    ax.set_ylabel(r"Mean $\Delta(d_{\mathrm{AT1}}-d_{\mathrm{ADI}})$", fontsize=7.5, color=INK)
    _bold_title(ax, "Exit vs trapping")
    ax.text(
        0.98,
        0.02,
        "neg. = toward AT1 exit; * $p\\leq0.05$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
        style="italic",
    )
    _style_axis(ax)
    _letter(ax, "B")


def _plot_entry_axis(ax, re: pd.DataFrame) -> None:
    x = np.arange(len(re))
    vals = re["proj_mean_dz_on_entry_axis"].to_numpy()
    colors = [AT2_BAR_C if i < N_CANDIDATE else HY_C for i in range(len(re))]
    ax.bar(x, vals, color=colors, width=0.62, edgecolor="none", zorder=2)
    ax.axhline(0, color=INK, lw=0.65, zorder=1)
    _mark_group_split(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(_gene_ticklabels(re), fontsize=5.8)
    ax.set_ylabel(r"Proj. mean $\Delta z$ on AT2→ADI", fontsize=7.5, color=INK)
    _bold_title(ax, r"AT2 → ADI entry axis")
    ax.text(
        0.98,
        0.02,
        "pos. = toward ADI entry",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
        style="italic",
    )
    _style_axis(ax)
    _letter(ax, "C")


def _plot_violin(ax, re: pd.DataFrame, arrays: dict[str, np.ndarray]) -> None:
    x = np.arange(len(re))
    data = [arrays[g] for g in re["gene"]]
    parts = ax.violinplot(data, positions=x, showmedians=True, showextrema=False, widths=0.62)
    for i, b in enumerate(parts["bodies"]):
        c = AT1_C if i < N_CANDIDATE else RE_C
        b.set_facecolor(c)
        b.set_alpha(0.22)
        b.set_edgecolor(c)
        b.set_linewidth(0.6)
    parts["cmedians"].set_color(INK)
    parts["cmedians"].set_linewidth(1.1)
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data):
        c = AT1_C if i < N_CANDIDATE else RE_C
        samp = arr if len(arr) <= 180 else rng.choice(arr, 180, replace=False)
        ax.scatter(
            i + rng.normal(0, 0.04, len(samp)),
            samp,
            s=3.0,
            c=c,
            alpha=0.26,
            linewidths=0,
            zorder=2,
        )
    ax.axhline(0, color=INK, lw=0.7, zorder=1)
    _mark_group_split(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [rf"$\mathit{{{r.gene}}}$ ({r.perturbation})" for r in re.itertuples()],
        fontsize=5.8,
    )
    ax.set_ylabel(r"Per-cell $\Delta(d_{\mathrm{AT1}}-d_{\mathrm{ADI}})$", fontsize=7.5, color=INK)
    _bold_title(ax, "Per-cell exit vs trapping")
    _style_axis(ax)
    _letter(ax, "D")


def _plot_exit_components(ax, re: pd.DataFrame) -> None:
    """Decompose panel-b's relative shift into its two absolute distances."""
    x = np.arange(len(re))
    w = 0.36
    d_adi = re["mean_delta_dist_ADI"].to_numpy()
    d_at1 = re["mean_delta_dist_AT1"].to_numpy()
    ax.bar(
        x - w / 2,
        d_adi,
        w,
        color=ADI_C,
        edgecolor="white",
        label=r"$\Delta d_{\mathrm{ADI}}$",
        zorder=2,
    )
    ax.bar(
        x + w / 2,
        d_at1,
        w,
        color=AT1_C,
        edgecolor="white",
        label=r"$\Delta d_{\mathrm{AT1}}$",
        zorder=2,
    )
    ax.axhline(0, color=INK, lw=0.65, zorder=1)
    _mark_group_split(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\mathit{{{g}}}$" for g in re["gene"]], fontsize=6.0)
    ax.set_ylabel("Mean change in latent distance", fontsize=7.5, color=INK)
    _bold_title(ax, "Exit mechanism: leave ADI vs approach AT1")
    ax.legend(fontsize=5.8, frameon=False, loc="upper right", handlelength=1.0)
    ax.text(
        0.02,
        0.03,
        r"$+\Delta d_{\mathrm{ADI}}$: leave ADI; $-\Delta d_{\mathrm{AT1}}$: approach AT1",
        transform=ax.transAxes,
        fontsize=5.6,
        color=MUTED,
        va="bottom",
    )
    _style_axis(ax)
    _letter(ax, "E")


def _plot_sink(ax) -> None:
    claim = json.loads((PROTO / "fig3a_model_claim_summary.json").read_text())
    sink = claim["protocol_metrics"]
    labels = ["Club", "AT2"]
    sinks = [sink["club_sink"], sink["at2_sink"]]
    inflows = [sink["club_inflow"], sink["at2_inflow"]]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(
        x - w / 2,
        sinks,
        w,
        color=[CLUB_C, AT2_C],
        edgecolor="white",
        label="Sink strength",
        zorder=2,
    )
    b2 = ax.bar(
        x + w / 2,
        inflows,
        w,
        color=["#c5cfa0", "#b7d7e6"],
        edgecolor="white",
        label="Inflow fraction",
        zorder=2,
    )
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.018,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Vector-field metric", fontsize=7.5, color=INK)
    ax.set_ylim(0, max(sinks + inflows) * 1.28)
    _bold_title(ax, r"Flow into high-$U$ $Krt8^+$ basin")
    ax.legend(fontsize=6.0, frameon=False, loc="upper right", handlelength=1.0)
    _style_axis(ax)
    _letter(ax, "F")


def _plot_path_quality(ax) -> None:
    """AT1-directed geometry is the better epithelial foil; AT1 n is small."""
    at1 = json.loads((PROTO / "fate_Krt8_to_AT1_summary.json").read_text())
    metrics = ["Reliability\n(higher better)", r"$\Omega$ uncertainty" + "\n(lower better)"]
    vals = [
        float(at1["omega"]["path_reliability"]),
        float(at1["omega"]["path_uncertainty_Omega"]),
    ]
    x = np.arange(len(metrics))
    bars = ax.bar(x, vals, color=AT1_C, edgecolor="white", width=0.52, zorder=2)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.025,
            f"{v:.2f}",
            ha="center",
            fontsize=7.0,
            color=INK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=6.6)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Path score", fontsize=7.5, color=INK)
    _bold_title(ax, r"$Krt8^+$→AT1 geometry (weak exit hypothesis)")
    ax.text(
        0.02,
        0.97,
        "AT1 rare; not a certified lineage exit",
        transform=ax.transAxes,
        fontsize=5.8,
        color=MUTED,
        va="top",
        style="italic",
    )
    _style_axis(ax)
    _letter(ax, "G")


def _plot_mismatch(ax) -> None:
    """Reject Krt8→Fibro as a non-lineage pseudo-path (tissue-consequence foil)."""
    decomp = pd.read_csv(P1 / "lung_action_decomposition.csv").set_index("branch")
    at1_mm = abs(float(decomp.loc["AT1", "frac_mismatch"]))
    fibro_mm = abs(float(decomp.loc["Fibro", "frac_mismatch"]))
    fibro_deg = bool(decomp.loc["Fibro", "path_degenerate"])
    labs = [
        "AT1-directed\n" + r"$Krt8$→AT1",
        "Tissue foil\n" + r"$Krt8$→Fibro" + "\n(not a fate)",
    ]
    mms = [at1_mm, fibro_mm]
    bars = ax.bar(labs, mms, color=[AT1_C, FIBRO_C], edgecolor="white", width=0.55, zorder=2)
    for bar, v in zip(bars, mms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.03,
            f"{v:.2f}",
            ha="center",
            fontsize=7.0,
            color=INK,
        )
    ax.set_ylabel("|Mismatch| / |action|", fontsize=7.5, color=INK)
    ax.set_ylim(0, max(1.25, max(mms) * 1.28))
    _bold_title(ax, "Reject Fibro as lineage endpoint")
    note = "Fibro = tissue consequence foil"
    if fibro_deg:
        note += "\n(mismatch-dominated, path_degenerate)"
    else:
        note += "\n(mismatch-dominated)"
    ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=5.8, color=MUTED, va="top")
    _style_axis(ax)
    _letter(ax, "H")


def _plot_process(ax) -> None:
    df = pd.read_csv(AUDIT / "02_parent_trajectory_classification.csv")
    ax.axvspan(0.55, 1.05, color="#EEF3F8", zorder=0, alpha=0.95)
    ax.axhline(1.7, color=DIR_C, ls="--", lw=0.9, alpha=0.75, zorder=1)
    ax.axhline(1.4, color=REV_C, ls="--", lw=0.9, alpha=0.75, zorder=1)
    ax.axvline(0.55, color="#888", ls=":", lw=0.8, zorder=1)

    mac = df[df["parent"] == "macrophages"].iloc[0]
    alv = df[df["parent"] == "alv_epithelium"].iloc[0]
    entries = [
        (mac, MAC_C, "Mac\nAM(PBS)→AM(Bleo)", (-22, 18), "center", "bottom"),
        (alv, AT2_C, "Alv\nAT2→Krt8 ADI", (-102, 10), "right", "center"),
    ]
    for row, color, short, offset, ha, va in entries:
        x = float(row["non_conservative_flux_fraction"])
        y = float(row["action_asymmetry_ratio"])
        ax.scatter([x], [y], s=150, c=color, edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(
            short,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va=va,
            fontsize=6.6,
            color=INK,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.7, shrinkA=2, shrinkB=3),
        )

    ax.set_xlim(0.40, 1.02)
    ax.set_ylim(0.85, 2.05)
    ax.set_xlabel("Non-conservative flux fraction", fontsize=7.5, color=INK)
    ax.set_ylabel(r"Action asymmetry $|S_{\mathrm{bwd}}|/|S_{\mathrm{fwd}}|$", fontsize=7.5, color=INK)
    _bold_title(ax, "Geodesic symmetry vs temporal direction")
    legend = [
        Line2D([0], [0], color=REV_C, ls="--", label="Reversible threshold (1.4)"),
        Line2D([0], [0], color=DIR_C, ls="--", label="Directional threshold (1.7)"),
        Line2D([0], [0], color="#888", ls=":", label=r"Flux $\geq$ 0.55"),
    ]
    ax.legend(handles=legend, fontsize=5.8, frameon=False, loc="upper left")
    ax.text(
        0.03,
        0.03,
        "Mac: reversible state coupling (geometry)\n"
        "Alv: symmetric geodesic; temporal AT2→ADI",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=MUTED,
    )
    _style_axis(ax)
    _letter(ax, "I")


def _ensure_gene_tables() -> None:
    if CSV.is_file() and ARR.is_dir():
        return
    print("[supp fig6] computing refined gene readouts on adopted GSE141259 checkpoint...", flush=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from refine_fig3c_gene_readouts import main as _refine

    _refine()


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    _ensure_gene_tables()
    re, arrays = _load_gene_tables()

    fig = plt.figure(figsize=(13.0, 11.2), facecolor="white")
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.0, 1.05, 1.08],
        hspace=0.42,
        wspace=0.32,
        left=0.065,
        right=0.985,
        top=0.93,
        bottom=0.05,
    )

    # Row 1: a b c — exit / trapping / entry (epithelial lineage)
    _plot_exit_axis(fig.add_subplot(gs[0, 0]), re)
    _plot_relative(fig.add_subplot(gs[0, 1]), re)
    _plot_entry_axis(fig.add_subplot(gs[0, 2]), re)

    # Row 2: d e f
    _plot_violin(fig.add_subplot(gs[1, 0]), re, arrays)
    _plot_exit_components(fig.add_subplot(gs[1, 1]), re)
    _plot_sink(fig.add_subplot(gs[1, 2]))

    # Row 3: g h i
    _plot_path_quality(fig.add_subplot(gs[2, 0]))
    _plot_mismatch(fig.add_subplot(gs[2, 1]))
    _plot_process(fig.add_subplot(gs[2, 2]))
    return save_fig(fig, out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    compose(out=Path(argv[0]) if argv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
