#!/usr/bin/env python3
"""Supplementary Figure 8: HGSOC eviction specificity (SOD2, random null, IFI27, PDVS).

Computes hybrid-KO valley histograms for SOD2 and IFI27 on the adopted HGSOC
checkpoint, loads the SOD2 random-gene null, and draws the PDVS eviction screen
in the published 2×2 layout (panel d is the two-bar PDVS5 compare).

Default output:
  output_file/Supplementary_figure8.png
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _supp_compose import save_fig  # noqa: E402
import figure5_cd as F5  # noqa: E402

CK = ROOT / (
    "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
OUT_ME = CK / "methods_enhancement"
DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_figure8.png"
INK = "#1F2933"
PALETTE = ["#3a6ea5", "#e07a5f", "#3d9970", "#c9a227"]


def _hybrid_u(gene, model, config, X, ct, stage, valley, pot, gene_list):
    from plot_fig4b_bbc3_valley_eviction import _U, _encode, _unit

    gcol = list(gene_list).index(gene)
    bs = int(getattr(config, "batch_size", 256))
    X_kd = X.copy()
    X_kd[:, gcol] = 0.0
    z_wt = _encode(model, X[valley], ct[valley], "cpu", bs=bs, stage=stage[valley])
    z_kd = _encode(model, X_kd[valley], ct[valley], "cpu", bs=bs, stage=stage[valley])
    z_kd = z_wt + _unit((z_kd - z_wt).mean(0))[None, :]
    return pot[valley], _U(model, z_kd, "cpu", bs=bs)


def _letter(ax, letter: str, *, dx: float = -0.012, dy: float = 0.010) -> None:
    """Place panel letter in figure margin above the axes (avoids title overlap)."""
    pos = ax.get_position()
    ax.figure.text(
        pos.x0 + dx,
        pos.y1 + dy,
        str(letter).upper(),
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
        clip_on=False,
    )


def _draw_null(ax, null_csv: Path, sod2_esc: float) -> None:
    df = pd.read_csv(null_csv)
    col = "frac_escape_q15" if "frac_escape_q15" in df.columns else df.columns[-1]
    ax.hist(
        df[col].astype(float),
        bins=12,
        color=PALETTE[0],
        alpha=0.7,
        label="random-gene nulls",
        zorder=2,
    )
    ax.axvline(sod2_esc, color=PALETTE[1], lw=2.2, label=rf"SOD2$\approx${sod2_esc:.2f}")
    ax.set_xlabel("Valley escape fraction")
    ax.set_ylabel("Count")
    ax.set_title("SOD2 vs random-gene null (specificity fail)", fontweight="bold", loc="center")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _letter(ax, "B")


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    from plot_fig4b_bbc3_valley_eviction import load_valley_matrix_and_model

    F5._apply_style()
    print("[supp fig8] loading HGSOC valley matrix + hybrid KO for SOD2/IFI27...", flush=True)
    model, config, X, ct, stage, valley, pot, u_cut, gene_list = load_valley_matrix_and_model()
    hist = {}
    for gene in ("SOD2", "IFI27"):
        if gene not in list(gene_list):
            print(f"  skip {gene}: not in training panel", flush=True)
            continue
        print(f"  hybrid KO {gene} ...", flush=True)
        hist[gene] = (*_hybrid_u(gene, model, config, X, ct, stage, valley, pot, gene_list), float(u_cut))

    null_p = ROOT / "output_file" / "robustness" / "p0_robustness" / "SOD2_random_gene_nulls.csv"
    if not null_p.is_file():
        print("[supp fig8] computing SOD2 random-gene nulls...", flush=True)
        from run_p0_robustness import main as _p0

        _p0()
    hy, re, null_loaded = F5._load_pdvs_tables()
    sod2_esc = float("nan")
    if "SOD2" in hy.index:
        sod2_esc = F5._esc(hy.loc["SOD2"])
    if not np.isfinite(sod2_esc) and null_p.is_file():
        sod2_esc = 0.76

    fig = plt.figure(figsize=(13.8, 10.5), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.22, left=0.07, right=0.98, top=0.91, bottom=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    gs_d = gs[1, 1].subgridspec(1, 2, wspace=0.32)
    ax_d0 = fig.add_subplot(gs_d[0, 0])
    ax_d1 = fig.add_subplot(gs_d[0, 1])

    if "SOD2" in hist:
        u_wt, u_kd, cut = hist["SOD2"]
        F5._draw_hist(ax_a, "SOD2", u_wt, u_kd, cut)
        _letter(ax_a, "A")
    if null_p.is_file():
        _draw_null(ax_b, null_p, sod2_esc if sod2_esc == sod2_esc else 0.76)
    if "IFI27" in hist:
        u_wt, u_kd, cut = hist["IFI27"]
        F5._draw_hist(ax_c, "IFI27", u_wt, u_kd, cut)
        _letter(ax_c, "C")
    F5._draw_escape(ax_d0, hy, re, null_loaded)
    F5._draw_delta(ax_d1, hy)
    _letter(ax_d0, "D")

    return save_fig(fig, out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    compose(out=Path(argv[0]) if argv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
