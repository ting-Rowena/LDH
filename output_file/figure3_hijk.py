#!/usr/bin/env python3
"""Figure 3: partner selectivity bars + SNIIC2→SNIIC1 observational drift.

Self-contained script consolidated from:
  - scripts/plot_fig2_partner_sniic_drift_journal.py
  - panel_style.py / plot_utils.py (minimal helpers inlined)

Default output:
  output_file/figure3_hijk.png
  (= archived panel Fig2_partner_bars_SNIIC2_to_SNIIC1_drift.png; manuscript Figure 3)

Usage:
  python output_file/figure3_hijk.py
  python output_file/figure3_hijk.py /path/to/out.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import CK_PAIN, cache_file, load_obs  # noqa: E402

CK = CK_PAIN
DEFAULT_OUT = Path(__file__).resolve().parent / "figure3_hijk.png"

CONDITION_ORDER = ["Control", "SNI 6h", "SNI 24h", "SNI 2d", "SNI 7d", "SNI 14d"]
COND_SHORT = {
    "Control": "Ctrl",
    "SNI 6h": "6h",
    "SNI 24h": "24h",
    "SNI 2d": "2d",
    "SNI 7d": "7d",
    "SNI 14d": "14d",
}

SNIIC_MODULES = {
    "SNIIC1": ("Atf3", "Gfra3", "Gal"),
    "SNIIC2": ("Atf3", "Mrgprd"),
    "SNIIC3": ("Atf3", "S100b", "Gal"),
}

# Shared palette
C1 = "#6F9E9C"  # SNIIC1
C2 = "#C9A59A"  # SNIIC2
FCOL = "#3D6F8E"
ARROW = "#4A5A68"

# ---------------------------------------------------------------------------
# Style (inlined)
# ---------------------------------------------------------------------------
PANEL_TITLE_SIZE = 10
PANEL_TITLE_WEIGHT = "bold"
PANEL_TITLE_LOC = "center"
PANEL_TITLE_PAD = 4
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 7.5
ANNOT_SIZE = 7

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e3e8ee"
PANEL_BG = "#ffffff"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": PANEL_TITLE_WEIGHT,
            "axes.titlelocation": PANEL_TITLE_LOC,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.9,
            "text.color": INK,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": PANEL_BG,
            "axes.axisbelow": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
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


def _set_panel_title(ax, title: str, **kwargs) -> None:
    kw = {
        "loc": PANEL_TITLE_LOC,
        "fontweight": PANEL_TITLE_WEIGHT,
        "fontsize": PANEL_TITLE_SIZE,
        "pad": PANEL_TITLE_PAD,
    }
    kw.update(kwargs)
    ax.set_title(title, **kw)


def _resolve_genes(var_names, aliases: Sequence[str]) -> list[str]:
    name_set = {str(g): str(g) for g in var_names}
    lower = {str(g).lower(): str(g) for g in var_names}
    out = []
    for a in aliases:
        if a in name_set:
            out.append(name_set[a])
        elif a.lower() in lower:
            out.append(lower[a.lower()])
    return list(dict.fromkeys(out))


def _gene_expression(adata, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        return np.full(adata.n_obs, np.nan)
    col = adata.var_names.get_loc(gene)
    x = adata.X
    if hasattr(x, "toarray"):
        return np.asarray(x[:, col].toarray(), dtype=float).ravel()
    return np.asarray(x[:, col], dtype=float).ravel()


def _module_score(adata, genes: Sequence[str]) -> np.ndarray:
    resolved = _resolve_genes(adata.var_names, genes)
    if not resolved:
        return np.full(adata.n_obs, np.nan)
    mats = [_gene_expression(adata, g) for g in resolved]
    return np.nanmean(np.vstack(mats), axis=0)


def _set_partner_title(ax) -> None:
    """Title with bold-italic Atf3."""
    fp_gene = FontProperties(family="DejaVu Sans", style="italic", weight="bold", size=PANEL_TITLE_SIZE)
    fp_rest = FontProperties(family="DejaVu Sans", style="normal", weight="bold", size=PANEL_TITLE_SIZE)
    pack = HPacker(
        children=[
            TextArea("Partner selectivity under ", textprops=dict(fontproperties=fp_rest, color="black")),
            TextArea("Atf3", textprops=dict(fontproperties=fp_gene, color="black")),
            TextArea("-KO", textprops=dict(fontproperties=fp_rest, color="black")),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    ax.set_title(" ", pad=PANEL_TITLE_PAD)
    box = AnchoredOffsetbox(
        loc="lower center",
        child=pack,
        pad=0.0,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.45,
    )
    ax.add_artist(box)


def _load_partner_df() -> pd.DataFrame:
    stats_path = cache_file("Atf3_module_robustness_stats.csv")
    src = ROOT / "output_file" / "robustness" / "p0_robustness" / "Atf3_module_robustness_stats.csv"
    if not stats_path.is_file() and src.is_file():
        pd.read_csv(src).to_csv(stats_path, index=False)
    if not stats_path.is_file():
        print("[figure3_hijk] computing Atf3 partner-module robustness...", flush=True)
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_p0_robustness import run_atf3_p0

        run_atf3_p0()
        if src.is_file():
            pd.read_csv(src).to_csv(stats_path, index=False)
    stats = pd.read_csv(stats_path)
    partner_order = [
        ("SNIIC1_noAtf3", r"$\mathit{Gfra3}$+$\mathit{Gal}$"),
        ("SNIIC3_noAtf3", r"$\mathit{S100b}$+$\mathit{Gal}$"),
        ("SNIIC2_noAtf3", r"$\mathit{Mrgprd}$"),
        ("Atf3_alone", r"$\mathit{Atf3}$ alone"),
    ]
    rows = []
    for mid, lab in partner_order:
        r = stats.loc[stats.module == mid].iloc[0]
        rows.append({"label": lab, "WT": float(r["end_WT"]), "KO": float(r["end_KO"])})
    return pd.DataFrame(rows)


def _load_cell_scores() -> pd.DataFrame:
    cell_cache = cache_file("SNIIC_substate_cell_scores.csv")
    if cell_cache.is_file():
        df = pd.read_csv(cell_cache)
        df = df[df["condition"].astype(str).isin(CONDITION_ORDER)].copy()
        df["condition"] = pd.Categorical(
            df["condition"].astype(str), categories=CONDITION_ORDER, ordered=True
        )
        return df

    import anndata as ad
    from scipy import sparse

    from dataset_pipeline import GSE155622, resolve_data_path

    obs = load_obs(CK)
    neu = obs[obs["annotation"].astype(str) == "Neuron"].copy()
    print("[figure3_hijk] loading neuron expression...", flush=True)
    raw = ad.read_h5ad(resolve_data_path(GSE155622), backed="r")
    common = neu.index.intersection(raw.obs_names.astype(str))
    adata = raw[common].to_memory()
    raw.file.close()
    if sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(np.log1p(adata.X.toarray().astype(float)))
    else:
        adata.X = np.log1p(np.asarray(adata.X, dtype=float))
    s1 = _module_score(adata, SNIIC_MODULES["SNIIC1"])
    s2 = _module_score(adata, SNIIC_MODULES["SNIIC2"])
    df = neu.loc[common, ["condition"]].copy()
    df["SNIIC1"] = np.asarray(s1, dtype=float)
    df["SNIIC2"] = np.asarray(s2, dtype=float)
    df = df[df["condition"].astype(str).isin(CONDITION_ORDER)].copy()
    eps = 1e-12
    df["f_SNIIC1"] = df["SNIIC1"] / (df["SNIIC1"] + df["SNIIC2"] + eps)
    df.to_csv(cell_cache, index=False)
    df["condition"] = pd.Categorical(
        df["condition"].astype(str), categories=CONDITION_ORDER, ordered=True
    )
    return df


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        n = len(sub)
        if n == 0:
            continue

        def sem(y):
            y = np.asarray(y, dtype=float)
            return float(np.nanstd(y, ddof=1) / np.sqrt(max(1, np.isfinite(y).sum())))

        rows.append(
            {
                "condition": cond,
                "n": n,
                "SNIIC1_mean": float(np.nanmean(sub["SNIIC1"])),
                "SNIIC1_sem": sem(sub["SNIIC1"]),
                "SNIIC2_mean": float(np.nanmean(sub["SNIIC2"])),
                "SNIIC2_sem": sem(sub["SNIIC2"]),
                "f_mean": float(np.nanmean(sub["f_SNIIC1"])),
                "f_sem": sem(sub["f_SNIIC1"]),
            }
        )
    return pd.DataFrame(rows)


def compose(out: Path | None = None) -> Path:
    if out is None:
        out = DEFAULT_OUT
    out = Path(out)

    _apply_style()
    partner = _load_partner_df()
    cells = _load_cell_scores()
    summ = _agg(cells)

    fig = plt.figure(figsize=(13.2, 3.55), facecolor="white")
    gs = GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1.12, 1.18, 1.05, 0.82],
        wspace=0.20,
        left=0.048,
        right=0.992,
        top=0.86,
        bottom=0.20,
    )

    # ---------- A: partner WT→KO dumbbells ----------
    ax = fig.add_subplot(gs[0, 0])
    wt_pt = "#1B4F72"
    ko_pt = "#8ECAE6"
    stem_down = "#7FA3BD"
    stem_up = "#1B4F72"
    x = np.arange(len(partner))
    ymax = float(max(partner["WT"].max(), partner["KO"].max()))
    for i, r in partner.iterrows():
        wt, ko = float(r["WT"]), float(r["KO"])
        stem = stem_down if ko < wt - 1e-9 else stem_up
        ax.plot([i, i], [wt, ko], color=stem, lw=1.8, solid_capstyle="round", zorder=2)
        ax.scatter(
            i,
            wt,
            s=52,
            color=wt_pt,
            edgecolors="white",
            linewidths=0.9,
            zorder=4,
            label="WT" if i == 0 else None,
        )
        ax.scatter(
            i,
            ko,
            s=52,
            color=ko_pt,
            edgecolors=wt_pt,
            linewidths=0.7,
            zorder=4,
            label=r"$\mathit{Atf3}$-KO" if i == 0 else None,
        )
    ax.axhline(0, color="0.75", lw=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(partner["label"], fontsize=TICK_LABEL_SIZE - 0.5)
    ax.set_ylabel("Endpoint module score", fontsize=AXIS_LABEL_SIZE, labelpad=2)
    ax.set_xlim(-0.55, len(partner) - 0.45)
    ax.set_ylim(-0.06 * ymax, ymax * 1.14)
    ax.yaxis.set_label_coords(-0.11, 0.5)
    _set_partner_title(ax)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="upper left", handlelength=1.2, markerscale=0.95)
    _style_axis(ax, grid_axis="y")

    # ---------- B: SNIIC scores over SNI ----------
    ax = fig.add_subplot(gs[0, 1])
    xx = np.arange(len(summ))
    labels = [COND_SHORT[c] for c in summ["condition"]]
    ax.errorbar(
        xx - 0.06,
        summ["SNIIC2_mean"],
        yerr=summ["SNIIC2_sem"],
        fmt="-o",
        color=C2,
        lw=2.1,
        ms=5.8,
        capsize=2.2,
        label="SNIIC2 (Atf3+Mrgprd)",
        markerfacecolor="white",
        markeredgecolor=C2,
        markeredgewidth=1.5,
        elinewidth=1.1,
        zorder=3,
    )
    ax.errorbar(
        xx + 0.06,
        summ["SNIIC1_mean"],
        yerr=summ["SNIIC1_sem"],
        fmt="-o",
        color=C1,
        lw=2.1,
        ms=5.8,
        capsize=2.2,
        label="SNIIC1 (Atf3+Gfra3+Gal)",
        markerfacecolor="white",
        markeredgecolor=C1,
        markeredgewidth=1.5,
        elinewidth=1.1,
        zorder=3,
    )
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel("SNI time course", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Module score (mean±SEM)", fontsize=AXIS_LABEL_SIZE, labelpad=2)
    ax.yaxis.set_label_coords(-0.11, 0.5)
    _set_panel_title(ax, "SNIIC modules across injury time")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE - 0.3, loc="upper left", handlelength=1.4)
    _style_axis(ax, grid_axis="y")

    # ---------- C: transition index ----------
    ax = fig.add_subplot(gs[0, 2])
    ax.axhline(0.5, color="#B0B0B0", ls="--", lw=0.9, zorder=1)
    ax.errorbar(
        xx,
        summ["f_mean"],
        yerr=summ["f_sem"],
        fmt="-o",
        color=FCOL,
        lw=2.0,
        ms=5.8,
        capsize=2.2,
        markerfacecolor="white",
        markeredgecolor=FCOL,
        markeredgewidth=1.5,
        elinewidth=1.0,
        zorder=3,
    )
    ax.fill_between(
        xx, 0.5, summ["f_mean"], where=summ["f_mean"] >= 0.5, color=C1, alpha=0.12, zorder=0, interpolate=True
    )
    ax.fill_between(
        xx, summ["f_mean"], 0.5, where=summ["f_mean"] <= 0.5, color=C2, alpha=0.12, zorder=0, interpolate=True
    )
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel("SNI time course", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(
        r"$f=\mathrm{SNIIC1}/(\mathrm{SNIIC1}+\mathrm{SNIIC2})$",
        fontsize=AXIS_LABEL_SIZE - 0.5,
        labelpad=1,
    )
    ax.yaxis.set_label_coords(-0.13, 0.5)
    ax.set_ylim(0.25, 0.85)
    _set_panel_title(ax, r"Transition index (↑ toward SNIIC1)")
    ax.text(0.03, 0.92, "SNIIC1-leaning", transform=ax.transAxes, fontsize=ANNOT_SIZE, color="0.45", va="top")
    ax.text(0.03, 0.08, "SNIIC2-leaning", transform=ax.transAxes, fontsize=ANNOT_SIZE, color="0.45", va="bottom")
    _style_axis(ax, grid_axis="y")

    # ---------- D: state plane ----------
    ax = fig.add_subplot(gs[0, 3])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(cells), size=min(3500, len(cells)), replace=False)
    ax.scatter(
        cells["SNIIC2"].to_numpy()[idx],
        cells["SNIIC1"].to_numpy()[idx],
        s=3.5,
        alpha=0.06,
        c="#A8B2BC",
        rasterized=True,
        linewidths=0,
        zorder=1,
    )
    xs = summ["SNIIC2_mean"].to_numpy()
    ys = summ["SNIIC1_mean"].to_numpy()
    ax.plot(xs, ys, "-", color=ARROW, lw=1.25, alpha=0.9, zorder=3)
    for i in range(len(summ) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.15, mutation_scale=9),
            zorder=3,
        )
    label_pos = {
        "Control": dict(xytext=(0, -9), ha="center", va="top"),
        "SNI 6h": dict(xytext=(-8, -2), ha="right", va="center"),
        "SNI 24h": dict(xytext=(3, -12), ha="center", va="top"),
        "SNI 2d": dict(xytext=(7, 8), ha="left", va="bottom"),
        "SNI 7d": dict(xytext=(-7, 6), ha="right", va="bottom"),
        "SNI 14d": dict(xytext=(-8, 3), ha="right", va="bottom"),
    }
    for i, cond in enumerate(summ["condition"]):
        ax.scatter(xs[i], ys[i], s=36, facecolors="white", edgecolors=ARROW, linewidths=1.35, zorder=4)
        st = label_pos[cond]
        ax.annotate(
            cond,
            xy=(xs[i], ys[i]),
            xytext=st["xytext"],
            textcoords="offset points",
            fontsize=ANNOT_SIZE,
            color="0.25",
            ha=st["ha"],
            va=st["va"],
            zorder=5,
            annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.04", facecolor="white", edgecolor="none", alpha=0.55),
        )
    lim1 = float(max(np.nanpercentile(cells["SNIIC1"], 99), np.nanpercentile(cells["SNIIC2"], 99)) * 1.02)
    ax.plot([0, lim1], [0, lim1], ls="--", color="#C5C5C5", lw=0.9, zorder=0)
    ax.set_xlim(0, lim1)
    ax.set_ylim(0, lim1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SNIIC2 score", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("SNIIC1 score", fontsize=AXIS_LABEL_SIZE, labelpad=2)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    _set_panel_title(ax, "State plane (condition means)")
    ax.text(
        0.97,
        0.05,
        "above diagonal:\nSNIIC1 > SNIIC2",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOT_SIZE,
        color="0.45",
    )
    _style_axis(ax, grid_axis="none")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
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
