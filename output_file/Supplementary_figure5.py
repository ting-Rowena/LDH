#!/usr/bin/env python3
"""Supplementary Figure 5 for GSE155622 (nerve injury / SNI).

a–c: tissue-wide injury DEGs × 9 cell-type correlation
d–f: four Neuron subtypes ($U_0$, $U_{rel}$, time course)

Outputs:
  output_file/Supplementary_figure5.png
  output_file/Supplementary_table9_GSE155622_injury_DEG_celltype_corr.csv
  output_file/Supplementary_table10_GSE155622_neuron4_potential.csv

Usage:
  python output_file/Supplementary_figure5.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import sparse, stats

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "output_file"))

from dataset_pipeline import (  # noqa: E402
    GSE155622,
    GSE155622_CELLTYPE_PALETTE,
    GSE155622_MAIN_CELL_TYPES,
    GSE155622_NEURON_SUBTYPE,
    resolve_data_path,
)
from _supp_compose import save_fig  # noqa: E402

CK = ROOT / (
    "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
OUT = Path(__file__).resolve().parent
REPORT = Path(__file__).resolve().parent / "robustness" / "gse155622_supp_injury_neuron4"
REPORT.mkdir(parents=True, exist_ok=True)

COND_ORDER = ["Control", "SNI 6h", "SNI 24h", "SNI 2d", "SNI 7d", "SNI 14d"]
LATE = {"SNI 7d", "SNI 14d"}
SUBTYPES = ["Myelinated", "Non_peptidergic", "Peptidergic", "SNI-induced"]
SUB_PAL = {
    "Myelinated": "#3D6F8E",
    "Non_peptidergic": "#5E8EAE",
    "Peptidergic": "#6B9B6E",
    "SNI-induced": "#C4736B",
}
SUB_DISP = {
    "Myelinated": "Myelinated",
    "Non_peptidergic": "Non-peptidergic",
    "Peptidergic": "Peptidergic",
    "SNI-induced": "SNI-induced",
}

INK = "#1F2933"
MUTED = "#6B7280"
GRID = "#E9EEF2"

N_TOP_DEG = 60
N_HEAT = 25


def _style_ax(ax, *, grid: str = "y") -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#D0D7DE")
    ax.spines["bottom"].set_color("#D0D7DE")
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(colors=MUTED, labelsize=7.2, length=2.2, width=0.6)
    if grid == "y":
        ax.yaxis.grid(True, color=GRID, lw=0.65, zorder=0)
    elif grid == "x":
        ax.xaxis.grid(True, color=GRID, lw=0.65, zorder=0)
    ax.set_axisbelow(True)


def _letter(ax, letter: str, *, x: float = -0.02, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        str(letter).upper(),
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _load_obs() -> pd.DataFrame:
    obs = pd.read_csv(CK / "obs.csv", low_memory=False)
    if "barcode" in obs.columns:
        obs.index = obs["barcode"].astype(str)
    elif "Unnamed: 0" in obs.columns:
        obs.index = obs["Unnamed: 0"].astype(str)
    else:
        obs.index = obs.index.astype(str)
    ct_col = "celltype" if "celltype" in obs.columns else "annotation"
    obs["celltype"] = obs[ct_col].astype(str)
    obs["condition"] = obs["condition"].astype(str)
    obs["neuron_subtype"] = obs["celltype_2"].map(GSE155622_NEURON_SUBTYPE)
    obs["U0"] = obs["potential_stationary"].astype(float)
    obs["U_rel"] = obs["potential_relative_type"].astype(float)
    return obs


def _training_genes() -> list[str]:
    return list(json.loads((CK / "training_var_names.json").read_text()))


def _to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _load_expression(cell_ids: pd.Index, genes: list[str]) -> tuple[np.ndarray, list[str], pd.Index]:
    """Load log1p expression for selected cells × genes from backed h5ad."""
    import anndata as ad

    h5 = Path(resolve_data_path(GSE155622))
    print(f"[load] {h5.name}  cells={len(cell_ids)} genes={len(genes)}", flush=True)
    raw = ad.read_h5ad(h5, backed="r")
    common = cell_ids.intersection(raw.obs_names.astype(str))
    if len(common) == 0:
        raw.file.close()
        raise RuntimeError("No overlapping cell barcodes with h5ad")

    lower = {str(g).lower(): str(g) for g in raw.var_names}
    use = []
    for g in genes:
        if g in raw.var_names:
            use.append(g)
        elif g.lower() in lower:
            use.append(lower[g.lower()])
    if not use:
        raw.file.close()
        raise RuntimeError("None of the requested genes found in h5ad")

    # Prefer one contiguous slice when possible; else chunk by gene batches.
    # Loading all selected cells for the gene panel at once is usually OK (~6466×3000).
    ad_sub = raw[common, use].to_memory()
    raw.file.close()
    X = np.log1p(_to_dense(ad_sub.X))
    return X, list(ad_sub.var_names.astype(str)), ad_sub.obs_names.astype(str)


def _fast_wilcoxon_late_vs_control(X: np.ndarray, is_late: np.ndarray, is_ctrl: np.ndarray) -> pd.DataFrame:
    """Gene-wise Mann–Whitney U (late SNI vs Control), vectorized per gene."""
    rows = []
    x_late = X[is_late]
    x_ctrl = X[is_ctrl]
    n1, n2 = x_late.shape[0], x_ctrl.shape[0]
    for j in range(X.shape[1]):
        a = x_late[:, j]
        b = x_ctrl[:, j]
        # Skip near-empty genes
        if (a > 0).sum() + (b > 0).sum() < 20:
            continue
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        mu_l, mu_c = float(a.mean()), float(b.mean())
        # effect on log1p scale
        lfc = mu_l - mu_c
        rows.append(
            {
                "gene_idx": j,
                "U": float(u),
                "pval": float(p),
                "mean_late": mu_l,
                "mean_control": mu_c,
                "log1p_diff": lfc,
                "n_late": n1,
                "n_control": n2,
            }
        )
    return pd.DataFrame(rows)


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out


def compute_injury_deg_and_corr(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tissue-wide injury DEGs (all cell types), then cell-type correlation.

    IMPORTANT: DEGs are NOT discovered inside Neuron alone — that would
    circularly inflate Neuron–gene correlations.
    """
    genes = _training_genes()
    # DEG discovery on Control + late SNI cells across ALL major types
    deg_mask = obs["condition"].isin({"Control"} | LATE) & obs["celltype"].isin(
        GSE155622_MAIN_CELL_TYPES
    )
    deg_ids = obs.index[deg_mask]
    X, gene_names, common = _load_expression(deg_ids, genes)
    obs_d = obs.loc[common]
    is_late = obs_d["condition"].isin(LATE).to_numpy()
    is_ctrl = (obs_d["condition"] == "Control").to_numpy()
    print(
        f"[DEG] tissue-wide late={is_late.sum()} control={is_ctrl.sum()} "
        f"genes={len(gene_names)} types={obs_d['celltype'].nunique()}",
        flush=True,
    )

    deg = _fast_wilcoxon_late_vs_control(X, is_late, is_ctrl)
    deg["gene"] = [gene_names[i] for i in deg["gene_idx"]]
    deg["pval_adj"] = _bh_fdr(deg["pval"].to_numpy())
    deg = deg.sort_values(["pval_adj", "pval"]).reset_index(drop=True)
    deg["significant"] = (deg["pval_adj"] < 0.05) & (deg["log1p_diff"].abs() >= 0.10)
    deg_path = REPORT / "injury_DEG_tissue_late_vs_control.csv"
    deg.to_csv(deg_path, index=False)
    print(f"[DEG] sig={int(deg['significant'].sum())} / {len(deg)}  → {deg_path}", flush=True)

    top = deg.loc[deg["significant"]].head(N_TOP_DEG)
    if top.empty:
        top = deg.head(N_TOP_DEG)
    top_genes = top["gene"].tolist()

    # Correlation / mean expression: top injury DEGs × all cells
    Xa, g2, all_common = _load_expression(obs.index, top_genes)
    obs_a = obs.loc[all_common]
    types = list(GSE155622_MAIN_CELL_TYPES)

    corr_rows = []
    mean_mat = np.zeros((len(g2), len(types)))
    for j, g in enumerate(g2):
        x = Xa[:, j]
        for ti, t in enumerate(types):
            y = (obs_a["celltype"].to_numpy() == t).astype(float)
            if y.std() < 1e-12 or np.nanstd(x) < 1e-12:
                r = 0.0
            else:
                r = float(np.corrcoef(x, y)[0, 1])
            mean_mat[j, ti] = float(x[y > 0.5].mean()) if (y > 0.5).any() else 0.0
            corr_rows.append({"gene": g, "celltype": t, "pearson_r": r})
        best = max(
            (rr for rr in corr_rows if rr["gene"] == g),
            key=lambda d: abs(d["pearson_r"]),
        )
        for rr in corr_rows:
            if rr["gene"] == g:
                rr["best_celltype"] = best["celltype"]
                rr["best_abs_r"] = abs(best["pearson_r"])

    corr_df = pd.DataFrame(corr_rows)
    summary = (
        corr_df.groupby("celltype", sort=False)
        .agg(
            mean_abs_r=("pearson_r", lambda s: float(np.mean(np.abs(s)))),
            mean_r=("pearson_r", "mean"),
            n_genes=("gene", "nunique"),
        )
        .reindex(types)
        .reset_index()
    )
    winners = (
        corr_df.drop_duplicates("gene")
        .groupby("best_celltype")
        .size()
        .reindex(types)
        .fillna(0)
        .astype(int)
    )
    summary["n_genes_max_abs_r"] = winners.to_numpy()
    summary["frac_genes_max_abs_r"] = summary["n_genes_max_abs_r"] / max(1, int(winners.sum()))

    deg_map = deg.set_index("gene")[["log1p_diff", "pval_adj", "mean_late", "mean_control"]]
    wide = corr_df.pivot(index="gene", columns="celltype", values="pearson_r")
    wide = wide.reindex(columns=types)
    wide = wide.join(deg_map, how="left")
    wide["best_celltype"] = corr_df.drop_duplicates("gene").set_index("gene")["best_celltype"]
    wide = wide.reset_index()

    wide.to_csv(REPORT / "injury_DEG_celltype_pearson.csv", index=False)
    summary.to_csv(REPORT / "injury_DEG_celltype_summary.csv", index=False)
    mean_df = pd.DataFrame(mean_mat, index=g2, columns=types)
    mean_df.to_csv(REPORT / "injury_DEG_celltype_mean_expr.csv")

    out_tab = OUT / "Supplementary_table9_GSE155622_injury_DEG_celltype_corr.csv"
    wide.to_csv(out_tab, index=False)
    print(f"[corr] Neuron wins {int(winners.get('Neuron', 0))}/{int(winners.sum())} genes", flush=True)
    return deg, summary, mean_df


def compose_sfig5(
    deg: pd.DataFrame,
    summary: pd.DataFrame,
    mean_df: pd.DataFrame,
    neu: pd.DataFrame,
    neu_summary: pd.DataFrame,
    by_cond: pd.DataFrame,
) -> Path:
    types = list(GSE155622_MAIN_CELL_TYPES)
    cand = deg.copy()
    if "significant" in cand.columns:
        cand = cand.loc[cand["significant"]].copy()
    cand = cand[cand["gene"].isin(mean_df.index)].copy()
    if cand.empty:
        cand = deg[deg["gene"].isin(mean_df.index)].copy()
    heat_genes = (
        cand.reindex(cand["log1p_diff"].abs().sort_values(ascending=False).index)
        .head(N_HEAT)["gene"]
        .tolist()
    )
    mat = mean_df.loc[heat_genes, types].to_numpy(dtype=float)
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True) + 1e-8
    mat_z = (mat - mu) / sd

    fig = plt.figure(figsize=(14.4, 14.2), facecolor="white")
    outer = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.05, 1.15, 1.05],
        hspace=0.34,
        left=0.07,
        right=0.93,
        top=0.94,
        bottom=0.04,
    )

    # ---- Row 1: a | b ----
    gs1 = outer[0].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.26)
    ax_a = fig.add_subplot(gs1[0, 0])
    dplot = deg.head(200).copy()
    colors = np.where(dplot["log1p_diff"] >= 0, "#C4736B", "#5E8EAE")
    ax_a.scatter(
        dplot["log1p_diff"],
        -np.log10(dplot["pval_adj"].clip(lower=1e-300)),
        c=colors,
        s=14,
        alpha=0.75,
        edgecolors="none",
        zorder=3,
    )
    ax_a.axvline(0, color="#CBD5E1", lw=0.8)
    ax_a.axhline(-np.log10(0.05), color="#94A3B8", lw=0.8, ls="--")
    ax_a.set_xlabel(r"Mean log1p difference (late SNI − Control)", fontsize=8.2, color=INK)
    ax_a.set_ylabel(r"$-\log_{10}$(FDR)", fontsize=8.2, color=INK)
    ax_a.set_title(
        "Tissue-wide injury DEGs (all cell types, training HVG panel)",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=6,
    )
    _style_ax(ax_a)
    _letter(ax_a, "A", x=-0.08, y=1.04)
    # Annotate extremes by effect size (many genes share FDR ceiling → avoid pile-up)
    dplot = dplot.assign(neglogp=-np.log10(dplot["pval_adj"].clip(lower=1e-300)))
    up = dplot.loc[dplot["log1p_diff"] > 0].nlargest(3, "log1p_diff")
    down = dplot.loc[dplot["log1p_diff"] < 0].nsmallest(3, "log1p_diff")
    # Stagger labels into side bands so neighboring genes do not collide
    up_offsets = [(8, 14), (18, -4), (10, -18)]
    down_offsets = [(-8, 16), (-10, 0), (-8, -16)]
    for (row, (dx, dy)) in list(zip(up.itertuples(index=False), up_offsets)) + list(
        zip(down.itertuples(index=False), down_offsets)
    ):
        ax_a.annotate(
            row.gene,
            (float(row.log1p_diff), float(row.neglogp)),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.5,
            color=MUTED,
            arrowprops=dict(
                arrowstyle="-",
                color="#C5CDD6",
                lw=0.55,
                shrinkA=0,
                shrinkB=3,
            ),
            ha="left" if dx >= 0 else "right",
            va="center",
            zorder=5,
            annotation_clip=False,
        )
    # Room for side labels
    x0, x1 = ax_a.get_xlim()
    ax_a.set_xlim(x0 - 0.15 * (x1 - x0), x1 + 0.08 * (x1 - x0))
    y0, y1 = ax_a.get_ylim()
    ax_a.set_ylim(y0, y1 * 1.04)

    ax_b = fig.add_subplot(gs1[0, 1])
    s = summary.set_index("celltype").reindex(types)
    cols = [GSE155622_CELLTYPE_PALETTE.get(t, "#888888") for t in types]
    y = np.arange(len(types))
    vals = s["mean_abs_r"].to_numpy(dtype=float)
    ax_b.barh(y, vals, color=cols, edgecolor="white", height=0.72, zorder=3)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(types, fontsize=8.0)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"Mean $|$Pearson $r|$ vs cell-type indicator", fontsize=8.0, color=INK)
    ax_b.set_title(
        "Injury DEGs ↔ major cell types",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=6,
    )
    _style_ax(ax_b, grid="x")
    _letter(ax_b, "B", x=-0.18, y=1.04)
    if "Neuron" in s.index:
        ni = types.index("Neuron")
        n_win = int(s.loc["Neuron", "n_genes_max_abs_r"])
        n_tot = int(s["n_genes_max_abs_r"].sum())
        ax_b.set_xlim(0, float(np.nanmax(vals)) * 1.12)
        # Place annotation inside axes (upper-right) so it cannot be clipped
        ax_b.text(
            0.98,
            0.12,
            f"Neuron max-|r|\n{n_win}/{n_tot} genes",
            transform=ax_b.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=INK,
            fontweight="bold",
            linespacing=1.25,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#E5E7EB", linewidth=0.6),
        )

    # ---- Row 2: c heatmap ----
    ax_c = fig.add_subplot(outer[1])
    vmax = float(np.nanpercentile(np.abs(mat_z), 98))
    vmax = max(1.0, float(np.ceil(vmax * 10) / 10))
    im = ax_c.imshow(mat_z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_c.set_xticks(np.arange(len(types)))
    ax_c.set_xticklabels(types, rotation=0, ha="center", fontsize=8.0)
    ax_c.set_yticks(np.arange(len(heat_genes)))
    ax_c.set_yticklabels(heat_genes, fontsize=7.0)
    ax_c.set_title(
        f"Top {len(heat_genes)} injury DEGs — mean expression by cell type (row z-score)",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=6,
    )
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.02, pad=0.015)
    cbar.set_label("Row z-score", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    _letter(ax_c, "C", x=-0.04, y=1.02)

    # ---- Row 3: d e f (former SFig8 a b c) ----
    gs3 = outer[2].subgridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.28)

    # d: U0 violin
    ax_d = fig.add_subplot(gs3[0, 0])
    data = [neu.loc[neu.neuron_subtype == st, "U0"].to_numpy() for st in SUBTYPES]
    parts = ax_d.violinplot(data, positions=np.arange(4), showmeans=False, showextrema=False, widths=0.78)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(SUB_PAL[SUBTYPES[i]])
        pc.set_edgecolor("white")
        pc.set_alpha(0.85)
        pc.set_linewidth(0.6)
    for i, st in enumerate(SUBTYPES):
        yv = data[i]
        rng = np.random.default_rng(i + 7)
        jitter = rng.uniform(-0.12, 0.12, size=min(400, yv.size))
        idx = rng.choice(yv.size, size=min(400, yv.size), replace=False)
        ax_d.scatter(
            np.full(idx.size, i) + jitter,
            yv[idx],
            s=3.5,
            color=SUB_PAL[st],
            alpha=0.25,
            edgecolors="none",
            zorder=3,
        )
        ax_d.hlines(np.mean(yv), i - 0.28, i + 0.28, colors=INK, lw=1.6, zorder=4)
    ax_d.set_xticks(range(4))
    ax_d.set_xticklabels([SUB_DISP[s] for s in SUBTYPES], fontsize=7.4, rotation=18, ha="right")
    ax_d.set_ylabel(r"Stationary potential $U_0$", fontsize=8.2, color=INK)
    ax_d.set_title(
        "Four Neuron subtypes — $U_0$",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=5,
    )
    _style_ax(ax_d)
    _letter(ax_d, "D", x=-0.10, y=1.05)
    u0 = neu_summary[neu_summary.metric == "U0"].set_index("neuron_subtype")
    deepest = u0["mean"].idxmin()
    ax_d.text(
        0.98,
        0.04,
        f"Deepest: {SUB_DISP[deepest]}",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=SUB_PAL[deepest],
        fontweight="bold",
    )

    # e: U_rel box
    ax_e = fig.add_subplot(gs3[0, 1])
    urel = [neu.loc[neu.neuron_subtype == st, "U_rel"].to_numpy() for st in SUBTYPES]
    bp = ax_e.boxplot(
        urel,
        positions=np.arange(4),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=INK, lw=1.4),
        whiskerprops=dict(color=MUTED, lw=0.9),
        capprops=dict(color=MUTED, lw=0.9),
        boxprops=dict(lw=0.8),
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(SUB_PAL[SUBTYPES[i]])
        box.set_alpha(0.85)
        box.set_edgecolor("white")
    ax_e.set_xticks(range(4))
    ax_e.set_xticklabels([SUB_DISP[s] for s in SUBTYPES], fontsize=7.4, rotation=18, ha="right")
    ax_e.set_ylabel(r"$U_{\mathrm{rel}}$ (within Neuron)", fontsize=8.0, color=INK)
    ax_e.set_title(
        "Within-Neuron relative potential",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=5,
    )
    _style_ax(ax_e)
    _letter(ax_e, "E", x=-0.14, y=1.05)

    # f: U0 time course
    ax_f = fig.add_subplot(gs3[0, 2])
    for st in SUBTYPES:
        sub = by_cond[by_cond["neuron_subtype"] == st].copy()
        x = np.arange(len(COND_ORDER))
        yv = []
        for cond in COND_ORDER:
            row = sub[sub["condition"] == cond]
            yv.append(float(row["U0_mean"].iloc[0]) if not row.empty else np.nan)
        ax_f.plot(x, yv, "-o", color=SUB_PAL[st], lw=1.7, ms=5.0, label=SUB_DISP[st], zorder=3)
    ax_f.set_xticks(range(len(COND_ORDER)))
    ax_f.set_xticklabels(COND_ORDER, fontsize=6.8, rotation=25, ha="right")
    ax_f.set_ylabel(r"Mean $U_0$", fontsize=8.0, color=INK)
    ax_f.set_title(
        "Subtype $U_0$ across SNI time",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=5,
    )
    ax_f.legend(frameon=False, fontsize=6.6, loc="best", labelcolor=INK)
    _style_ax(ax_f)
    _letter(ax_f, "F", x=-0.10, y=1.05)

    return save_fig(fig, OUT / "Supplementary_figure5.png")


def compute_neuron4_tables(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neu = obs[(obs["celltype"] == "Neuron") & (obs["neuron_subtype"].isin(SUBTYPES))].copy()
    neu["neuron_subtype"] = pd.Categorical(neu["neuron_subtype"], categories=SUBTYPES, ordered=True)
    neu["condition"] = pd.Categorical(neu["condition"], categories=COND_ORDER, ordered=True)

    summ_rows = []
    for st in SUBTYPES:
        sub = neu.loc[neu["neuron_subtype"] == st]
        for metric, col in (("U0", "U0"), ("U_rel", "U_rel")):
            v = sub[col].to_numpy(dtype=float)
            summ_rows.append(
                {
                    "neuron_subtype": st,
                    "metric": metric,
                    "n_cells": int(v.size),
                    "mean": float(np.mean(v)),
                    "median": float(np.median(v)),
                    "std": float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
                    "q25": float(np.percentile(v, 25)),
                    "q75": float(np.percentile(v, 75)),
                }
            )
    summary = pd.DataFrame(summ_rows)
    u0 = summary[summary.metric == "U0"].sort_values("mean")
    u0 = u0.assign(rank_deepest=np.arange(1, len(u0) + 1))

    by_cond = (
        neu.groupby(["neuron_subtype", "condition"], observed=False)
        .agg(n_cells=("U0", "size"), U0_mean=("U0", "mean"), U_rel_mean=("U_rel", "mean"))
        .reset_index()
    )

    from itertools import combinations

    pair_rows = []
    for a, b in combinations(SUBTYPES, 2):
        xa = neu.loc[neu.neuron_subtype == a, "U0"].to_numpy()
        xb = neu.loc[neu.neuron_subtype == b, "U0"].to_numpy()
        _, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        pair_rows.append(
            {
                "a": a,
                "b": b,
                "delta_mean_U0": float(xa.mean() - xb.mean()),
                "mannwhitney_p": float(p),
            }
        )
    pairwise = pd.DataFrame(pair_rows)

    summary.to_csv(REPORT / "neuron4_potential_summary.csv", index=False)
    by_cond.to_csv(REPORT / "neuron4_potential_by_condition.csv", index=False)
    pairwise.to_csv(REPORT / "neuron4_U0_pairwise.csv", index=False)
    u0.to_csv(OUT / "Supplementary_table10_GSE155622_neuron4_potential.csv", index=False)
    return neu, summary, by_cond


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.unicode_minus": False,
        }
    )
    obs = _load_obs()

    deg_path = REPORT / "injury_DEG_tissue_late_vs_control.csv"
    sum_path = REPORT / "injury_DEG_celltype_summary.csv"
    mean_path = REPORT / "injury_DEG_celltype_mean_expr.csv"
    if deg_path.is_file() and sum_path.is_file() and mean_path.is_file():
        print("[1/2] using cached tissue-wide DEG/corr tables...", flush=True)
        deg = pd.read_csv(deg_path)
        summary = pd.read_csv(sum_path)
        mean_df = pd.read_csv(mean_path, index_col=0)
    else:
        print("[1/2] injury DEG + cell-type correlation...", flush=True)
        deg, summary, mean_df = compute_injury_deg_and_corr(obs)

    print("[2/2] neuron subtype potentials + compose...", flush=True)
    neu, summ4, by_cond = compute_neuron4_tables(obs)
    compose_sfig5(deg, summary, mean_df, neu, summ4, by_cond)
    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
