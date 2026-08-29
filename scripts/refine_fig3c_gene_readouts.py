#!/usr/bin/env python
"""Alveolar-lineage gene-KO readouts aligned to lung-injury biology.

Scientific framing (not AT1 vs Fibro lineage conversion):
  Main:   Krt8+ ADI → AT1 exit  vs  ADI persistent / trapping
  Aux:    AT2 → Krt8+ ADI entry
  Note:   Fibro/Myofibro are downstream tissue consequences, not ADI fates.

Metrics (reencode_only primary; hybrid_shift ablation):
  1. On Krt8+ ADI: Δz projected on ADI→AT1 exit axis
  2. On Krt8+ ADI: Δ(d_AT1 − d_ADI); negative = relatively closer to AT1 (exit)
  3. On AT2 cells: Δz projected on AT2→ADI entry axis
  4. Ablation: reencode vs hybrid unit-shift on the exit axis

Outputs:
  methods_enhancement/Fig3C_refined_gene_readouts.csv
  methods_enhancement/Fig3C_refined_gene_readouts.png
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import shutil
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from analysis_protocol_utils import resolve_genes
from dataset_pipeline import (
    PROJECT_ROOT,
    GSE141259,
    apply_train_config,
    recommended_checkpoint_dir,
    resolve_data_path,
)
from methods_model_utils import _load_state_dict_compat, inject_genes_into_panel
from plot_utils import configure_headless
from train_model import TemporalSDENetwork

configure_headless()
mpl.rcParams.update(
    {
        "axes.titlelocation": "center",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

CK = Path(
    recommended_checkpoint_dir("GSE141259")
    or PROJECT_ROOT
    / (
        "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
    )
)
OUT = CK / "methods_enhancement"
PANELS = OUT
TABLES = OUT

# High-U_rel candidates vs anti-varying controls (all KO)
GENES = [
    ("Lgals3", 0.0, "KO"),
    ("Cdkn1a", 0.0, "KO"),
    ("Spp1", 0.0, "KO"),
    ("Cbr2", 0.0, "KO"),
    ("Hc", 0.0, "KO"),
    ("Chi3l1", 0.0, "KO"),
]

ADI = "Krt8 ADI"
AT1 = "AT1 cells"
AT2 = "AT2 cells"
ACT_AT2 = "Activated AT2 cells"
ALV_TYPES = (AT2, ACT_AT2, ADI, AT1)

INK, MUTED, AT1_C, ADI_C, AT2_C, GRID = (
    "#111111",
    "#555555",
    "#1b6a7a",
    "#8B4A6B",
    "#2E7D4F",
    "#e8e8e8",
)


def _unit(v):
    n = np.linalg.norm(v)
    return v * 0.0 if (not np.isfinite(n) or n < 1e-12) else v / n


def _boot_mean_gt0(x, n=500, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    c = sum(1 for _ in range(n) if float(np.mean(rng.choice(x, x.size, True))) <= 0)
    return float((c + 1) / (n + 1))


def _boot_mean_lt0(x, n=500, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    c = sum(1 for _ in range(n) if float(np.mean(rng.choice(x, x.size, True))) >= 0)
    return float((c + 1) / (n + 1))


def _encode(model, X, ct_codes, device, bs=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            x = torch.tensor(X[s : s + bs], dtype=torch.float32, device=device)
            ct = torch.tensor(ct_codes[s : s + bs], dtype=torch.long, device=device)
            outs.append(model.encode(x, ct).cpu().numpy())
    return np.vstack(outs)


def _spine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, length=3, width=0.7)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def analyze_gene(
    *,
    model,
    config,
    X,
    ct,
    labels,
    gene: str,
    factor: float,
    tag: str,
    gene_col: int,
    mode: str,
):
    """mode: reencode_only | hybrid_shift"""
    print(f"  [{mode}] {tag} {gene} factor={factor}", flush=True)
    adi = labels == ADI
    at1 = labels == AT1
    at2 = labels == AT2

    X_kd = X.copy()
    X_kd[:, gene_col] = X_kd[:, gene_col] * float(factor)

    z_wt = _encode(model, X, ct, config.device, bs=int(config.batch_size))
    z_kd = _encode(model, X_kd, ct, config.device, bs=int(config.batch_size))

    # WT centroids / axes (epithelial lineage only)
    c_adi = z_wt[adi].mean(0)
    c_at1 = z_wt[at1].mean(0)
    c_at2 = z_wt[at2].mean(0)
    axis_exit = _unit(c_at1 - c_adi)  # ADI → AT1 maturation
    axis_entry = _unit(c_adi - c_at2)  # AT2 → ADI entry

    if mode == "hybrid_shift":
        # unit mean encoder delta on ADI cells, applied globally
        mean_dz_adi = (z_kd[adi] - z_wt[adi]).mean(0)
        direction = _unit(mean_dz_adi)
        z_kd = z_wt + direction[None, :]

    # ---- Main: ADI → AT1 exit vs trapping ----
    dz_adi = z_kd[adi] - z_wt[adi]
    mean_dz_adi = dz_adi.mean(0)
    proj_exit_cells = dz_adi @ axis_exit

    d_at1_wt = np.linalg.norm(z_wt[adi] - c_at1, axis=1)
    d_at1_kd = np.linalg.norm(z_kd[adi] - c_at1, axis=1)
    d_adi_wt = np.linalg.norm(z_wt[adi] - c_adi, axis=1)
    d_adi_kd = np.linalg.norm(z_kd[adi] - c_adi, axis=1)
    # negative = relatively closer to AT1 than to ADI centroid (favor exit)
    rel_exit = (d_at1_kd - d_adi_kd) - (d_at1_wt - d_adi_wt)
    delta_d_adi = d_adi_kd - d_adi_wt  # >0 = leave ADI basin
    delta_d_at1 = d_at1_kd - d_at1_wt  # <0 = closer to AT1

    # ---- Aux: AT2 → ADI entry ----
    dz_at2 = z_kd[at2] - z_wt[at2]
    mean_dz_at2 = dz_at2.mean(0)
    proj_entry_cells = dz_at2 @ axis_entry
    d_adi_from_at2_wt = np.linalg.norm(z_wt[at2] - c_adi, axis=1)
    d_adi_from_at2_kd = np.linalg.norm(z_kd[at2] - c_adi, axis=1)
    d_at2_wt = np.linalg.norm(z_wt[at2] - c_at2, axis=1)
    d_at2_kd = np.linalg.norm(z_kd[at2] - c_at2, axis=1)
    # negative = relatively closer to ADI than to AT2 (favor entry)
    rel_entry = (d_adi_from_at2_kd - d_at2_kd) - (d_adi_from_at2_wt - d_at2_wt)

    return {
        "gene": gene,
        "perturbation": tag,
        "expr_factor": factor,
        "mode": mode,
        "n_adi": int(adi.sum()),
        "n_at1": int(at1.sum()),
        "n_at2": int(at2.sum()),
        # exit / trapping (ADI cells)
        "proj_mean_dz_on_exit_axis": float(np.dot(mean_dz_adi, axis_exit)),
        "mean_proj_exit_cells": float(np.mean(proj_exit_cells)),
        "p_exit_proj_gt0": _boot_mean_gt0(proj_exit_cells, seed=1),
        "mean_relative_AT1_vs_ADI": float(np.mean(rel_exit)),
        "p_relative_prefer_AT1_exit": _boot_mean_lt0(rel_exit, seed=5),
        "mean_delta_dist_ADI": float(np.mean(delta_d_adi)),
        "mean_delta_dist_AT1": float(np.mean(delta_d_at1)),
        "frac_closer_AT1": float(np.mean(delta_d_at1 < 0)),
        "frac_leave_ADI": float(np.mean(delta_d_adi > 0)),
        # entry (AT2 cells)
        "proj_mean_dz_on_entry_axis": float(np.dot(mean_dz_at2, axis_entry)),
        "mean_proj_entry_cells": float(np.mean(proj_entry_cells)),
        "p_entry_proj_gt0": _boot_mean_gt0(proj_entry_cells, seed=11),
        "mean_relative_ADI_vs_AT2": float(np.mean(rel_entry)),
        "p_relative_prefer_ADI_entry": _boot_mean_lt0(rel_entry, seed=15),
        # legacy aliases used by older plotters (exit axis = "AT1 axis" conceptually)
        "proj_mean_dz_on_AT1_axis": float(np.dot(mean_dz_adi, axis_exit)),
        "mean_relative_AT1_vs_Fibro": float(np.mean(rel_exit)),  # deprecated name; now AT1 vs ADI
        "p_relative_prefer_AT1": _boot_mean_lt0(rel_exit, seed=5),
        "_arrays": {
            "proj_exit_cells": proj_exit_cells,
            "relative_AT1_vs_ADI": rel_exit,
            "delta_d_adi": delta_d_adi,
            "delta_d_at1": delta_d_at1,
            "proj_entry_cells": proj_entry_cells,
            "relative_ADI_vs_AT2": rel_entry,
            # keep old key for temporary compatibility
            "relative_closer_at1": rel_exit,
        },
    }


def draw_figure(rows, out_png: Path):
    re_rows = [r for r in rows if r["mode"] == "reencode_only"]
    hy_rows = [r for r in rows if r["mode"] == "hybrid_shift"]
    x = np.arange(len(re_rows))
    w = 0.36
    labels = [f"{r['gene']}\n{r['perturbation']}" for r in re_rows]

    fig = plt.figure(figsize=(9.6, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.45, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x, [r["proj_mean_dz_on_exit_axis"] for r in re_rows], color=AT1_C, width=0.62, edgecolor="none")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Proj. mean $\Delta z$ on ADI→AT1")
    ax.set_title("ADI → AT1 exit axis", loc="center", fontweight="bold")
    ax.axvline(2.5, color="#bbb", ls="--", lw=0.8)
    _spine(ax)

    ax = fig.add_subplot(gs[0, 1])
    vals = [r["mean_relative_AT1_vs_ADI"] for r in re_rows]
    cols = [AT1_C if v < 0 else ADI_C for v in vals]
    ax.bar(x, vals, color=cols, width=0.62, edgecolor="none")
    ax.axhline(0, color=INK, lw=0.6)
    for i, r in enumerate(re_rows):
        p = r["p_relative_prefer_AT1_exit"]
        if p <= 0.05:
            off = 0.002 if vals[i] >= 0 else -0.002
            ax.text(i, vals[i] + off, "*", ha="center", va="bottom" if vals[i] >= 0 else "top", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Mean $\Delta(d_{\mathrm{AT1}}-d_{\mathrm{ADI}})$")
    ax.set_title("Exit vs trapping (neg. = toward AT1)", loc="center", fontweight="bold")
    ax.axvline(2.5, color="#bbb", ls="--", lw=0.8)
    _spine(ax)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(x, [r["proj_mean_dz_on_entry_axis"] for r in re_rows], color=AT2_C, width=0.62, edgecolor="none")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Proj. mean $\Delta z$ on AT2→ADI")
    ax.set_title("AT2 → ADI entry axis", loc="center", fontweight="bold")
    ax.axvline(2.5, color="#bbb", ls="--", lw=0.8)
    _spine(ax)

    ax = fig.add_subplot(gs[1, 1])
    hy_map = {r["gene"]: r for r in hy_rows}
    ax.bar(
        x - w / 2,
        [r["proj_mean_dz_on_exit_axis"] for r in re_rows],
        w,
        color=AT1_C,
        label="reencode only",
        edgecolor="none",
    )
    ax.bar(
        x + w / 2,
        [hy_map[g]["proj_mean_dz_on_exit_axis"] for g in [r["gene"] for r in re_rows]],
        w,
        color="#8aa6ad",
        label="hybrid unit-shift",
        edgecolor="none",
    )
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x, [r["gene"] for r in re_rows])
    ax.set_ylabel(r"Proj. mean $\Delta z$ on exit axis")
    ax.set_title("Ablation: reencode vs hybrid", loc="center", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False)
    ax.axvline(2.5, color="#bbb", ls="--", lw=0.8)
    _spine(ax)

    fig.suptitle(
        r"Refined KO readouts: ADI→AT1 exit / trapping (+ AT2→ADI entry)"
        "\n(Fibro not used as lineage fate)",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    print("Saved", out_png, flush=True)


def _load_alv_panel_and_model():
    """Load AT2 / Act.AT2 / Krt8 ADI / AT1 × training genes."""
    import scipy.sparse as sp

    print("[1/5] read obs + gene panel ...", flush=True)
    obs = pd.read_csv(CK / "obs.csv", index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    gene_list = json.loads((CK / "training_var_names.json").read_text(encoding="utf-8"))
    missing_forced = [g for g, _, _ in GENES if g not in gene_list]
    if missing_forced:

        class _Panel:
            var_names = gene_list

        gene_list = inject_genes_into_panel(gene_list, _Panel(), missing_forced)

    mask = obs["cell.type"].astype(str).isin(ALV_TYPES)
    barcodes = obs.index[mask].tolist()
    print(
        f"[1/5] alv barcodes={len(barcodes)} "
        + ", ".join(f"{t}={(obs.loc[barcodes,'cell.type'].astype(str)==t).sum()}" for t in ALV_TYPES),
        flush=True,
    )

    h5ad = resolve_data_path(GSE141259)
    print(f"[2/5] backed read {h5ad} ...", flush=True)
    raw = ad.read_h5ad(h5ad, backed="r")
    raw_names = raw.obs_names.astype(str)
    name_to_i = {b: i for i, b in enumerate(raw_names)}
    idx = [name_to_i[b] for b in barcodes if b in name_to_i]
    keep_barcodes = [b for b in barcodes if b in name_to_i]
    present_genes = [g for g in gene_list if g in set(map(str, raw.var_names))]
    print(f"[2/5] overlap cells={len(idx)} genes_present={len(present_genes)}/{len(gene_list)}", flush=True)
    # h5py backed AnnData allows only one fancy index at a time.
    sub = raw[idx].to_memory()
    raw.file.close()
    present_genes = [g for g in present_genes if g in set(map(str, sub.var_names))]
    sub = sub[:, present_genes].copy()

    missing = [g for g in gene_list if g not in sub.var_names]
    if missing:
        zeros = sp.csr_matrix((sub.n_obs, len(missing)), dtype=np.float32)
        filler = ad.AnnData(X=zeros, obs=sub.obs.copy(), var=pd.DataFrame(index=pd.Index(missing)))
        sub = ad.concat([sub, filler], axis=1, join="outer", merge="same")
    sub = sub[:, list(gene_list)].copy()
    sub.obs_names = pd.Index(keep_barcodes)
    sub.obs["cell.type"] = obs.loc[keep_barcodes, "cell.type"].astype(str).values
    sub.obs["cell_type"] = obs.loc[keep_barcodes, "cell_type"].astype(int).values

    if "log1p" not in sub.uns:
        sc.pp.normalize_total(sub, target_sum=1e4, inplace=True)
        sc.pp.log1p(sub)

    print("[3/5] load model on CPU ...", flush=True)
    config = apply_train_config(GSE141259)
    config.n_top_genes = len(gene_list)
    config.use_hvg = True
    config.device = "cpu"
    config.show_figures = False
    config.cell_type_key = "cell_type"

    n_types = int(obs["cell_type"].max()) + 1
    stub = ad.AnnData(X=np.zeros((n_types, len(gene_list)), dtype=np.float32))
    stub.var_names = pd.Index(gene_list)
    stub.obs["cell_type"] = np.arange(n_types, dtype=int)
    model = TemporalSDENetwork(config, stub)
    state = torch.load(CK / "best_model.pth", map_location="cpu")
    _load_state_dict_compat(model, state)
    model = model.to("cpu").eval()
    print(f"[3/5] model ready input_dim={model.input_dim} n_types={model.n_types}", flush=True)

    X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X, dtype=float)
    labels = sub.obs["cell.type"].astype(str).values
    ct = sub.obs["cell_type"].astype(int).values
    return model, config, X, labels, ct, list(gene_list)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    model, config, X, labels, ct, gene_list = _load_alv_panel_and_model()
    print(
        f"[4/5] matrix {X.shape} AT2={(labels==AT2).sum()} ActAT2={(labels==ACT_AT2).sum()} "
        f"ADI={(labels==ADI).sum()} AT1={(labels==AT1).sum()}",
        flush=True,
    )

    resolved_map = {}
    for g, _, _ in GENES:
        rr = resolve_genes(gene_list, [g])
        if not rr:
            raise ValueError(f"Gene missing from panel: {g}")
        resolved_map[g] = gene_list.index(rr[0])

    print("[5/5] gene metrics ...", flush=True)
    rows = []
    for gene, factor, tag in GENES:
        for mode in ("reencode_only", "hybrid_shift"):
            rows.append(
                analyze_gene(
                    model=model,
                    config=config,
                    X=X,
                    ct=ct,
                    labels=labels,
                    gene=gene,
                    factor=factor,
                    tag=tag,
                    gene_col=resolved_map[gene],
                    mode=mode,
                )
            )

    arr_dir = OUT / "fig3c_refined_arrays"
    arr_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for r in rows:
        arrays = r.pop("_arrays")
        stem = f"{r['gene']}_{r['perturbation']}_{r['mode']}"
        np.savez_compressed(arr_dir / f"{stem}.npz", **arrays)
        table.append(r)
    df = pd.DataFrame(table)
    csv = OUT / "Fig3C_refined_gene_readouts.csv"
    df.to_csv(csv, index=False)

    for r in rows:
        stem = f"{r['gene']}_{r['perturbation']}_{r['mode']}"
        r["_arrays"] = dict(np.load(arr_dir / f"{stem}.npz"))

    fig_out = OUT / "figures" / "Fig3C_refined_gene_readouts.png"
    draw_figure(rows, fig_out)

    re_df = df[df["mode"] == "reencode_only"]
    summary = {
        "framework": "ADI→AT1 exit/trapping (+ AT2→ADI entry); Fibro not a lineage fate",
        "ranking_exit_axis": re_df.sort_values("proj_mean_dz_on_exit_axis", ascending=False)[
            ["gene", "proj_mean_dz_on_exit_axis", "p_exit_proj_gt0"]
        ].to_dict(orient="records"),
        "ranking_prefer_AT1_exit": re_df.sort_values("mean_relative_AT1_vs_ADI")[
            ["gene", "mean_relative_AT1_vs_ADI", "p_relative_prefer_AT1_exit"]
        ].to_dict(orient="records"),
        "ranking_entry_axis": re_df.sort_values("proj_mean_dz_on_entry_axis", ascending=False)[
            ["gene", "proj_mean_dz_on_entry_axis", "p_entry_proj_gt0"]
        ].to_dict(orient="records"),
    }
    (OUT / "Fig3C_refined_gene_readouts_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
