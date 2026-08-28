#!/usr/bin/env python
"""
HGSOC nactpair protocol analysis: EOC resistance attractor basin, Stromal iCAF climb,
and targeted ligand–receptor scoring between deep-valley EOC and high-U Stromal cells.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from plot_utils import (
    PALETTE,
    configure_headless,
    gradient_barh,
    polished_colorbar,
    style_axis,
)

configure_headless()

from analysis_protocol_utils import (
    CURATED_LR_PAIRS,
    ICAF_MARKERS,
    STRESS_ASSOCIATED_STATE,
    deep_valley_mask,
    fast_wilcoxon_deg,
    fig_path,
    init_protocol_outdir,
    module_score,
    resolve_genes,
    result_path,
    score_lr_pairs,
    spearman_safe,
    write_json,
    write_output_file_index,
)
from celltype_analysis import HGSOC_PROFILE, load_annotated_adata
from dataset_pipeline import PROJECT_ROOT
from deg_enrichment_workflow import run_pathway_enrichment

DEFAULT_CHECKPOINT = (
    "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)


def _resolve_ckpt(override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()
    return (PROJECT_ROOT / DEFAULT_CHECKPOINT).resolve()


def _phase_col(adata) -> str:
    if "treatment_phase" in adata.obs:
        return "treatment_phase"
    if "stage" in adata.obs and set(adata.obs["stage"].astype(str).unique()) & {
        "treatment-naive",
        "post-NACT",
    }:
        return "stage"
    return "time"


def _normalize_phase(series: pd.Series) -> pd.Series:
    m = {
        "treatment-naive": "naive",
        "treatment_naive": "naive",
        "naive": "naive",
        "0": "naive",
        "0.0": "naive",
        "post-NACT": "post",
        "post_NACT": "post",
        "post-nact": "post",
        "1": "post",
        "1.0": "post",
    }
    return series.astype(str).map(lambda x: m.get(x, x))


def step_eoc_attractor_basin(adata_full, checkpoint: Path, out_dir: Path) -> dict:
    report: Dict[str, object] = {}
    eoc = adata_full[adata_full.obs["annotation"].astype(str) == "EOC"].copy()
    print(f"EOC cells: {eoc.n_obs}")
    sc.pp.normalize_total(eoc, target_sum=1e4, inplace=True)
    sc.pp.log1p(eoc)

    pot_key = "potential_stationary" if "potential_stationary" in eoc.obs else "potential"
    eoc.obs["potential"] = eoc.obs[pot_key].astype(float)
    if "stability_score" not in eoc.obs:
        raise KeyError("stability_score missing")
    valley = deep_valley_mask(
        eoc.obs["potential"].values,
        eoc.obs["stability_score"].astype(float).values,
        u_quantile=0.15,
        stability_quantile=0.7,
    )
    eoc.obs["deep_valley"] = valley
    barcodes = eoc.obs_names[valley].astype(str).tolist()
    pd.Series(barcodes, name="barcode").to_csv(result_path(out_dir, "eoc_attractor_deep_valley_barcodes.csv"), index=False)
    report["n_eoc"] = int(eoc.n_obs)
    report["n_deep_valley"] = int(valley.sum())
    report["mean_U_valley"] = float(eoc.obs.loc[valley, "potential"].mean()) if valley.any() else float("nan")
    report["mean_U_other"] = float(eoc.obs.loc[~valley, "potential"].mean()) if (~valley).any() else float("nan")

    # Landscape figures by phase
    phase_col = _phase_col(eoc)
    eoc.obs["phase"] = _normalize_phase(eoc.obs[phase_col])
    try:
        from latent_embeddings import ensure_latent_embeddings

        ensure_latent_embeddings(eoc, checkpoint_dir=str(checkpoint), warn=True)
    except Exception as exc:
        warnings.warn(str(exc), UserWarning)
    if "X_umap" not in eoc.obsm:
        sc.pp.pca(eoc, n_comps=min(40, max(2, eoc.n_vars - 1)))
        sc.pp.neighbors(eoc, n_neighbors=15)
        sc.tl.umap(eoc)

    for phase in ("naive", "post"):
        sub = eoc[eoc.obs["phase"] == phase]
        if sub.n_obs < 30:
            continue
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        coords = np.asarray(sub.obsm["X_umap"][:, :2], dtype=float)
        c = sub.obs["potential"].astype(float).values
        scatt = ax.scatter(coords[:, 0], coords[:, 1], c=c, s=10, cmap="RdBu_r",
                           alpha=0.8, linewidths=0, zorder=2)
        polished_colorbar(scatt, ax, label=pot_key)
        m = sub.obs["deep_valley"].astype(bool).values
        if m.any():
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                s=42,
                facecolors="none",
                edgecolors="#1f2933",
                linewidths=1.1,
                label="deep-valley attractor",
                zorder=4,
            )
            ax.legend(loc="upper right", markerscale=1.1)
        phase_name = "chemo-naive" if phase == "naive" else "post-NACT"
        ax.set_title(f"EOC Waddington landscape · {phase_name}", fontsize=12)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        style_axis(ax, grid_axis="none")
        ax.margins(0.04)
        fig.tight_layout()
        phase_slug = "naive" if phase == "naive" else "postNACT"
        fig.savefig(
            fig_path(out_dir, f"eoc_attractor_Waddington_landscape_{phase_slug}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # DEG deep valley vs other EOC
    eoc.obs["valley_bin"] = np.where(valley, "deep_valley", "other_EOC")
    deg = fast_wilcoxon_deg(
        eoc,
        group_key="valley_bin",
        group_1="deep_valley",
        group_2="other_EOC",
        n_top_genes=3000,
        min_cells=15,
    )
    stress_hits = []
    if deg is not None and not deg.empty:
        deg.to_csv(result_path(out_dir, "eoc_attractor_deep_valley_DEG.csv"), index=False)
        report["deg_top"] = deg.head(20).to_dict(orient="records")
        padj = deg["pval_adj"] if "pval_adj" in deg.columns else deg.get("pval", 1.0)
        up_genes = set(deg.loc[(deg["logfoldchange"] > 0) & (padj < 0.05), "gene"].astype(str))
        stress_resolved = resolve_genes(eoc.var_names, STRESS_ASSOCIATED_STATE)
        stress_hits = sorted(up_genes.intersection(stress_resolved))
        report["stress_state_overlap_genes"] = stress_hits
        report["stress_state_overlap_n"] = len(stress_hits)
        report["stress_state_coverage"] = float(len(stress_hits) / max(len(stress_resolved), 1))
        enr, ewarn = run_pathway_enrichment(
            list(up_genes)[:300],
            comparison="deep_valley",
            direction="up",
            organism="Human",
            gene_sets=["MSigDB_Hallmark_2020", "GO_Biological_Process_2023"],
        )
        if ewarn:
            report["enrichment_warning"] = ewarn
        if enr is not None and not enr.empty:
            enr.to_csv(result_path(out_dir, "eoc_attractor_deep_valley_enrichment.csv"), index=False)
            report["enrichment_top"] = enr.head(15).to_dict(orient="records")

    # Module score coverage
    scs = module_score(eoc, STRESS_ASSOCIATED_STATE, obs_key="stress_module")
    report["stress_module_mean_valley"] = float(np.nanmean(scs[valley])) if valley.any() else float("nan")
    report["stress_module_mean_other"] = float(np.nanmean(scs[~valley])) if (~valley).any() else float("nan")
    write_json(report, result_path(out_dir, "eoc_attractor_summary.json"))
    print(
        f"[EOC basin] deep_valley n={report['n_deep_valley']} "
        f"stress_overlap={report.get('stress_state_overlap_n', 0)}"
    )
    return report


def step_stromal_icaf_climb(adata_full, out_dir: Path) -> dict:
    report: Dict[str, object] = {}
    st = adata_full[adata_full.obs["annotation"].astype(str) == "Stromal"].copy()
    print(f"Stromal cells: {st.n_obs}")
    sc.pp.normalize_total(st, target_sum=1e4, inplace=True)
    sc.pp.log1p(st)

    pot_key = "potential_stationary" if "potential_stationary" in st.obs else "potential"
    st.obs["potential"] = st.obs[pot_key].astype(float)
    phase_col = _phase_col(st)
    st.obs["phase"] = _normalize_phase(st.obs[phase_col])

    # Centroid displacement naive → post in UMAP / PCA
    if "X_umap" not in st.obsm:
        sc.pp.pca(st, n_comps=min(40, max(2, st.n_vars - 1)))
        sc.pp.neighbors(st, n_neighbors=15)
        sc.tl.umap(st)
    coords = np.asarray(st.obsm["X_umap"][:, :2], dtype=float)
    naive = st.obs["phase"].astype(str).values == "naive"
    post = st.obs["phase"].astype(str).values == "post"
    c0 = coords[naive].mean(axis=0) if naive.any() else np.zeros(2)
    c1 = coords[post].mean(axis=0) if post.any() else np.zeros(2)
    report["centroid_displacement"] = float(np.linalg.norm(c1 - c0))
    report["mean_U_naive"] = float(st.obs.loc[naive, "potential"].mean()) if naive.any() else float("nan")
    report["mean_U_post"] = float(st.obs.loc[post, "potential"].mean()) if post.any() else float("nan")
    report["potential_climb"] = float(report["mean_U_post"] - report["mean_U_naive"])

    icaf = module_score(st, ICAF_MARKERS, obs_key="iCAF_score")
    report["iCAF_mean_naive"] = float(np.nanmean(icaf[naive])) if naive.any() else float("nan")
    report["iCAF_mean_post"] = float(np.nanmean(icaf[post])) if post.any() else float("nan")
    report["iCAF_vs_potential_spearman"] = spearman_safe(icaf, st.obs["potential"].astype(float).values)

    # Gene dynamics for climbing cells (high U post-NACT)
    climb_mask = post & (
        st.obs["potential"].astype(float).values
        >= np.nanquantile(st.obs.loc[post, "potential"].astype(float), 0.6)
        if post.any()
        else False
    )
    st.obs["climb_bin"] = np.where(climb_mask, "highU_post", "other")
    deg = fast_wilcoxon_deg(
        st,
        group_key="climb_bin",
        group_1="highU_post",
        group_2="other",
        n_top_genes=3000,
        min_cells=20,
    )
    ligand_candidates = []
    if deg is not None and not deg.empty:
        deg.to_csv(result_path(out_dir, "stromal_iCAF_climb_DEG.csv"), index=False)
        padj = deg["pval_adj"] if "pval_adj" in deg.columns else deg.get("pval", 1.0)
        up = deg.loc[(deg["logfoldchange"] > 0) & (padj < 0.1), "gene"].astype(str)
        ligands = set(resolve_genes(st.var_names, [p[0] for p in CURATED_LR_PAIRS] + list(ICAF_MARKERS)))
        ligand_candidates = sorted(set(up).intersection(ligands))
        report["upstream_ligand_candidates"] = ligand_candidates
        report["deg_top"] = deg.head(15).to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for phase, color, name in (("naive", PALETTE[0], "chemo-naive"), ("post", PALETTE[5], "post-NACT")):
        m = st.obs["phase"].astype(str).values == phase
        ax.scatter(coords[m, 0], coords[m, 1], s=8, c=color, alpha=0.55,
                   label=name, linewidths=0, zorder=2)
    ax.annotate("", xy=c1, xytext=c0,
                arrowprops=dict(arrowstyle="-|>", color="#1f2933", lw=2.4,
                                mutation_scale=22), zorder=5)
    ax.scatter(*c0, s=90, color="white", edgecolors=PALETTE[0], linewidths=2.2, zorder=6)
    ax.scatter(*c1, s=90, color="white", edgecolors=PALETTE[5], linewidths=2.2, zorder=6)
    ax.set_title(
        "Stromal fibroblast climb naive → post-NACT\n"
        f"centroid shift = {report['centroid_displacement']:.2f}   "
        f"ΔU = {report['potential_climb']:+.4f}   iCAF$_{{post}}$ = {report['iCAF_mean_post']:.2f}",
        fontsize=11.5,
    )
    ax.legend(loc="upper right", markerscale=1.6)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    style_axis(ax, grid_axis="none")
    ax.margins(0.04)
    fig.tight_layout()
    fig.savefig(fig_path(out_dir, "stromal_iCAF_centroid_climb_naive_to_postNACT.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "metric": k,
                "value": v,
            }
            for k, v in report.items()
            if not isinstance(v, (list, dict))
        ]
    ).to_csv(result_path(out_dir, "stromal_iCAF_climb_metrics.csv"), index=False)
    write_json(report, result_path(out_dir, "stromal_iCAF_summary.json"))
    print(
        f"[Stromal] Δcentroid={report['centroid_displacement']:.3f} "
        f"iCAF {report['iCAF_mean_naive']:.3f}→{report['iCAF_mean_post']:.3f}"
    )
    return report


def step_targeted_ccc(adata_full, eoc_report: dict, out_dir: Path) -> dict:
    report: Dict[str, object] = {}
    eoc = adata_full[adata_full.obs["annotation"].astype(str) == "EOC"].copy()
    st = adata_full[adata_full.obs["annotation"].astype(str) == "Stromal"].copy()
    sc.pp.normalize_total(eoc, target_sum=1e4, inplace=True)
    sc.pp.log1p(eoc)
    sc.pp.normalize_total(st, target_sum=1e4, inplace=True)
    sc.pp.log1p(st)

    pot_e = eoc.obs.get("potential_stationary", eoc.obs["potential"]).astype(float).values
    stab = eoc.obs["stability_score"].astype(float).values
    valley = deep_valley_mask(pot_e, stab)
    eoc_deep = eoc[valley].copy()

    pot_s = st.obs.get("potential_stationary", st.obs["potential"]).astype(float).values
    high_u = pot_s >= np.nanquantile(pot_s, 0.7)
    st_high = st[high_u].copy()

    # Export barcodes / expression subsets for optional CellChat
    pd.Series(eoc_deep.obs_names.astype(str), name="barcode").to_csv(
        result_path(out_dir, "ccc_EOC_deep_valley_barcodes.csv"), index=False
    )
    pd.Series(st_high.obs_names.astype(str), name="barcode").to_csv(
        result_path(out_dir, "ccc_Stromal_highU_barcodes.csv"), index=False
    )
    report["n_eoc_deep"] = int(eoc_deep.n_obs)
    report["n_stromal_highU"] = int(st_high.n_obs)

    # Bidirectional LR scores
    lr_s2e = score_lr_pairs(st_high, eoc_deep, CURATED_LR_PAIRS)
    lr_e2s = score_lr_pairs(eoc_deep, st_high, CURATED_LR_PAIRS)
    if not lr_s2e.empty:
        lr_s2e.insert(0, "direction", "Stromal_highU→EOC_deep")
        lr_s2e.to_csv(result_path(out_dir, "ccc_LR_Stromal_to_EOC.csv"), index=False)
    if not lr_e2s.empty:
        lr_e2s.insert(0, "direction", "EOC_deep→Stromal_highU")
        lr_e2s.to_csv(result_path(out_dir, "ccc_LR_EOC_to_Stromal.csv"), index=False)
    both = pd.concat([lr_s2e, lr_e2s], ignore_index=True) if len(lr_s2e) or len(lr_e2s) else pd.DataFrame()
    if not both.empty:
        both.to_csv(result_path(out_dir, "ccc_LR_paracrine_feedforward.csv"), index=False)
        report["lr_top"] = both.head(20).to_dict(orient="records")
        # Top ligand-receptor candidates, colored by signalling direction
        top = both.sort_values("lr_score", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        labels = [f"{r.ligand} → {r.receptor}" for r in top.itertuples()]
        dirs = top["direction"].astype(str).values if "direction" in top.columns else np.array([""] * len(top))
        dir_colors = {
            "Stromal_highU→EOC_deep": PALETTE[2],
            "EOC_deep→Stromal_highU": PALETTE[5],
        }
        colors = [dir_colors.get(d, PALETTE[0]) for d in dirs]
        y = np.arange(len(labels))
        ax.barh(y, top["lr_score"].values, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("LR score  (mean ligand × mean receptor)")
        ax.set_title("Paracrine feed-forward loop:\ndeep-valley EOC ↔ high-U stromal", fontsize=12)
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=col, label=("Stromal → EOC" if d.startswith("Stromal") else "EOC → Stromal"))
                   for d, col in dir_colors.items()]
        ax.legend(handles=handles, loc="lower right")
        style_axis(ax, grid_axis="x")
        fig.tight_layout()
        fig.savefig(fig_path(out_dir, "ccc_LR_paracrine_feedforward_top.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        report["lr_top"] = []
    report["n_lr_pairs"] = int(len(both))
    write_json(report, result_path(out_dir, "ccc_summary.json"))
    print(f"[CCC] LR pairs={report['n_lr_pairs']} EOC_deep={report['n_eoc_deep']} Stromal_highU={report['n_stromal_highU']}")
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="HGSOC nactpair protocol analysis")
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    checkpoint = _resolve_ckpt(args.checkpoint_dir)
    out_root = init_protocol_outdir(checkpoint / "analysis_protocol_HGSOC")
    print(f"Checkpoint: {checkpoint}")
    print("Loading AnnData...")
    adata = load_annotated_adata(HGSOC_PROFILE, str(checkpoint))
    print(adata.obs["annotation"].value_counts())

    r1 = step_eoc_attractor_basin(adata, checkpoint, out_root)
    r2 = step_stromal_icaf_climb(adata, out_root)
    r3 = step_targeted_ccc(adata, r1, out_root)
    write_json(
        {
            "checkpoint": str(checkpoint),
            "eoc_attractor": r1,
            "stromal_icaf": r2,
            "cell_communication": r3,
        },
        result_path(out_root, "analysis_summary.json"),
    )
    write_output_file_index(
        out_root,
        [
            ("figures/eoc_attractor_Waddington_landscape_naive.png",
             "EOC UMAP potential landscape (chemo-naive); deep-valley attractor outlined"),
            ("figures/eoc_attractor_Waddington_landscape_postNACT.png",
             "EOC UMAP potential landscape (post-NACT); deep-valley attractor outlined"),
            ("figures/stromal_iCAF_centroid_climb_naive_to_postNACT.png",
             "Stromal fibroblast centroid shift naive→post-NACT with ΔU / iCAF metrics"),
            ("figures/ccc_LR_paracrine_feedforward_top.png",
             "Top ligand–receptor scores (deep-valley EOC ↔ high-U Stromal)"),
            ("eoc_attractor_deep_valley_barcodes.csv",
             "Cell barcodes of EOC deep-valley attractor cells"),
            ("eoc_attractor_deep_valley_DEG.csv",
             "DEG: deep-valley EOC vs other EOC"),
            ("eoc_attractor_deep_valley_enrichment.csv",
             "Pathway enrichment of deep-valley EOC DEG"),
            ("eoc_attractor_summary.json",
             "EOC attractor basin summary metrics"),
            ("stromal_iCAF_climb_metrics.csv",
             "Stromal centroid displacement, potential climb, iCAF scores"),
            ("stromal_iCAF_climb_DEG.csv",
             "DEG: post-NACT vs naive stromal fibroblasts"),
            ("stromal_iCAF_summary.json",
             "Stromal iCAF climb summary + upstream ligand candidates"),
            ("ccc_EOC_deep_valley_barcodes.csv",
             "Sender/receiver barcodes: deep-valley EOC used in LR scoring"),
            ("ccc_Stromal_highU_barcodes.csv",
             "Sender/receiver barcodes: high-U stromal used in LR scoring"),
            ("ccc_LR_Stromal_to_EOC.csv",
             "Ligand–receptor scores: Stromal_highU → EOC_deep"),
            ("ccc_LR_EOC_to_Stromal.csv",
             "Ligand–receptor scores: EOC_deep → Stromal_highU"),
            ("ccc_LR_paracrine_feedforward.csv",
             "Combined bidirectional LR score table"),
            ("ccc_summary.json",
             "Cell–cell communication summary"),
            ("analysis_summary.json",
             "Top-level summary aggregating all HGSOC protocol steps"),
            ("OUTPUT_FILE_INDEX.md",
             "This file: human-readable description of every output"),
        ],
    )
    print(f"Done. Results under: {out_root}")


if __name__ == "__main__":
    main()
