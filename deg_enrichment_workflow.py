"""
DEG + pathway enrichment as primary biological interpretation layer.

Pioneer genes are supporting evidence only.
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from sklearn.neighbors import NearestNeighbors

from lap_helpers import clip_path_index, path_nearest_cell_indices
from dataset_pipeline import setup_validation_layout
from plot_utils import TECH_BLUE_CMAP
from region_labels import (
    DEG_REGION_ORDER,
    LAP_REGION_CATEGORIES,
    PATH_PAIRWISE_DEG_COMPARISONS,
    PATH_REGION_HEATMAP_MAX_GENES,
    PATH_REGION_HEATMAP_PER_COMPARISON_CAP,
    REGION_PRIORITY,
    deg_comparison_specs,
    path_region_heatmap_categories,
)
from plot_utils import configure_headless

configure_headless()

try:
    import gseapy as gp

    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False

BIOLOGICAL_FRAMING = (
    "DEG and enrichment analysis provide the primary biological interpretation of "
    "cell-state remodeling and candidate remodeling-region biology. "
    "Pioneer-gene ranking is used only to prioritize transition/remodeling-associated "
    "candidate genes and does not establish causality."
)

DEFAULT_GENE_SETS = [
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
]

ORGANISM_MAP = {
    "Human": "human",
    "Mouse": "mouse",
    "human": "human",
    "mouse": "mouse",
}


def _normalize_organism(organism: str) -> str:
    return ORGANISM_MAP.get(organism, str(organism).lower())

def _slug(value: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(value).strip())


def setup_validation_dirs(save_dir: str) -> Dict[str, Path]:
    return setup_validation_layout(save_dir)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def assign_lap_regions(
    adata,
    analyzer,
    path_result: dict,
    *,
    stage_key: str = "stage",
    potential_key: str = "potential",
    pseudotime_key: str = "pseudotime",
    window_size: int = 5,
    group_key: str = "lap_region",
    neighbors_per_path_point: int = 10,
) -> Tuple[Any, pd.DataFrame]:
    """Label cells by start / transition / end path windows with overlap priority."""
    adata = adata.copy()
    indices, _, _, _ = path_nearest_cell_indices(
        analyzer,
        path_result,
        path_result["start_state"],
        path_result["end_state"],
        stage_key,
    )
    flat_idx = np.asarray(indices.flatten(), dtype=int)
    n_path = len(flat_idx)
    ts_idx = clip_path_index(int(path_result["transition_state_idx"]), n_path)
    path = np.asarray(path_result.get("path_compute", path_result["path"]), dtype=float)
    positions = np.asarray(analyzer.cell_positions_2d, dtype=float)
    nbrs = NearestNeighbors(n_neighbors=min(neighbors_per_path_point, len(positions))).fit(positions)

    region_ranges = {
        "start_basin": list(range(0, min(window_size, n_path))),
        "transition_window": list(
            range(max(0, ts_idx - window_size), min(n_path, ts_idx + window_size + 1))
        ),
        "end_basin": list(range(max(0, n_path - window_size), n_path)),
    }

    cell_region: Dict[str, str] = {str(n): "other" for n in adata.obs_names}
    cell_path_pos: Dict[str, float] = {}
    cell_dist: Dict[str, float] = {}

    for region, path_indices in region_ranges.items():
        pri = REGION_PRIORITY[region]
        for pi in path_indices:
            _, nn_idx = nbrs.kneighbors([path[pi]])
            for cidx in np.asarray(nn_idx[0], dtype=int):
                cid = str(adata.obs_names[cidx])
                dist = float(np.linalg.norm(positions[cidx] - path[pi]))
                cur_pri = REGION_PRIORITY.get(cell_region[cid], 0)
                if pri > cur_pri or (pri == cur_pri and dist < cell_dist.get(cid, np.inf)):
                    cell_region[cid] = region
                    cell_path_pos[cid] = float(pi)
                    cell_dist[cid] = dist

    path_local_categories = ["start_basin", "transition_window", "end_basin", "other"]
    adata.obs[group_key] = pd.Categorical(
        [cell_region[str(n)] for n in adata.obs_names],
        categories=path_local_categories,
    )

    start_state = str(path_result.get("start_state", ""))
    end_state = str(path_result.get("end_state", ""))
    if stage_key in adata.obs.columns:
        for cid in adata.obs_names:
            cid_s = str(cid)
            if cell_region[cid_s] != "other":
                continue
            st = str(adata.obs.loc[cid, stage_key])
            if start_state and st == start_state:
                cell_region[cid_s] = "start_basin"
            elif end_state and st == end_state:
                cell_region[cid_s] = "end_basin"
        adata.obs[group_key] = pd.Categorical(
            [cell_region[str(n)] for n in adata.obs_names],
            categories=path_local_categories,
        )

    rows = []
    for cid in adata.obs_names:
        cid_s = str(cid)
        obs_row = adata.obs.loc[cid]
        rows.append(
            {
                "cell_id": cid_s,
                "lap_region": cell_region[cid_s],
                "stage": str(obs_row[stage_key]) if stage_key in adata.obs.columns else "",
                "pseudotime": float(obs_row[pseudotime_key])
                if pseudotime_key in adata.obs.columns and pd.notna(obs_row[pseudotime_key])
                else np.nan,
                "potential": float(obs_row[potential_key])
                if potential_key in adata.obs.columns and pd.notna(obs_row[potential_key])
                else np.nan,
                "nearest_path_position": cell_path_pos.get(cid_s, np.nan),
                "distance_to_path": cell_dist.get(cid_s, np.nan),
            }
        )
    return adata, pd.DataFrame(rows)


def _lap_region_assignment_table(
    adata,
    *,
    group_key: str = "lap_region",
    stage_key: str = "stage",
    potential_key: str = "potential",
    pseudotime_key: str = "pseudotime",
) -> pd.DataFrame:
    rows = []
    for cid in adata.obs_names:
        cid_s = str(cid)
        obs_row = adata.obs.loc[cid]
        rows.append(
            {
                "cell_id": cid_s,
                "lap_region": str(obs_row[group_key]),
                "stage": str(obs_row[stage_key]) if stage_key in adata.obs.columns else "",
                "pseudotime": float(obs_row[pseudotime_key])
                if pseudotime_key in adata.obs.columns and pd.notna(obs_row[pseudotime_key])
                else np.nan,
                "potential": float(obs_row[potential_key])
                if potential_key in adata.obs.columns and pd.notna(obs_row[potential_key])
                else np.nan,
                "nearest_path_position": np.nan,
                "distance_to_path": np.nan,
            }
        )
    return pd.DataFrame(rows)


def assign_potential_quantile_groups(
    adata,
    potential_key: str = "potential",
    group_key: str = "potential_group",
    lower_q: float = 0.25,
    upper_q: float = 0.75,
) -> Any:
    adata = adata.copy()
    pot = pd.to_numeric(adata.obs[potential_key], errors="coerce")
    lo = float(pot.quantile(lower_q))
    hi = float(pot.quantile(upper_q))
    labels = np.full(adata.n_obs, "middle", dtype=object)
    labels[pot.values <= lo] = "low_potential"
    labels[pot.values >= hi] = "high_potential"
    adata.obs[group_key] = pd.Categorical(
        labels, categories=["low_potential", "middle", "high_potential"]
    )
    return adata


def _gene_expression_stats(adata, cells_mask, gene: str) -> Tuple[float, float]:
    if gene not in adata.var_names:
        return np.nan, np.nan
    idx = adata.var_names.get_loc(gene)
    sub = adata[cells_mask]
    if sub.n_obs == 0:
        return np.nan, np.nan
    x = sub[:, gene].X
    if sparse.issparse(x):
        vals = np.asarray(x.mean(axis=1)).ravel()
        expressed = np.asarray(x.getnnz(axis=1)).ravel() > 0
    else:
        vals = np.asarray(x).ravel() if x.ndim == 1 else np.asarray(x).mean(axis=1).ravel()
        expressed = vals > 0
    return float(np.mean(vals)), float(np.mean(expressed))


def run_deg_comparison(
    adata,
    *,
    comparison: str,
    group_1: str,
    group_2: str,
    group_key: str = "lap_region",
    method: str = "wilcoxon",
    min_cells: int = 10,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Run one DE comparison; return table and optional warning."""
    m1 = adata.obs[group_key].astype(str) == group_1
    m2 = adata.obs[group_key].astype(str) == group_2
    n1, n2 = int(m1.sum()), int(m2.sum())
    if n1 < min_cells or n2 < min_cells:
        return None, f"Skipped {comparison}: {group_1} n={n1}, {group_2} n={n2} (min={min_cells})"

    sub = adata[m1 | m2].copy()
    sub.obs["_deg_group"] = sub.obs[group_key].astype(str)
    try:
        sc.tl.rank_genes_groups(
            sub,
            groupby="_deg_group",
            groups=[group_1],
            reference=group_2,
            method=method,
            n_genes=sub.n_vars,
            use_raw=False,
        )
        raw = sc.get.rank_genes_groups_df(sub, group=group_1)
    except Exception as exc:
        return None, f"DE failed for {comparison}: {exc}"

    if raw.empty:
        return None, f"No DE results for {comparison}"

    warn = None
    if "pvals_adj" not in raw.columns:
        warn = f"pvals_adj missing for {comparison}; using raw pvals"

    rows = []
    seen: set = set()
    for _, r in raw.iterrows():
        gene = str(r["names"])
        if gene in seen:
            continue
        seen.add(gene)
        lfc = float(r.get("logfoldchanges", np.nan))
        direction = "up_in_group_1" if lfc > 0 else "down_in_group_1"
        mean1, pct1 = _gene_expression_stats(sub, sub.obs["_deg_group"] == group_1, str(r["names"]))
        mean2, pct2 = _gene_expression_stats(sub, sub.obs["_deg_group"] == group_2, str(r["names"]))
        rows.append(
            {
                "gene": gene,
                "comparison": comparison,
                "group_1": group_1,
                "group_2": group_2,
                "score": float(r.get("scores", np.nan)),
                "logfoldchange": lfc,
                "pval": float(r.get("pvals", np.nan)),
                "pval_adj": float(r.get("pvals_adj", np.nan))
                if "pvals_adj" in raw.columns
                else np.nan,
                "mean_expr_group_1": mean1,
                "mean_expr_group_2": mean2,
                "pct_expr_group_1": pct1,
                "pct_expr_group_2": pct2,
                "direction": direction,
            }
        )
    return pd.DataFrame(rows), warn


def filter_significant_deg(
    df: pd.DataFrame,
    *,
    padj_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    padj = out["pval_adj"].fillna(out["pval"])
    out = out[(padj < padj_cutoff) & (out["logfoldchange"].abs() > logfc_cutoff)]
    return out


def collect_path_regions_union_genes(
    deg_results: Dict[str, Optional[pd.DataFrame]],
    *,
    padj_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
    max_genes: int = PATH_REGION_HEATMAP_MAX_GENES,
    per_comparison_cap: int = PATH_REGION_HEATMAP_PER_COMPARISON_CAP,
) -> List[str]:
    """Union of significant top genes from the three path pairwise DEG tables."""
    best: Dict[str, Tuple[float, float]] = {}
    for comparison in PATH_PAIRWISE_DEG_COMPARISONS:
        df = deg_results.get(comparison)
        if df is None or df.empty:
            continue
        sig = filter_significant_deg(df, padj_cutoff=padj_cutoff, logfc_cutoff=logfc_cutoff)
        pool = sig if not sig.empty else df
        pool = pool.sort_values("pval_adj", na_position="last").head(per_comparison_cap)
        for _, row in pool.iterrows():
            gene = str(row["gene"])
            padj = float(row["pval_adj"]) if pd.notna(row.get("pval_adj")) else float(row["pval"])
            lfc_abs = abs(float(row["logfoldchange"])) if pd.notna(row.get("logfoldchange")) else 0.0
            if gene not in best or padj < best[gene][0]:
                best[gene] = (padj, lfc_abs)
    ranked = sorted(best.items(), key=lambda item: (item[1][0], -item[1][1]))
    return [gene for gene, _ in ranked[:max_genes]]


def _adata_sorted_for_region_heatmap(
    adata,
    group_key: str,
    pseudotime_key: str,
    *,
    region_categories: Optional[Sequence[str]] = None,
) -> Any:
    """Subset to path-region columns (optional) and order cells by region then pseudotime."""
    adata = adata.copy()
    if region_categories:
        mask = adata.obs[group_key].astype(str).isin([str(r) for r in region_categories])
        adata = adata[mask].copy()
        adata.obs[group_key] = pd.Categorical(
            adata.obs[group_key].astype(str),
            categories=[str(r) for r in region_categories],
        )
    region_order = list(region_categories) if region_categories else list(DEG_REGION_ORDER)
    obs = adata.obs.copy()
    if pseudotime_key in obs.columns:
        obs["_pt"] = pd.to_numeric(obs[pseudotime_key], errors="coerce")
    else:
        obs["_pt"] = 0.0
    obs["_reg"] = obs[group_key].astype(str)
    obs["_ord"] = obs["_reg"].map({r: i for i, r in enumerate(region_order)}).fillna(99)
    order = np.lexsort((obs["_pt"].values, obs["_ord"].values))
    return adata[order].copy()


def _plot_volcano(df: pd.DataFrame, comparison: str, cell_type: str, fig_path: Path, top_n: int = 10) -> None:
    if df is None or df.empty:
        return
    padj = df["pval_adj"].fillna(df["pval"]).clip(lower=1e-300)
    y = -np.log10(padj)
    x = df["logfoldchange"].values
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(x, y, s=8, alpha=0.4, c="gray")
    sig = filter_significant_deg(df)
    if not sig.empty:
        ax.scatter(sig["logfoldchange"], -np.log10(sig["pval_adj"].fillna(sig["pval"]).clip(lower=1e-300)), s=10, c="#c61586", alpha=0.7)
    top = df.assign(_mp=padj, _afc=df["logfoldchange"].abs()).sort_values(["_mp", "_afc"]).head(top_n)
    for _, r in top.iterrows():
        ax.text(r["logfoldchange"], -np.log10(max(r["pval_adj"], r["pval"], 1e-300)), str(r["gene"]), fontsize=5)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10(p-adj)")
    ax.set_title(f"{cell_type}: {comparison}", fontsize=9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_deg_heatmap(
    adata,
    df: pd.DataFrame,
    genes: Sequence[str],
    group_key: str,
    pseudotime_key: str,
    cell_type: str,
    comparison: str,
    fig_path: Path,
) -> None:
    genes = [g for g in genes if g in adata.var_names][:20]
    if not genes:
        return
    sub = _adata_sorted_for_region_heatmap(adata, group_key, pseudotime_key)
    sc.pl.heatmap(
        sub,
        var_names=genes,
        groupby=group_key,
        show=False,
        swap_axes=True,
        figsize=(6, 4),
        cmap=TECH_BLUE_CMAP,
        dendrogram=False,
    )
    plt.suptitle(f"{cell_type}: {comparison} top DEG", fontsize=9, y=1.02)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_path_regions_union_heatmap(
    adata,
    genes: Sequence[str],
    group_key: str,
    pseudotime_key: str,
    cell_type: str,
    fig_path: Path,
    *,
    include_other: bool = True,
) -> None:
    """Summary heatmap: union of pairwise path DEGs across start / transition / end (+ other)."""
    genes = [g for g in genes if g in adata.var_names][:PATH_REGION_HEATMAP_MAX_GENES]
    if not genes:
        return
    display_regions = path_region_heatmap_categories(
        adata.obs[group_key].astype(str).unique(),
        include_other=include_other,
    )
    if len(display_regions) < 2:
        return
    sub = _adata_sorted_for_region_heatmap(
        adata,
        group_key,
        pseudotime_key,
        region_categories=display_regions,
    )
    fig_h = max(4.0, min(8.0, 0.22 * len(genes) + 1.5))
    sc.pl.heatmap(
        sub,
        var_names=genes,
        groupby=group_key,
        show=False,
        swap_axes=True,
        figsize=(6.5, fig_h),
        cmap=TECH_BLUE_CMAP,
        dendrogram=False,
    )
    region_label = " | ".join(display_regions)
    plt.suptitle(
        f"{cell_type}: path regions top DEG ({region_label})\n"
        "(genes = union(transition_vs_start, transition_vs_end, end_vs_start); not 4-group ANOVA)",
        fontsize=8,
        y=1.03,
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_deg_dotplot(
    adata,
    genes: Sequence[str],
    group_key: str,
    cell_type: str,
    comparison: str,
    fig_path: Path,
) -> None:
    genes = [g for g in genes if g in adata.var_names][:20]
    if not genes:
        return
    sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby=group_key,
        show=False,
        standard_scale="var",
        figsize=(5, 4),
    )
    plt.suptitle(f"{cell_type}: {comparison} dotplot", fontsize=9, y=1.02)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_pathway_enrichment(
    genes: Sequence[str],
    *,
    comparison: str,
    direction: str,
    organism: str,
    gene_sets: Sequence[str],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not HAS_GSEAPY:
        return None, "gseapy not installed; skipping enrichment"
    organism = _normalize_organism(organism)
    if not genes:
        return None, f"No genes for enrichment ({comparison}, {direction})"
    try:
        enr = gp.enrichr(
            gene_list=list(dict.fromkeys(str(g) for g in genes)),
            gene_sets=list(gene_sets),
            organism=organism,
            outdir=None,
        )
        res = enr.results.copy()
    except Exception as exc:
        return None, f"Enrichment failed ({comparison}, {direction}): {exc}"

    if res is None or res.empty:
        return None, f"Empty enrichment ({comparison}, {direction})"

    rows = []
    for _, r in res.iterrows():
        rows.append(
            {
                "comparison": comparison,
                "direction": direction,
                "term": r.get("Term", r.get("term", "")),
                "overlap": r.get("Overlap", r.get("overlap", "")),
                "adjusted_p_value": float(r.get("Adjusted P-value", r.get("Adjusted P-value", np.nan))),
                "combined_score": float(r.get("Combined Score", r.get("Combined Score", np.nan)))
                if pd.notna(r.get("Combined Score", np.nan))
                else np.nan,
                "genes": r.get("Genes", r.get("genes", "")),
                "gene_set": r.get("Gene_set", r.get("gene_set", "")),
            }
        )
    return pd.DataFrame(rows), None


def _plot_enrichment_dotplot(df: pd.DataFrame, cell_type: str, comparison: str, fig_path: Path, top_n: int = 15) -> None:
    if df is None or df.empty:
        return
    sub = df.sort_values("adjusted_p_value").head(top_n)
    fig, ax = plt.subplots(figsize=(5, max(3, 0.25 * len(sub))))
    y = -np.log10(sub["adjusted_p_value"].clip(lower=1e-300))
    ax.barh(sub["term"][::-1], y[::-1], color="#DFB199")
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(f"{cell_type}: {comparison} enrichment", fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def integrate_deg_pioneer_candidates(
    deg_results: Dict[str, pd.DataFrame],
    pioneer_df: Optional[pd.DataFrame],
    bootstrap_df: Optional[pd.DataFrame],
    enrichment_results: Dict[str, pd.DataFrame],
    *,
    padj_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
    pioneer_bootstrap_threshold: float = 0.5,
) -> pd.DataFrame:
    sig_flags = {}
    for comp, df in deg_results.items():
        sig = filter_significant_deg(df, padj_cutoff=padj_cutoff, logfc_cutoff=logfc_cutoff) if df is not None else pd.DataFrame()
        sig_flags[comp] = set(sig["gene"].tolist()) if not sig.empty else set()

    pioneer_map = {}
    if pioneer_df is not None and not pioneer_df.empty:
        for _, r in pioneer_df.iterrows():
            pioneer_map[str(r["gene"])] = r

    boot_freq = {}
    if bootstrap_df is not None and not bootstrap_df.empty:
        for _, r in bootstrap_df.iterrows():
            boot_freq[str(r["gene"])] = float(r.get("bootstrap_frequency", 0))

    gene_pathways: Dict[str, List[str]] = defaultdict(list)
    for comp, edf in enrichment_results.items():
        if edf is None or edf.empty:
            continue
        top_terms = edf.sort_values("adjusted_p_value").head(5)["term"].astype(str).tolist()
        for gset in sig_flags.get(comp, set()):
            gene_pathways[gset].extend(top_terms)

    all_genes = set()
    for s in sig_flags.values():
        all_genes |= s
    all_genes |= set(pioneer_map.keys())
    all_genes |= set(boot_freq.keys())

    rows = []
    for gene in sorted(all_genes):
        is_deg_ts = gene in sig_flags.get("transition_vs_start", set())
        is_deg_te = gene in sig_flags.get("transition_vs_end", set())
        is_deg_es = gene in sig_flags.get("end_vs_start", set())
        is_pioneer = gene in pioneer_map
        boot_f = boot_freq.get(gene, np.nan)

        best_comp, best_lfc, best_padj = "", np.nan, np.nan
        for comp, df in deg_results.items():
            if df is None or df.empty:
                continue
            sub = df[df["gene"] == gene]
            if sub.empty:
                continue
            r = sub.iloc[0]
            padj = r["pval_adj"] if pd.notna(r["pval_adj"]) else r["pval"]
            if pd.isna(best_padj) or padj < best_padj:
                best_comp, best_lfc, best_padj = comp, float(r["logfoldchange"]), float(padj)

        transition_deg = is_deg_ts or is_deg_te
        high_conf = (
            transition_deg
            and is_pioneer
            and (np.isnan(boot_f) or boot_f >= pioneer_bootstrap_threshold)
        )
        if high_conf:
            cls = "high_confidence_transition_gene"
        elif transition_deg:
            cls = "DEG_supported_transition_gene"
        elif is_pioneer:
            cls = "pioneer_only_candidate"
        elif is_deg_es:
            cls = "terminal_state_DEG"
        else:
            cls = "exploratory"

        p_row = pioneer_map.get(gene, {})
        rows.append(
            {
                "gene": gene,
                "is_DEG_transition_vs_start": is_deg_ts,
                "is_DEG_transition_vs_end": is_deg_te,
                "is_DEG_end_vs_start": is_deg_es,
                "best_DEG_comparison": best_comp,
                "best_logfoldchange": best_lfc,
                "best_pval_adj": best_padj,
                "is_pioneer": is_pioneer,
                "pioneer_rank": p_row.get("rank", np.nan),
                "pioneer_score": p_row.get("score", np.nan),
                "pioneer_empirical_p_value": p_row.get("empirical_p_value", np.nan),
                "pioneer_bootstrap_frequency": boot_f,
                "is_high_confidence_candidate": high_conf,
                "candidate_class": cls,
                "supporting_pathways": "; ".join(list(dict.fromkeys(gene_pathways.get(gene, [])))[:5]),
            }
        )
    return pd.DataFrame(rows)


def _count_enrichment_terms(enrichment_results: Dict[str, pd.DataFrame], padj_cutoff: float = 0.05) -> int:
    n = 0
    for comp in ("transition_vs_start", "transition_vs_end"):
        df = enrichment_results.get(comp)
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        padj_col = "adjusted_p_value" if "adjusted_p_value" in df.columns else None
        if padj_col:
            n += int((pd.to_numeric(df[padj_col], errors="coerce") < padj_cutoff).sum())
        else:
            n += len(df)
    return n


def summarize_molecular_support(
    deg_results: Dict[str, Optional[pd.DataFrame]],
    enrichment_results: Dict[str, pd.DataFrame],
    *,
    padj_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
    min_transition_deg: int = 10,
    min_enrichment_terms: int = 1,
) -> dict:
    """
    Summarize molecular evidence for remodeling-region support.

    Internal comparison keys remain transition_vs_start / transition_vs_end;
    reports display them as remodeling_vs_start / remodeling_vs_end.
    """
    n_start = len(
        filter_significant_deg(
            deg_results.get("transition_vs_start"),
            padj_cutoff=padj_cutoff,
            logfc_cutoff=logfc_cutoff,
        )
    )
    n_end = len(
        filter_significant_deg(
            deg_results.get("transition_vs_end"),
            padj_cutoff=padj_cutoff,
            logfc_cutoff=logfc_cutoff,
        )
    )
    n_enrichment = _count_enrichment_terms(enrichment_results, padj_cutoff=padj_cutoff)

    transition_vs_start_deg_passed = n_start >= min_transition_deg
    transition_vs_end_deg_passed = n_end >= min_transition_deg
    enrichment_passed = n_enrichment >= min_enrichment_terms

    if (
        n_start >= min_transition_deg
        and n_end >= min_transition_deg
        and n_enrichment >= min_enrichment_terms
    ):
        molecular_support = "strong"
    elif (
        (n_start >= min_transition_deg or n_end >= min_transition_deg)
        and n_enrichment >= min_enrichment_terms
    ):
        molecular_support = "moderate"
    elif (
        n_start >= min_transition_deg
        or n_end >= min_transition_deg
        or n_enrichment >= min_enrichment_terms
    ):
        molecular_support = "weak"
    else:
        molecular_support = "none"

    molecular_score = sum(
        [
            transition_vs_start_deg_passed,
            transition_vs_end_deg_passed,
            enrichment_passed,
        ]
    )

    return {
        "n_deg_remodeling_vs_start": n_start,
        "n_deg_remodeling_vs_end": n_end,
        "n_enrichment_terms": n_enrichment,
        "transition_vs_start_deg_passed": transition_vs_start_deg_passed,
        "transition_vs_end_deg_passed": transition_vs_end_deg_passed,
        "enrichment_passed": enrichment_passed,
        "molecular_score": molecular_score,
        "molecular_support": molecular_support,
    }


def _summarize_enrichment_terms(enrichment_results: Dict[str, pd.DataFrame], comparison: str, top_n: int = 5) -> List[str]:
    df = enrichment_results.get(comparison)
    if df is None or df.empty:
        return []
    return df.sort_values("adjusted_p_value")["term"].astype(str).head(top_n).tolist()


def write_biological_interpretation_report(
    cell_type: str,
    save_dir: str,
    deg_results: Dict[str, Optional[pd.DataFrame]],
    enrichment_results: Dict[str, pd.DataFrame],
    integrated_df: pd.DataFrame,
    warnings: Sequence[str],
) -> str:
    dirs = setup_validation_dirs(save_dir)
    slug = _slug(cell_type)
    path = dirs["reports"] / f"{slug}_biological_interpretation.md"

    lines = [
        f"# Biological Interpretation: {cell_type}",
        "",
        "## Scientific framing",
        "",
        BIOLOGICAL_FRAMING,
        "",
        "- DEG + enrichment = **primary** biological evidence for remodeling-region biology",
        "- Pioneer genes = **optional** transition/remodeling-associated candidate prioritization (not causal proof)",
        "",
        "## 1. DEG Summary (remodeling window comparisons)",
        "",
    ]
    display_names = {
        "transition_vs_start": "remodeling_vs_start",
        "transition_vs_end": "remodeling_vs_end",
        "end_vs_start": "end_vs_start",
    }
    for comp in list(deg_results.keys()):
        df = deg_results.get(comp)
        n_sig = len(filter_significant_deg(df)) if df is not None and not df.empty else 0
        n_total = len(df) if df is not None else 0
        label = display_names.get(comp, comp)
        lines.append(f"- **{label}** ({comp}): {n_sig} significant DEG (of {n_total} tested)")

    lines.extend(["", "## 2. Enrichment Summary", ""])
    for comp in enrichment_results:
        terms = _summarize_enrichment_terms(enrichment_results, comp)
        if terms:
            lines.append(f"- **{comp}**: {', '.join(terms)}")
        else:
            lines.append(f"- **{comp}**: _no significant enrichment_")

    lines.extend(["", "## 3. Pioneer Gene Integration", ""])
    if integrated_df is not None and not integrated_df.empty:
        for cls in (
            "high_confidence_transition_gene",
            "DEG_supported_transition_gene",
            "pioneer_only_candidate",
        ):
            genes = integrated_df.loc[integrated_df["candidate_class"] == cls, "gene"].tolist()
            lines.append(f"- **{cls}**: {', '.join(genes[:15]) if genes else 'none'}")
    else:
        lines.append("_No integrated candidates._")

    trans_terms = _summarize_enrichment_terms(enrichment_results, "transition_vs_start")
    trans_end_terms = _summarize_enrichment_terms(enrichment_results, "transition_vs_end")
    hi_conf = (
        integrated_df.loc[integrated_df["candidate_class"] == "high_confidence_transition_gene", "gene"].tolist()
        if integrated_df is not None and not integrated_df.empty
        else []
    )
    deg_sup = (
        integrated_df.loc[integrated_df["candidate_class"] == "DEG_supported_transition_gene", "gene"].tolist()
        if integrated_df is not None and not integrated_df.empty
        else []
    )

    lines.extend(
        [
            "",
            "## 4. Interpretation",
            "",
            f"For **{cell_type}**, DEG analysis indicates that the candidate remodeling window differs from the "
            f"start basin mainly through **{', '.join(trans_terms[:3]) or 'pathway terms not available'}**. "
            f"Compared with the end basin, the remodeling window shows "
            f"**{', '.join(trans_end_terms[:3]) or 'limited enriched terms'}**, suggesting that these cells "
            f"represent a transient remodeling state rather than a terminal state. "
            f"Pioneer-gene analysis further prioritizes **{', '.join(hi_conf[:5]) or ', '.join(deg_sup[:5]) or 'no high-confidence candidates'}** "
            f"as transition/remodeling-associated candidate genes. Pioneer-gene ranking is used only to "
            f"prioritize transition/remodeling-associated candidate genes and does not establish causality.",
            "",
        ]
    )
    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _read_csv_if_nonempty(path: Optional[str]) -> pd.DataFrame:
    if not path or not Path(path).is_file():
        return pd.DataFrame()
    if Path(path).stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _build_main_interpretation(integrated_df: pd.DataFrame, enrichment_results: Dict[str, pd.DataFrame]) -> str:
    trans_terms = _summarize_enrichment_terms(enrichment_results, "transition_vs_start")
    end_terms = _summarize_enrichment_terms(enrichment_results, "end_vs_start")
    hi = []
    if integrated_df is not None and not integrated_df.empty:
        hi = integrated_df.loc[
            integrated_df["candidate_class"].isin(
                ["high_confidence_transition_gene", "DEG_supported_transition_gene"]
            ),
            "gene",
        ].head(5).tolist()
    return (
        f"Remodeling-window enrichment suggests {', '.join(trans_terms[:3]) or 'remodeling-associated programs'}; "
        f"end basin shows {', '.join(end_terms[:3]) or 'terminal-state programs'}. "
        f"DEG-supported candidates include {', '.join(hi) or 'none identified'}. "
        f"Pioneer genes provide optional candidate prioritization only."
    )


def run_deg_enrichment_workflow(
    adata,
    analyzer,
    canonical_path_result: dict,
    cell_type: str,
    save_dir: str,
    stage_key: str = "stage",
    potential_key: str = "potential",
    pseudotime_key: str = "pseudotime",
    group_key: str = "lap_region",
    window_size: int = 5,
    preassigned_regions: bool = False,
    top_n_genes: int = 50,
    organism: str = "Human",
    run_go: bool = True,
    run_kegg: bool = False,
    run_reactome: bool = False,
    pioneer_gene_table: Optional[str] = None,
    pioneer_bootstrap_table: Optional[str] = None,
    deg_method: str = "wilcoxon",
    deg_min_cells: int = 10,
    deg_logfc_cutoff: float = 0.25,
    deg_padj_cutoff: float = 0.05,
    gene_sets: Optional[Sequence[str]] = None,
) -> dict:
    """Main DEG + enrichment + pioneer integration workflow."""
    dirs = setup_validation_dirs(save_dir)
    slug = _slug(cell_type)
    organism = _normalize_organism(organism)
    result: Dict[str, Any] = {
        "cell_type": cell_type,
        "status": "ok",
        "warnings": [],
        "deg_results": {},
        "enrichment_results": {},
    }

    if preassigned_regions and group_key in adata.obs.columns:
        present = set(adata.obs[group_key].astype(str).unique())
        if "transition_region" in present:
            required = {"start_basin", "transition_region", "end_basin"}
        else:
            required = {"start_basin", "transition_window", "end_basin"}
        missing = required - present
        if missing:
            warnings.warn(f"Missing required region labels: {missing}", UserWarning, stacklevel=2)
            result["warnings"].append(f"Missing required region labels: {sorted(missing)}")
        adata.obs[group_key] = pd.Categorical(
            adata.obs[group_key].astype(str),
            categories=[c for c in LAP_REGION_CATEGORIES if c in present or c in required],
        )
        assign_df = _lap_region_assignment_table(
            adata,
            group_key=group_key,
            stage_key=stage_key,
            potential_key=potential_key,
            pseudotime_key=pseudotime_key,
        )
    else:
        adata, assign_df = assign_lap_regions(
            adata,
            analyzer,
            canonical_path_result,
            stage_key=stage_key,
            potential_key=potential_key,
            pseudotime_key=pseudotime_key,
            window_size=window_size,
            group_key=group_key,
        )
    _save_csv(assign_df, dirs["tables"] / f"{slug}_lap_region_cell_assignments.csv")
    result["lap_region_assignment_table"] = assign_df
    result["lap_region_group_key"] = group_key
    result["lap_region_source"] = "preassigned_consensus" if preassigned_regions else "assigned_in_deg_workflow"
    result["region_counts"] = assign_df["lap_region"].value_counts().to_dict()
    try:
        adata.write_h5ad(dirs["tables"] / f"{slug}_adata_with_lap_region.h5ad")
        result["adata_with_lap_region_path"] = str(dirs["tables"] / f"{slug}_adata_with_lap_region.h5ad")
    except Exception as exc:
        result["warnings"].append(f"Could not save adata with lap_region: {exc}")

    comparison_specs = deg_comparison_specs(adata.obs[group_key].astype(str).unique())
    for comparison, g1, g2 in comparison_specs:
        df, warn = run_deg_comparison(
            adata,
            comparison=comparison,
            group_1=g1,
            group_2=g2,
            group_key=group_key,
            method=deg_method,
            min_cells=deg_min_cells,
        )
        if warn:
            result["warnings"].append(warn)
        result["deg_results"][comparison] = df
        if df is not None and not df.empty:
            _save_csv(df, dirs["tables"] / f"{slug}_DEG_{comparison}.csv")
            _plot_volcano(df, comparison, cell_type, dirs["figures"] / f"{slug}_{comparison}_volcano.png")
            sig = filter_significant_deg(df, padj_cutoff=deg_padj_cutoff, logfc_cutoff=deg_logfc_cutoff)
            top_genes = (
                sig.sort_values("pval_adj", na_position="last").head(20)["gene"].tolist()
                if not sig.empty
                else df.sort_values("pval_adj", na_position="last").head(20)["gene"].tolist()
            )
            try:
                _plot_deg_heatmap(
                    adata, df, top_genes, group_key, pseudotime_key, cell_type, comparison,
                    dirs["figures"] / f"{slug}_{comparison}_top_DEG_heatmap.png",
                )
                _plot_deg_dotplot(
                    adata, top_genes, group_key, cell_type, comparison,
                    dirs["figures"] / f"{slug}_{comparison}_top_DEG_dotplot.png",
                )
            except Exception as exc:
                result["warnings"].append(f"Figure failed for {comparison}: {exc}")

    union_genes = collect_path_regions_union_genes(
        result["deg_results"],
        padj_cutoff=deg_padj_cutoff,
        logfc_cutoff=deg_logfc_cutoff,
    )
    if union_genes:
        result["path_regions_union_genes"] = union_genes
        _save_csv(
            pd.DataFrame({"gene": union_genes}),
            dirs["tables"] / f"{slug}_path_regions_union_DEG_genes.csv",
        )
        try:
            _plot_path_regions_union_heatmap(
                adata,
                union_genes,
                group_key,
                pseudotime_key,
                cell_type,
                dirs["figures"] / f"{slug}_path_regions_top_DEG_heatmap.png",
                include_other=True,
            )
        except Exception as exc:
            result["warnings"].append(f"Path regions union heatmap failed: {exc}")

    adata_pot = assign_potential_quantile_groups(adata, potential_key=potential_key)
    df_hp, warn_hp = run_deg_comparison(
        adata_pot,
        comparison="high_potential_vs_low_potential",
        group_1="high_potential",
        group_2="low_potential",
        group_key="potential_group",
        method=deg_method,
        min_cells=deg_min_cells,
    )
    if warn_hp:
        result["warnings"].append(warn_hp)
    result["deg_results"]["high_potential_vs_low_potential"] = df_hp
    if df_hp is not None and not df_hp.empty:
        _save_csv(df_hp, dirs["tables"] / f"{slug}_DEG_high_potential_vs_low_potential.csv")
        _plot_volcano(
            df_hp, "high_potential_vs_low_potential", cell_type,
            dirs["figures"] / f"{slug}_high_potential_vs_low_potential_volcano.png",
        )

    selected_sets = list(gene_sets or DEFAULT_GENE_SETS)
    if run_go is False:
        selected_sets = [s for s in selected_sets if "GO" not in s]
    if run_reactome is False:
        selected_sets = [s for s in selected_sets if "Reactome" not in s]
    if run_kegg:
        selected_sets.append("KEGG_2021_Human" if organism == "Human" else "KEGG_2021_Mouse")

    if run_go or run_reactome or run_kegg:
        for comparison, df in result["deg_results"].items():
            if df is None or df.empty:
                continue
            sig = filter_significant_deg(df, padj_cutoff=deg_padj_cutoff, logfc_cutoff=deg_logfc_cutoff)
            if sig.empty:
                result["warnings"].append(f"No significant genes for enrichment: {comparison}")
                continue
            up = sig.loc[sig["direction"] == "up_in_group_1", "gene"].tolist()
            down = sig.loc[sig["direction"] == "down_in_group_1", "gene"].tolist()
            enr_parts = []
            for direction, genes in (("up_in_group_1", up), ("down_in_group_1", down)):
                edf, ew = run_pathway_enrichment(
                    genes, comparison=comparison, direction=direction, organism=organism, gene_sets=selected_sets
                )
                if ew:
                    result["warnings"].append(ew)
                if edf is not None and not edf.empty:
                    enr_parts.append(edf)
            if enr_parts:
                combined = pd.concat(enr_parts, ignore_index=True)
                result["enrichment_results"][comparison] = combined
                _save_csv(combined, dirs["tables"] / f"{slug}_{comparison}_GO_enrichment.csv")
                _plot_enrichment_dotplot(
                    combined, cell_type, comparison,
                    dirs["figures"] / f"{slug}_{comparison}_GO_dotplot.png",
                )

    pioneer_df = _read_csv_if_nonempty(pioneer_gene_table)
    bootstrap_df = _read_csv_if_nonempty(pioneer_bootstrap_table)

    integrated = integrate_deg_pioneer_candidates(
        result["deg_results"],
        pioneer_df if not pioneer_df.empty else None,
        bootstrap_df if not bootstrap_df.empty else None,
        result["enrichment_results"],
        padj_cutoff=deg_padj_cutoff,
        logfc_cutoff=deg_logfc_cutoff,
    )
    _save_csv(integrated, dirs["tables"] / f"{slug}_DEG_pioneer_integrated_candidates.csv")
    result["integrated_candidates"] = integrated

    result["n_DEG_transition_vs_start"] = len(
        filter_significant_deg(
            result["deg_results"].get("transition_vs_start"),
            padj_cutoff=deg_padj_cutoff,
            logfc_cutoff=deg_logfc_cutoff,
        )
    )
    result["n_DEG_transition_vs_end"] = len(
        filter_significant_deg(
            result["deg_results"].get("transition_vs_end"),
            padj_cutoff=deg_padj_cutoff,
            logfc_cutoff=deg_logfc_cutoff,
        )
    )
    result["n_DEG_end_vs_start"] = len(
        filter_significant_deg(
            result["deg_results"].get("end_vs_start"),
            padj_cutoff=deg_padj_cutoff,
            logfc_cutoff=deg_logfc_cutoff,
        )
    )
    result["top_transition_enrichment_terms"] = ", ".join(
        _summarize_enrichment_terms(result["enrichment_results"], "transition_vs_start")
    )
    result["top_end_state_enrichment_terms"] = ", ".join(
        _summarize_enrichment_terms(result["enrichment_results"], "end_vs_start")
    )
    hi = integrated.loc[integrated["candidate_class"] == "high_confidence_transition_gene", "gene"].tolist()
    pioneer_only = integrated.loc[integrated["candidate_class"] == "pioneer_only_candidate", "gene"].tolist()
    result["n_high_confidence_transition_genes"] = len(hi)
    result["high_confidence_transition_genes"] = ", ".join(hi[:10])
    result["n_pioneer_only_candidates"] = len(pioneer_only)
    result["main_biological_interpretation"] = _build_main_interpretation(integrated, result["enrichment_results"])

    result["molecular_summary"] = summarize_molecular_support(
        result["deg_results"],
        result["enrichment_results"],
        padj_cutoff=deg_padj_cutoff,
        logfc_cutoff=deg_logfc_cutoff,
        min_transition_deg=10,
        min_enrichment_terms=1,
    )

    result["report_path"] = write_biological_interpretation_report(
        cell_type,
        save_dir,
        result["deg_results"],
        result["enrichment_results"],
        integrated,
        result["warnings"],
    )
    return result


def run_endpoint_deg_fallback(
    adata,
    cell_type: str,
    save_dir: str,
    start_state: str,
    end_state: str,
    stage_key: str = "stage",
    *,
    deg_method: str = "wilcoxon",
    deg_min_cells: int = 10,
    deg_logfc_cutoff: float = 0.25,
    deg_padj_cutoff: float = 0.05,
    organism: str = "Human",
    run_go: bool = True,
    gene_sets: Optional[Sequence[str]] = None,
) -> dict:
    """Endpoint-level DEG/enrichment fallback: end_state vs start_state within cell type."""
    dirs = setup_validation_dirs(save_dir)
    slug = _slug(cell_type)
    organism = _normalize_organism(organism)
    result: Dict[str, Any] = {
        "cell_type": cell_type,
        "status": "ok",
        "comparison": "endpoint_end_vs_start",
        "group_a": end_state,
        "group_b": start_state,
        "endpoint_deg_fallback_used": False,
        "warnings": [],
    }

    if stage_key not in adata.obs.columns:
        result["status"] = "failed"
        result["error"] = f"stage_key {stage_key!r} missing"
        return result

    sub = adata.copy()
    sub.obs["_endpoint_group"] = "other"
    m_start = sub.obs[stage_key].astype(str) == str(start_state)
    m_end = sub.obs[stage_key].astype(str) == str(end_state)
    sub.obs.loc[m_start, "_endpoint_group"] = "start_endpoint"
    sub.obs.loc[m_end, "_endpoint_group"] = "end_endpoint"
    n_start, n_end = int(m_start.sum()), int(m_end.sum())
    result["n_group_a"] = n_end
    result["n_group_b"] = n_start

    df, warn = run_deg_comparison(
        sub,
        comparison="endpoint_end_vs_start",
        group_1="end_endpoint",
        group_2="start_endpoint",
        group_key="_endpoint_group",
        method=deg_method,
        min_cells=deg_min_cells,
    )
    if warn:
        result["warnings"].append(warn)
        result["skip_reason"] = warn
        result["endpoint_deg_count"] = 0
        result["endpoint_enrichment_count"] = 0
        return result

    result["endpoint_deg_fallback_used"] = True
    result["molecular_evidence_source"] = "endpoint_deg_fallback"
    if df is not None and not df.empty:
        _save_csv(df, dirs["tables"] / f"{slug}_endpoint_deg_fallback.csv")
        sig = filter_significant_deg(df, padj_cutoff=deg_padj_cutoff, logfc_cutoff=deg_logfc_cutoff)
        result["endpoint_deg_count"] = int(len(sig))
        result["top_endpoint_deg"] = sig.sort_values("pval_adj").head(20)["gene"].tolist()
    else:
        result["endpoint_deg_count"] = 0
        result["top_endpoint_deg"] = []

    enrichment_results = {}
    if run_go and df is not None and not df.empty:
        sig = filter_significant_deg(df, padj_cutoff=deg_padj_cutoff, logfc_cutoff=deg_logfc_cutoff)
        if not sig.empty:
            up = sig.loc[sig["direction"] == "up_in_group_1", "gene"].tolist()
            selected_sets = list(gene_sets or DEFAULT_GENE_SETS)
            edf, ew = run_pathway_enrichment(
                up, comparison="endpoint_end_vs_start", direction="up_in_group_1",
                organism=organism, gene_sets=selected_sets,
            )
            if ew:
                result["warnings"].append(ew)
            if edf is not None and not edf.empty:
                enrichment_results["endpoint_end_vs_start"] = edf
                _save_csv(edf, dirs["tables"] / f"{slug}_endpoint_enrichment_fallback.csv")
                result["top_endpoint_enrichment"] = _summarize_enrichment_terms(enrichment_results, "endpoint_end_vs_start")
    result["endpoint_enrichment_count"] = _count_enrichment_terms(enrichment_results, padj_cutoff=deg_padj_cutoff)
    result["enrichment_results"] = enrichment_results
    return result
