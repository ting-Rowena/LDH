"""
Shared helpers for the three-dataset literature-protocol analysis runners.

Provides: knn potential gradients, gene-module scores, fate-endpoint selection,
lightweight ligand–receptor scoring, and common I/O helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Gene modules (literature-defined)
# ---------------------------------------------------------------------------

SNIIC_MODULES = {
    "SNIIC1": ("Atf3", "Gfra3", "Gal"),
    "SNIIC2": ("Atf3", "Mrgprd"),
    "SNIIC3": ("Atf3", "S100b", "Gal"),
}
PAIN_CHANNELS = (
    "Trpv1",
    "Scn9a",
    "Scn10a",
    "Scn11a",
    "TRPV1",
    "SCN9A",
    "SCN10A",
    "SCN11A",
)
# Classical nociceptive voltage-gated Na+ channels (Nav1.7 / Nav1.8 / Nav1.9).
NOCICEPTIVE_SODIUM_CHANNELS = ("Scn9a", "Scn10a", "Scn11a")
# Fig2B plot order: TRP channel + classical Nav triad.
PAIN_CHANNELS_PLOT = ("Trpv1", "Scn9a", "Scn10a", "Scn11a")
SWITCH_FACTORS_155622 = ("Atf3", "Egr1", "Cpeb1", "ATF3", "EGR1", "CPEB1")

# Human HGSOC stress / iCAF modules (case-insensitive match via resolve_genes)
STRESS_ASSOCIATED_STATE = (
    "HIF1A", "DDIT3", "ATF4", "ATF3", "XBP1", "HSPA5", "HSP90B1",
    "SOD2", "NQO1", "GDF15", "CDKN1A", "BNIP3", "VEGFA", "LDHA",
    "SLC2A1", "PGK1", "ENO1", "TP53", "NFKB1", "RELA",
)
ICAF_MARKERS = (
    "IL6", "CXCL12", "LIF", "IL11", "CXCL1", "CXCL2", "CCL2",
    "FGF2", "PDGFRA", "FAP", "ACTA2", "COL1A1", "COL3A1", "MMP2",
    "HAS2", "TGFB1", "TNF", "IL1B",
)

# Curated human LR pairs relevant to tumor–stroma feed-forward loops
CURATED_LR_PAIRS: List[Tuple[str, str]] = [
    ("IL6", "IL6R"), ("IL6", "IL6ST"), ("LIF", "LIFR"), ("LIF", "IL6ST"),
    ("CXCL12", "CXCR4"), ("CXCL12", "ACKR3"), ("CCL2", "CCR2"),
    ("CXCL1", "CXCR2"), ("CXCL2", "CXCR2"), ("IL11", "IL11RA"),
    ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"), ("FGF2", "FGFR1"),
    ("FGF2", "FGFR2"), ("VEGFA", "FLT1"), ("VEGFA", "KDR"),
    ("PDGFA", "PDGFRA"), ("PDGFB", "PDGFRB"), ("TNF", "TNFRSF1A"),
    ("TNF", "TNFRSF1B"), ("IL1B", "IL1R1"), ("WNT5A", "FZD5"),
    ("GAS6", "AXL"), ("EFNA1", "EPHA2"), ("JAG1", "NOTCH1"),
    ("JAG1", "NOTCH2"), ("DLL1", "NOTCH1"), ("ICAM1", "ITGAL"),
    ("FN1", "ITGAV"), ("FN1", "ITGB1"), ("COL1A1", "ITGB1"),
    ("BMP4", "BMPR1A"), ("HGF", "MET"), ("AREG", "EGFR"),
    ("HBEGF", "EGFR"), ("EREG", "EGFR"),
]

P53_NFKB_ALIASES = {
    "p53": ("Trp53", "TP53", "Tp53"),
    "NFKB": ("Nfkb1", "NFKB1", "Rela", "RELA", "Relb", "RELB", "Nfkb2", "NFKB2"),
}


def ensure_dir(path: Path) -> Path:
    """Create ``path`` itself (no nested tables/reports/figures)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_protocol_outdir(root: Path) -> Path:
    """
    Flat protocol layout:

    analysis_protocol_<DATASET>/
      figures/          # all PNG/PDF figures
      *.csv/*.json/*.npy # all tabular / JSON / array results at root
      OUTPUT_FILE_INDEX.md
    """
    root = ensure_dir(root)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    return root


def fig_path(out_root: Path, filename: str) -> Path:
    """Path under ``out_root/figures/``; appends ``.png`` when no extension given."""
    out_root = Path(out_root)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    name = str(filename)
    if not name.lower().endswith((".png", ".pdf", ".svg")):
        name = f"{name}.png"
    return out_root / "figures" / name


def result_path(out_root: Path, filename: str) -> Path:
    """Path for non-figure outputs saved directly under ``out_root``."""
    out_root = ensure_dir(out_root)
    return out_root / str(filename)


def write_output_file_index(out_root: Path, entries: Sequence[Tuple[str, str]]) -> Path:
    """
    Write a human-readable index of output files.

    ``entries`` is a sequence of ``(relative_path, description)``.
    """
    out_root = Path(out_root)
    lines = [
        f"# Output file index — `{out_root.name}`",
        "",
        "Layout: all figures live in `figures/`; all other results live at this directory root.",
        "",
        "| File | Description |",
        "|---|---|",
    ]
    for rel, desc in entries:
        lines.append(f"| `{rel}` | {desc} |")
    lines.append("")
    path = out_root / "OUTPUT_FILE_INDEX.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def resolve_genes(var_names, aliases: Sequence[str]) -> List[str]:
    """Match aliases against var_names with case-insensitive fallback."""
    name_set = {str(g): str(g) for g in var_names}
    lower = {str(g).lower(): str(g) for g in var_names}
    out = []
    for a in aliases:
        if a in name_set:
            out.append(name_set[a])
        elif a.lower() in lower:
            out.append(lower[a.lower()])
    return list(dict.fromkeys(out))


def gene_expression(adata, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        return np.full(adata.n_obs, np.nan)
    col = adata.var_names.get_loc(gene)
    x = adata.X
    if hasattr(x, "toarray"):
        return np.asarray(x[:, col].toarray(), dtype=float).ravel()
    return np.asarray(x[:, col], dtype=float).ravel()


def module_score(adata, genes: Sequence[str], *, obs_key: Optional[str] = None) -> np.ndarray:
    resolved = resolve_genes(adata.var_names, genes)
    if not resolved:
        scores = np.full(adata.n_obs, np.nan)
    else:
        mats = [gene_expression(adata, g) for g in resolved]
        scores = np.nanmean(np.vstack(mats), axis=0)
    if obs_key:
        adata.obs[obs_key] = scores
    return scores


def knn_potential_gradient(
    positions: np.ndarray,
    potential: np.ndarray,
    *,
    n_neighbors: int = 30,
) -> np.ndarray:
    """Local least-squares ∇U on embedding neighbors (fixes VectorField −∇U term)."""
    positions = np.asarray(positions, dtype=float)
    pot = np.asarray(potential, dtype=float).ravel()
    n, d = positions.shape
    k = min(max(n_neighbors, d + 2), max(2, n))
    nbrs = NearestNeighbors(n_neighbors=k).fit(positions)
    _, idx = nbrs.kneighbors(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        neigh = idx[i, 1:] if k > 1 else idx[i]
        if neigh.size == 0:
            continue
        A = positions[neigh] - positions[i]
        b = pot[neigh] - pot[i]
        try:
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            grad[i] = coef
        except Exception:
            continue
    return grad


def select_fate_core_indices(
    positions: np.ndarray,
    labels: np.ndarray,
    state: str,
    *,
    potential: Optional[np.ndarray] = None,
    plasticity: Optional[np.ndarray] = None,
    prefer_high_potential: bool = False,
    prefer_high_plasticity: bool = False,
    core_fraction: float = 0.2,
    min_cells: int = 5,
) -> np.ndarray:
    """Medoid-neighborhood core for a fate label, optionally biased to high U / plasticity."""
    labels = np.asarray(labels).astype(str)
    mask = labels == str(state)
    idx = np.where(mask)[0]
    if idx.size < min_cells:
        return idx
    score = np.zeros(idx.size, dtype=float)
    if potential is not None and prefer_high_potential:
        p = np.asarray(potential, dtype=float)[idx]
        score += (p - np.nanmean(p)) / (np.nanstd(p) + 1e-8)
    if plasticity is not None and prefer_high_plasticity:
        pl = np.asarray(plasticity, dtype=float)[idx]
        score += (pl - np.nanmean(pl)) / (np.nanstd(pl) + 1e-8)
    if prefer_high_potential or prefer_high_plasticity:
        keep_n = max(min_cells, int(np.ceil(core_fraction * idx.size)))
        order = np.argsort(-score)
        idx = idx[order[:keep_n]]
    pos = np.asarray(positions, dtype=float)[idx]
    med = np.median(pos, axis=0)
    d = np.linalg.norm(pos - med[None, :], axis=1)
    keep_n = max(min_cells, int(np.ceil(core_fraction * len(idx))))
    return idx[np.argsort(d)[:keep_n]]


def geodesic_path(start: np.ndarray, end: np.ndarray, n_points: int = 12) -> np.ndarray:
    return np.linspace(np.asarray(start, dtype=float), np.asarray(end, dtype=float), int(n_points))


def bootstrap_geodesic_paths(
    positions: np.ndarray,
    start_core: np.ndarray,
    end_core: np.ndarray,
    *,
    n_bootstrap: int = 15,
    n_points: int = 10,
    seed: int = 0,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    paths = []
    if len(start_core) < 1 or len(end_core) < 1:
        return paths
    for _ in range(int(n_bootstrap)):
        s = int(rng.choice(start_core))
        e = int(rng.choice(end_core))
        paths.append(geodesic_path(positions[s], positions[e], n_points=n_points))
    return paths


def annotate_tf_families(genes: Sequence[str]) -> pd.DataFrame:
    rows = []
    for g in genes:
        families = []
        for fam, aliases in P53_NFKB_ALIASES.items():
            if str(g) in aliases or str(g).lower() in {a.lower() for a in aliases}:
                families.append(fam)
        rows.append({"gene": g, "literature_family": ";".join(families) if families else ""})
    return pd.DataFrame(rows)


def mean_expr(adata, genes: Sequence[str], mask: np.ndarray) -> Dict[str, float]:
    out = {}
    for g in resolve_genes(adata.var_names, genes):
        out[g] = float(np.nanmean(gene_expression(adata, g)[mask]))
    return out


def score_lr_pairs(
    adata_sender,
    adata_receiver,
    pairs: Sequence[Tuple[str, str]] = CURATED_LR_PAIRS,
    *,
    min_expr: float = 0.05,
) -> pd.DataFrame:
    """
    Lightweight ligand–receptor score = mean(L_sender) * mean(R_receiver).

    Does not require CellChat/NicheNet; exports ranks for paracrine feed-forward claims.
    """
    rows = []
    for lig, rec in pairs:
        lg = resolve_genes(adata_sender.var_names, [lig])
        rg = resolve_genes(adata_receiver.var_names, [rec])
        if not lg or not rg:
            continue
        l_mean = float(np.nanmean(gene_expression(adata_sender, lg[0])))
        r_mean = float(np.nanmean(gene_expression(adata_receiver, rg[0])))
        if l_mean < min_expr and r_mean < min_expr:
            continue
        rows.append(
            {
                "ligand": lg[0],
                "receptor": rg[0],
                "ligand_mean_sender": l_mean,
                "receptor_mean_receiver": r_mean,
                "lr_score": l_mean * r_mean,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("lr_score", ascending=False).reset_index(drop=True)


def deep_valley_mask(
    potential: np.ndarray,
    stability: np.ndarray,
    *,
    u_quantile: float = 0.15,
    stability_quantile: float = 0.7,
) -> np.ndarray:
    pot = np.asarray(potential, dtype=float)
    stab = np.asarray(stability, dtype=float)
    u_cut = np.nanquantile(pot, u_quantile)
    s_cut = np.nanquantile(stab, stability_quantile)
    return np.isfinite(pot) & np.isfinite(stab) & (pot <= u_cut) & (stab >= s_cut)


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    r, _ = spearmanr(x[m], y[m])
    return float(r)


def fast_wilcoxon_deg(
    adata,
    *,
    group_key: str,
    group_1: str,
    group_2: str,
    n_top_genes: int = 3000,
    min_cells: int = 10,
) -> pd.DataFrame:
    """
    Fast DEG via scanpy Wilcoxon on HVG (or all genes if fewer).

    Avoids deg_enrichment_workflow.run_deg_comparison's per-gene mean/pct loop
    which is prohibitively slow for full transcriptomes.
    """
    import scanpy as sc

    m1 = adata.obs[group_key].astype(str) == group_1
    m2 = adata.obs[group_key].astype(str) == group_2
    if int(m1.sum()) < min_cells or int(m2.sum()) < min_cells:
        return pd.DataFrame()
    sub = adata[m1 | m2].copy()
    if sub.n_vars > n_top_genes:
        try:
            sc.pp.highly_variable_genes(sub, n_top_genes=n_top_genes, flavor="seurat_v3")
            sub = sub[:, sub.var["highly_variable"]].copy()
        except Exception:
            # Fall back to variance filter
            x = sub.X
            if hasattr(x, "toarray"):
                # sparse variance approx via mean of squares
                mean = np.asarray(x.mean(axis=0)).ravel()
                mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel() if hasattr(x, "multiply") else mean**2
                var = mean_sq - mean**2
            else:
                var = np.asarray(x, dtype=float).var(axis=0)
            keep = np.argsort(-var)[:n_top_genes]
            sub = sub[:, keep].copy()
    sub.obs["_deg_group"] = sub.obs[group_key].astype(str)
    sc.tl.rank_genes_groups(
        sub,
        groupby="_deg_group",
        groups=[group_1],
        reference=group_2,
        method="wilcoxon",
        n_genes=sub.n_vars,
        use_raw=False,
    )
    df = sc.get.rank_genes_groups_df(sub, group=group_1)
    df = df.rename(
        columns={
            "names": "gene",
            "logfoldchanges": "logfoldchange",
            "pvals": "pval",
            "pvals_adj": "pval_adj",
            "scores": "score",
        }
    )
    df["comparison"] = f"{group_1}_vs_{group_2}"
    df["direction"] = np.where(df["logfoldchange"] > 0, "up_in_group_1", "down_in_group_1")
    return df
