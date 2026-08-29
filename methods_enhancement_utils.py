"""
Shared helpers for Methods-paper robustness / benchmarking / in-silico perturbation.

Flat output layout under each checkpoint:
  methods_enhancement/
  ├── figures/                 # all PNGs
  ├── *.csv / *.json / *.md    # all tabular / summary results at root
  └── OUTPUT_FILE_INDEX.md     # filename legend
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis_protocol_utils import ensure_dir


def methods_outdir(checkpoint_dir: Path, name: str = "methods_enhancement") -> Path:
    """
    Return the flat methods_enhancement root (creates figures/ only).

    Nested names like ``methods_enhancement/sota_benchmark`` are ignored —
    everything writes under ``<ckpt>/methods_enhancement/``.
    """
    root_name = str(name).split("/")[0] if name else "methods_enhancement"
    if root_name != "methods_enhancement":
        root_name = "methods_enhancement"
    root = ensure_dir(Path(checkpoint_dir) / root_name)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    return root


def result_path(root: Path, filename: str) -> Path:
    """Path for a tabular/JSON result at methods_enhancement root."""
    return ensure_dir(root) / Path(filename).name


def fig_path(root: Path, filename: str) -> Path:
    """Path for a figure under methods_enhancement/figures/."""
    d = ensure_dir(root) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    name = Path(filename).name
    if not name.lower().endswith((".png", ".pdf", ".svg")):
        name = f"{name}.png"
    return d / name


def cache_path(root: Path, *parts: str) -> Path:
    """Optional download cache under methods_enhancement/_cache/ (not for paper outputs)."""
    d = ensure_dir(Path(root) / "_cache")
    for p in parts:
        d = ensure_dir(d / p)
    return d


# Canonical file descriptions used when writing / merging OUTPUT_FILE_INDEX.md
FILE_DESCRIPTIONS: Dict[str, str] = {
    # SOTA benchmark
    "sota_benchmark_{ds}.csv": "MomentumNetwork vs scVelo/kNN proxy: trajectory–time PCC and sink strength",
    "figures/sota_benchmark_{ds}.png": "Bar chart: trajectory–time PCC and sink strength by method",
    # Physical consistency / negative controls
    "physical_consistency_{ds}.csv": "U vs −log KDE correlations for real vs label-shuffle controls",
    "physical_consistency_{ds}_summary.json": "Summary: real Spearman, null median, collapse ratio",
    "figures/physical_consistency_{ds}.png": "Boxplot: Spearman(U, −log KDE) under time/stage/condition shuffles",
    # Downsampling
    "downsampling_{ds}.csv": "U–KDE consistency at 10/30/50/80/100% cell subsample fractions",
    "figures/downsampling_{ds}.png": "Stability curve: U–KDE correlation vs subsample fraction",
    # In-silico KO
    "in_silico_KO_Cpeb1_SNIIC_track.csv": "Hamiltonian 24h→2d rollout: SNIIC1/SNIIC2 module scores (WT vs Cpeb1 KO)",
    "figures/in_silico_KO_Cpeb1_SNIIC.png": "SNIIC module tracks over simulated time (WT vs Cpeb1 KO)",
    "in_silico_KO_Lgals3_fate_flux.csv": "Krt8+ fate-branch flux metrics (WT vs Lgals3 KO)",
    "figures/in_silico_KO_Lgals3_fate_flux.png": "Fibro vs AT1 sink/alignment bars (WT vs Lgals3 KO)",
    # Physical retrain null controls
    "physical_retrain_controls_{ds}.csv": "Matched temporal/pairing null retrain metrics (canonical e500·mc5000·bs128)",
    "physical_retrain_controls_{ds}_summary.json": "Real vs null Spearman/PCC after matched null retrain; collapse ratios",
    "figures/physical_retrain_controls_{ds}.png": "Boxplot: U₀–KDE / PCC after matched temporal null vs real",
    # Combined SOTA
    "sota_benchmark_all_datasets.csv": "MomentumNetwork vs scVelo proxy on all datasets",
    "sota_benchmark_PCC_summary.csv": "Pivot: trajectory–time PCC by dataset and method",
    # Clinical
    "pcs_clinical_summary.csv": "PCS validation summary (TCGA-OV KM + AOCS expression)",
    "pcs_TCGA_OV_scores.csv": "Per-sample PCS on TCGA-OV",
    "pcs_AOCS_scores.csv": "Per-sample PCS on AOCS/ICGC GEO cohort",
    "figures/KM_OS_TCGA_OV_PCS.png": "Kaplan–Meier OS: high vs low PCS (TCGA-OV)",
    "figures/KM_PFS_TCGA_OV_PCS.png": "Kaplan–Meier PFS: high vs low PCS (TCGA-OV)",
    # IPF cross-validation
    "ipf_switch_gene_expression.csv": "Mouse Krt8→Fibro switch gene means in human IPF vs control epithelium",
    "ipf_switch_genes_requested.csv": "Switch genes requested when IPF scRNA was unavailable",
    "ipf_cross_validation_summary.csv": "IPF cross-validation status and enrichment summary",
    "figures/ipf_switch_gene_barplot.png": "Bar plot: mouse switch genes in human IPF epithelium",
    # Suite
    "methods_enhancement_suite_summary.json": "Combined JSON summary across all methods-enhancement pillars",
}


def describe_file(filename: str, dataset_key: str = "") -> str:
    """Resolve a human description for a methods_enhancement file."""
    name = filename.replace("\\", "/")
    ds = dataset_key or ""
    for pattern, desc in FILE_DESCRIPTIONS.items():
        filled = pattern.replace("{ds}", ds) if ds else pattern
        if name == filled or name.endswith("/" + filled.split("/")[-1]) or name == filled.split("/")[-1]:
            return desc.replace("{ds}", ds)
        # wildcard match without ds: sota_benchmark_GSE155622.csv vs sota_benchmark_{ds}.csv
        if "{ds}" in pattern:
            prefix = pattern.split("{ds}")[0]
            suffix = pattern.split("{ds}")[-1]
            base = Path(name).name
            if base.startswith(Path(prefix).name if "/" not in prefix else prefix.split("/")[-1]) and base.endswith(suffix):
                return desc.replace("{ds}", ds or "dataset")
    return "Methods enhancement output"


def write_output_file_index(
    root: Path,
    *,
    dataset_key: str = "",
    extra: Optional[Sequence[Tuple[str, str]]] = None,
) -> Path:
    """
    Scan methods_enhancement (figures + root results) and write OUTPUT_FILE_INDEX.md.
    """
    root = Path(root)
    entries: List[Tuple[str, str]] = []
    for p in sorted(root.glob("*")):
        if p.name.startswith("_") or p.name in ("figures", "OUTPUT_FILE_INDEX.md", "METHODS_ENHANCEMENT_INDEX.md"):
            continue
        if p.is_file():
            entries.append((p.name, describe_file(p.name, dataset_key)))
    fig_dir = root / "figures"
    if fig_dir.is_dir():
        for p in sorted(fig_dir.glob("*")):
            if p.is_file():
                rel = f"figures/{p.name}"
                entries.append((rel, describe_file(rel, dataset_key)))
    if extra:
        known = {e[0] for e in entries}
        for rel, desc in extra:
            if rel not in known:
                entries.append((rel, desc))

    lines = [
        f"# Methods enhancement output index — `{dataset_key or root.parent.name}`",
        "",
        "Layout:",
        "",
        "- `figures/` — all PNG figures",
        "- root — all CSV / JSON result tables",
        "",
        "| File | Description |",
        "|---|---|",
    ]
    for rel, desc in entries:
        lines.append(f"| `{rel}` | {desc} |")
    lines.append("")
    out = root / "OUTPUT_FILE_INDEX.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_methods_index(root: Path, entries: Sequence[Tuple[str, str]]) -> None:
    """Backward-compatible alias: merge into OUTPUT_FILE_INDEX.md."""
    write_output_file_index(root, extra=entries)


def trajectory_time_pcc(
    pseudotime: np.ndarray,
    time: np.ndarray,
    *,
    min_cells: int = 50,
) -> float:
    """Pearson correlation between path pseudotime and biological time (higher = better alignment)."""
    from scipy.stats import pearsonr

    pt = np.asarray(pseudotime, dtype=float)
    tt = np.asarray(time, dtype=float)
    m = np.isfinite(pt) & np.isfinite(tt)
    if int(m.sum()) < min_cells:
        return float("nan")
    r, _ = pearsonr(pt[m], tt[m])
    return float(r)


def sink_convergence_score(
    coords: np.ndarray,
    velocities: np.ndarray,
    query_points: np.ndarray,
    *,
    n_neighbors: int = 30,
    grid_points: int = 80,
) -> Dict[str, float]:
    """Wrapper around VectorFieldAnalyzer.sink_strength_at_points."""
    from VectorField import VectorFieldAnalyzer

    vfa = VectorFieldAnalyzer(n_neighbors=n_neighbors, grid_points=grid_points)
    vfa.compute_vector_field_dynamo_style(coords, velocities, n_dims=2)
    return vfa.sink_strength_at_points(query_points)


def potential_kde_consistency(
    adata,
    *,
    latent_key: str = "X_latent",
    potential_key: str = "potential",
    checkpoint_dir: str = None,
) -> Dict[str, float]:
    """U vs -log KDE(z) Pearson/Spearman (physical consistency)."""
    if latent_key not in adata.obsm and checkpoint_dir:
        try:
            from latent_embeddings import ensure_latent_embeddings

            ensure_latent_embeddings(adata, checkpoint_dir=str(checkpoint_dir), warn=False)
        except Exception:
            pass
    from train_model import compute_potential_neglogp_consistency

    if latent_key in adata.obsm and potential_key in adata.obs.columns:
        z = np.asarray(adata.obsm[latent_key], dtype=float)
        u = np.asarray(adata.obs[potential_key], dtype=float)
        finite = np.isfinite(z).all(axis=1) & np.isfinite(u)
        if finite.sum() < 50:
            return {"pearson_U_neglogKDE": 0.0, "spearman_U_neglogKDE": 0.0}
        if finite.sum() < adata.n_obs:
            sub = adata[finite].copy()
            pearson, spearman, _, _ = compute_potential_neglogp_consistency(
                sub, latent_key=latent_key, potential_key=potential_key
            )
            return {"pearson_U_neglogKDE": pearson, "spearman_U_neglogKDE": spearman}

    pearson, spearman, _, _ = compute_potential_neglogp_consistency(
        adata, latent_key=latent_key, potential_key=potential_key
    )
    return {"pearson_U_neglogKDE": pearson, "spearman_U_neglogKDE": spearman}


def subsample_adata(adata, fraction: float, seed: int = 42, checkpoint_dir: str = None):
    """Random cell subsample (for stability / downsampling tests)."""
    import scanpy as sc

    frac = float(np.clip(fraction, 0.01, 1.0))
    if frac >= 0.999:
        out = adata.copy()
    else:
        n = max(50, int(adata.n_obs * frac))
        out = sc.pp.subsample(adata, n_obs=n, random_state=seed, copy=True)
    if checkpoint_dir:
        try:
            from latent_embeddings import merge_latent_embeddings_from_checkpoint

            merge_latent_embeddings_from_checkpoint(out, checkpoint_dir)
        except Exception:
            pass
    return out
