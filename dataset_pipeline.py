"""
Shared configuration and helpers for train / analysis / LAP scripts.

All three stages for each dataset must use the same DatasetSpec so checkpoint
paths, data paths, and model outputs stay aligned.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
import scanpy as sc

from plot_utils import ensure_figure_dir, setup_scanpy_figdir
from train_model import Config, INERTIA_MOMENTUM_MIX, RECON_MMD_MIX_RATIO


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    checkpoint_prefix: str
    data_paths: Tuple[str, ...]
    n_top_genes: int = 3000
    epochs: int = 1000
    batch_size: int = 512
    lr: float = 1e-5
    lambda_reg: float = 0.01
    lambda_hjb: Optional[float] = None
    lambda_recon: float = 0.1
    lambda_pseudo: float = 0.05  # legacy; mapped to lambda_momentum when unset
    lambda_residual: float = 0.01  # legacy; unused in unified Hamiltonian loss
    lambda_momentum: Optional[float] = None
    early_stop_metric: str = "pcc_then_mse"
    checkpoint_metric: str = "pcc_then_mse"
    checkpoint_pcc_tie_epsilon: float = 0.005
    early_stop_patience: int = 200
    early_stop_min_delta: float = 1e-4
    val_mode: str = "cells"
    val_ratio: float = 0.15
    profile_tag: Optional[str] = None
    hidden_dim: Optional[int] = 512
    n_layers: Optional[int] = 3
    dropout: Optional[float] = 0.15
    min_genes: int = 400
    use_hvg: bool = True
    temporal_group_key: str = "stage"
    train_time: Tuple[float, ...] = (0.0, 1.0, 2.0)
    start_time: float = 0.0


HGSOC = DatasetSpec(
    name="HGSOC",
    checkpoint_prefix="HGSOC",
    data_paths=(
        "./HGSOC.h5ad",
        "./HGSOC/HGSOC.h5ad",
        "../Transformer_julie/HGSOC.h5ad",
    ),
    # Temporal axis for NACT-paired training: treatment-naive (0) → post-NACT (1).
    # Clinical stage is a conditioner, not time (see apply_hgsoc_nact_paired_options).
    temporal_group_key="treatment_phase",
    train_time=(0, 1),
    start_time=0.0,
    epochs=3000,
    batch_size=512,
    lr=1e-5,
    lambda_hjb=0.05,
    lambda_recon=0.1,
    lambda_pseudo=0.05,
    early_stop_metric="pcc_then_mse",
    checkpoint_metric="pcc_then_mse",
    early_stop_patience=250,
    val_mode="patients",
    val_ratio=0.18,
    profile_tag="nactpair",
)

GSE155622 = DatasetSpec(
    name="GSE155622",
    checkpoint_prefix="GSE155622",
    data_paths=(
        "./GSE155622_raw_UMI_counts_3.h5ad",
        "../Transformer_julie/GSE155622_raw_UMI_counts_3.h5ad",
    ),
    temporal_group_key="condition",
    train_time=(0, 0.25, 1, 2, 7, 14),
    epochs=3000,
    batch_size=384,
    lr=8e-6,
    lambda_hjb=0.05,
    lambda_recon=0.1,
    lambda_pseudo=0.03,
    early_stop_metric="pcc_then_mse",
    checkpoint_metric="pcc_then_mse",
    early_stop_patience=150,
)

GSE225948_BRAIN_B1 = DatasetSpec(
    name="GSE225948_Brain",
    checkpoint_prefix="GSE225948_Brain",
    data_paths=(
        "./GSE225948_Brain.h5ad",
        "../Transformer_julie/GSE225948_Brain.h5ad",
    ),
    temporal_group_key="treatment",
    train_time=(0, 2, 14),
    epochs=3000,
    batch_size=512,
    lr=1e-5,
    lambda_hjb=0.05,
    lambda_recon=0.1,
    lambda_pseudo=0.05,
    lambda_residual=0.01,
    early_stop_metric="pcc_then_mse",
    checkpoint_metric="pcc_then_mse",
    early_stop_patience=200,
)

GSE141259 = DatasetSpec(
    name="GSE141259",
    checkpoint_prefix="GSE141259",
    data_paths=(
        "./GSE141259_WholeLung.h5ad",
        "../Transformer_julie/GSE141259_WholeLung.h5ad",
    ),
    temporal_group_key="stage",
    train_time=(0, 3, 7, 10, 14, 21, 28),
    epochs=5000,
    batch_size=512,
    lr=1e-5,
    lambda_hjb=0.05,
    lambda_recon=0.1,
    lambda_pseudo=0.05,
    early_stop_metric="pcc_then_mse",
    checkpoint_metric="pcc_then_mse",
    early_stop_patience=250,
    val_mode="time_extrapolate",
    val_ratio=0.12,
    profile_tag="valholdD28",
)

# Legacy GSE141259 profile (hold-out D21+D28, val_ratio=0.15) for comparison.
GSE141259_LEGACY = replace(
    GSE141259,
    val_ratio=0.15,
    early_stop_patience=200,
    profile_tag="legacy",
)

# Tuning profiles (valhold D28). Run via: python run_training.py --dataset GSE141259 --profile <name> --loss-normalization
GSE141259_L2_155622HP = replace(
    GSE141259,
    batch_size=384,
    lr=8e-6,
    lambda_pseudo=0.03,
    profile_tag="valholdD28_l2",
)
GSE141259_L3_RECON = replace(
    GSE141259,
    lambda_recon=0.15,
    lambda_hjb=0.06,
    profile_tag="valholdD28_l3",
)
# l3_recon derivatives used by --profile l3_l2pseudo / l3_capacity / l3_genes5000.
GSE141259_L3_L2PSEUDO = replace(
    GSE141259_L3_RECON,
    lambda_pseudo=0.03,
    profile_tag="valholdD28_l3pseudo",
)
GSE141259_L3_CAPACITY = replace(
    GSE141259_L3_RECON,
    batch_size=384,
    hidden_dim=384,
    profile_tag="valholdD28_l3cap",
)
GSE141259_L3_RECON018 = replace(
    GSE141259_L3_RECON,
    lambda_recon=0.18,
    profile_tag="valholdD28_l3r18",
)
GSE141259_L3_RECON012 = replace(
    GSE141259_L3_RECON,
    lambda_recon=0.12,
    profile_tag="valholdD28_l3r12",
)
GSE141259_L3_HJB008 = replace(
    GSE141259_L3_RECON,
    lambda_hjb=0.08,
    profile_tag="valholdD28_l3h08",
)
# Paper lung profile (5000 HVGs). Adopted weights also override energy/recon/latent
# (see DATA_AND_CHECKPOINTS.md). Do not add --total-drift-hjb for the paper run.
GSE141259_L3_GENES5000 = replace(
    GSE141259_L3_RECON,
    n_top_genes=5000,
    hidden_dim=512,
    n_layers=3,
    dropout=0.15,
    early_stop_metric="pcc_then_mse",
    checkpoint_metric="pcc_then_mse",
    profile_tag="valD28",
)
GSE141259_RECOMMENDED_PROFILE = "l3_genes5000"
GSE141259_L4_CAPACITY = replace(
    GSE141259,
    batch_size=384,
    hidden_dim=384,
    profile_tag="valholdD28_l4",
)

GSE141259_PROFILES = {
    "default": GSE141259,
    "valholdD28": GSE141259,
    "legacy": GSE141259_LEGACY,
    "l2_155622hp": GSE141259_L2_155622HP,
    "l3_recon": GSE141259_L3_RECON,
    "l3_l2pseudo": GSE141259_L3_L2PSEUDO,
    "l3_capacity": GSE141259_L3_CAPACITY,
    "l3_recon018": GSE141259_L3_RECON018,
    "l3_recon012": GSE141259_L3_RECON012,
    "l3_hjb008": GSE141259_L3_HJB008,
    "l3_genes5000": GSE141259_L3_GENES5000,
    "l4_capacity": GSE141259_L4_CAPACITY,
}

# Default Brain training profile (B1: balanced lr / recon / hjb).
GSE225948_BRAIN = GSE225948_BRAIN_B1

# Smoke / ablation profiles for Brain hyperparameter comparison.
GSE225948_BRAIN_PROFILES = {
    "B0": replace(
        GSE225948_BRAIN_B1,
        lr=5e-6,
        batch_size=256,
        lambda_hjb=0.03,
        lambda_recon=0.15,
        lambda_pseudo=0.08,
        profile_tag="brainB0",
    ),
    "B1": GSE225948_BRAIN_B1,
    "B2": replace(
        GSE225948_BRAIN_B1,
        lr=8e-6,
        batch_size=384,
        lambda_hjb=0.06,
        lambda_recon=0.1,
        lambda_pseudo=0.05,
        lambda_residual=0.005,
        profile_tag="brainB2",
    ),
}


@dataclass(frozen=True)
class PlotStyle:
    """Dataset-specific UMAP colors and obs keys (aligned with *_analysis.py)."""

    celltype_key: str
    stage_key: str
    celltype_palette: Dict[str, str]
    stage_palette: Dict[str, str]
    pseudotime_cmap: str = "magma"
    potential_cmap: str = "RdYlBu_r"
    n_pcs: int = 50
    n_neighbors: int = 15
    umap_min_dist: float = 0.3
    umap_spread: Optional[float] = None
    scale_max_value: Optional[float] = None
    harmony_key: Optional[str] = None
    harmony_basis: str = "X_harmony"
    harmony_max_iter: int = 20
    neighbors_use_rep: Optional[str] = None


HGSOC_CELLTYPE_PALETTE = {
    "EOC": "#F9637C",
    "Immune": "#8FB943",
    "Stromal": "#78B9D2",
}
HGSOC_STAGE_PALETTE = {"IIIC": "#D2F1DC", "IVA": "#518463", "IVB": "#254750"}

GSE155622_CELLTYPE_PALETTE = {
    "Fibroblast": "#6E77A2",
    "Immune": "#8FB943",
    "Neuron": "#941B14",
    "RBC": "#04686B",
    "Satellite": "#F5CF36",
    "Schwann": "#F46E49",
    "VEC": "#708AB9",
    "VECC": "#1663A9",
    "VSMC": "#597C8B",
}
GSE155622_STAGE_PALETTE = {
    "Control": "#7c9559",
    "SNI 6h": "#90ac7c",
    "SNI 24h": "#bdbb55",
    "SNI 2d": "#deb956",
    "SNI 7d": "#9dbdd2",
    "SNI 14d": "#779ebd",
}

GSE225948_CELLTYPE_PALETTE = {
    "BAM": "#8386a8",
    "Bc": "#f5cf36",
    "DC": "#c88324",
    "EC": "#046868",
    "Epi": "#85982c",
    "Gran": "#78b9d2",
    "MC": "#1663a9",
    "MaC": "#7ca7ae",
    "MdC": "#939650",
    "Mg": "#b9181a",
    "NK": "#594335",
    "OD": "#fab378",
    "Tc": "#a95465",
}
GSE225948_STAGE_PALETTE = {"Sham": "#D2F1DC", "D02": "#518463", "D14": "#254750"}

GSE141259_STAGE_DAYS = (0, 3, 7, 10, 14, 21, 28)
GSE141259_STAGE_LABELS = tuple(f"D{int(d)}" for d in GSE141259_STAGE_DAYS)
GSE141259_STAGE_TIME = {int(d): float(d) for d in GSE141259_STAGE_DAYS}
GSE141259_STAGE_LABEL_TIME = {
    label: GSE141259_STAGE_TIME[day] for day, label in zip(GSE141259_STAGE_DAYS, GSE141259_STAGE_LABELS)
}
GSE141259_STAGE_PALETTE = {
    "D0": "#D2F1DC",
    "D3": "#A8D5BA",
    "D7": "#518463",
    "D10": "#6B9E78",
    "D14": "#779ebd",
    "D21": "#4A7C94",
    "D28": "#254750",
}
GSE141259_CELLTYPE_PALETTE = {
    "macrophages": "#939650",
    "alv_epithelium": "#78b9d2",
    "dendritic_cells": "#c88324",
    "T_cells": "#a95465",
    "endothelial_cells": "#046868",
    "monocytes": "#1663a9",
    "B_cells": "#f5cf36",
    "club_cells": "#85982c",
    "ciliated_cells": "#708ab9",
    "fibroblasts": "#6E77A2",
    "mesothelia_cells": "#597C8B",
    "granulocytes": "#b9181a",
    "goblet_cells": "#EABFC3",
    "NK_cells": "#594335",
    "smooth_muscle_cells": "#8386a8",
    "unassigned": "#BDBDBD",
}

PLOT_STYLES: Dict[str, PlotStyle] = {
    "HGSOC": PlotStyle(
        celltype_key="annotation",
        stage_key="stage",
        celltype_palette=HGSOC_CELLTYPE_PALETTE,
        stage_palette=HGSOC_STAGE_PALETTE,
    ),
    "GSE155622": PlotStyle(
        celltype_key="annotation",
        stage_key="condition",
        celltype_palette=GSE155622_CELLTYPE_PALETTE,
        stage_palette=GSE155622_STAGE_PALETTE,
    ),
    "GSE225948_Brain": PlotStyle(
        celltype_key="parent",
        stage_key="treatment",
        celltype_palette=GSE225948_CELLTYPE_PALETTE,
        stage_palette=GSE225948_STAGE_PALETTE,
        n_pcs=40,
        n_neighbors=15,
        umap_min_dist=0.5,
        umap_spread=2.0,
        scale_max_value=10,
    ),
    "GSE141259": PlotStyle(
        celltype_key="metacelltype",
        stage_key="stage",
        celltype_palette=GSE141259_CELLTYPE_PALETTE,
        stage_palette=GSE141259_STAGE_PALETTE,
        n_pcs=50,
        n_neighbors=15,
        umap_min_dist=0.4,
    ),
}


def harmony_integrate_adata(
    adata,
    key: str,
    *,
    basis: str = "X_pca",
    adjusted_basis: str = "X_harmony",
    **kwargs,
):
    """Run Harmony and store corrected PCs in ``adata.obsm`` (harmonypy 1.x/2.x shapes)."""
    import numpy as np

    try:
        import harmonypy
    except ImportError as e:
        msg = "\nplease install harmonypy:\n\n\tpip install harmonypy"
        raise ImportError(msg) from e

    X = adata.obsm[basis].astype(np.float64)
    harmony_out = harmonypy.run_harmony(X, adata.obs, key, **kwargs)
    Z = np.asarray(harmony_out.Z_corr)
    if Z.shape[0] == adata.n_obs:
        adata.obsm[adjusted_basis] = Z
    elif Z.shape[1] == adata.n_obs:
        adata.obsm[adjusted_basis] = Z.T
    else:
        raise ValueError(
            f"Harmony output shape {Z.shape} incompatible with n_obs={adata.n_obs}"
        )
    return adata


def compute_training_umap(adata, plot_style=None, config=None, *, min_dist=None, force=False):
    """PCA (+ optional Harmony) + neighbors + UMAP for training / overview figures."""
    import scanpy as sc

    if force:
        for key in ("X_umap", "X_pca", "X_harmony"):
            adata.obsm.pop(key, None)
        adata.uns.pop("neighbors", None)
        adata.uns.pop("umap", None)
        adata.obsp.pop("distances", None)
        adata.obsp.pop("connectivities", None)

    if not force and "X_umap" in adata.obsm:
        return adata

    style = plot_style
    if style is None and config is not None:
        style = getattr(config, "plot_style", None)
    n_pcs = style.n_pcs if style else 50
    n_neighbors = style.n_neighbors if style else 15
    md = min_dist if min_dist is not None else (style.umap_min_dist if style else 0.3)
    n_comps = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)

    if style and style.scale_max_value is not None:
        sc.pp.scale(adata, max_value=style.scale_max_value)

    sc.tl.pca(adata, n_comps=n_comps)

    use_rep = None
    if style and style.harmony_key and style.harmony_key in adata.obs:
        harmony_integrate_adata(
            adata,
            style.harmony_key,
            adjusted_basis=style.harmony_basis,
            max_iter_harmony=style.harmony_max_iter,
        )
        use_rep = style.neighbors_use_rep or style.harmony_basis
    elif style and style.neighbors_use_rep:
        use_rep = style.neighbors_use_rep

    if use_rep:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    umap_kw = {"min_dist": md}
    if style and style.umap_spread is not None:
        umap_kw["spread"] = style.umap_spread
    sc.tl.umap(adata, **umap_kw)
    return adata


def get_plot_style(spec: DatasetSpec) -> PlotStyle:
    if spec.name not in PLOT_STYLES:
        raise KeyError(f"No PlotStyle registered for dataset {spec.name!r}")
    return PLOT_STYLES[spec.name]


def resolve_data_path(spec: DatasetSpec) -> str:
    for rel in spec.data_paths:
        path = PROJECT_ROOT / rel
        if path.exists():
            return str(path)
    return str(PROJECT_ROOT / spec.data_paths[0])


def checkpoint_dir(
    spec: DatasetSpec,
    epochs: int | None = None,
    *,
    lambda_hjb: float | None = None,
    lambda_recon: float | None = None,
) -> str:
    """Legacy checkpoint path (no val-mode / ablation tags). Used by validation/LAP scripts."""
    gene_part = spec.n_top_genes if spec.use_hvg else spec.min_genes
    hjb = (
        float(lambda_hjb)
        if lambda_hjb is not None
        else (spec.lambda_hjb if spec.lambda_hjb is not None else spec.lambda_reg)
    )
    recon = float(lambda_recon) if lambda_recon is not None else float(spec.lambda_recon)
    tag = f"_{spec.profile_tag}" if spec.profile_tag else ""
    train_epochs = int(epochs) if epochs is not None else int(spec.epochs)
    folder = (
        f"{spec.checkpoint_prefix}_checkpoints_"
        f"{gene_part}_{train_epochs}_{spec.batch_size}_{hjb:g}_recon{recon:g}{tag}"
    )
    return str(PROJECT_ROOT / folder)


def _tag_float(value: float) -> str:
    """Compact float tag safe for checkpoint directory names."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def training_checkpoint_tags(config: Config) -> list[str]:
    """Compact suffix tags for training runs (loss weights + key protocol flags).

    Naming pattern (adopted checkpoints):
      ``lossnorm_qp_d{λ_density}_z{λ_latent}_k{λ_kinetic}_ld{λ_lat_disp}``
    plus short protocol tags when needed (``timeX``, ``patients``, ablations).
    """
    tags: list[str] = []
    val_mode = getattr(config, "val_mode", "cells")
    if val_mode == "time_extrapolate":
        tags.append("timeX")
    elif val_mode == "patients":
        # Redundant with profile/pair nactpair on HGSOC; keep only when unpaired.
        pair_keys_early = getattr(config, "pair_group_keys", None) or []
        if "patient_id" not in pair_keys_early:
            tags.append("patients")
    elif val_mode != "cells":
        tags.append(f"valmode-{val_mode}")
    if getattr(config, "use_loss_normalization", False):
        tags.append("lossnorm")
    if getattr(config, "use_total_drift_hjb", False):
        tags.append("totaldrift")
    if getattr(config, "potential_time_mode", "time_varying") == "quasi_stationary":
        tags.append("qp")
    # Default publication recon is mse_mmd; omit from folder names. Tag only deviations.
    recon_mode = getattr(config, "reconstruction_mode", "mse_mmd")
    if recon_mode not in ("mse", "mse_mmd"):
        tags.append(f"recon-{recon_mode}")
    lambda_density = float(getattr(config, "lambda_density", 0.0) or 0.0)
    if lambda_density > 0:
        tags.append(f"d{_tag_float(lambda_density)}")
    lambda_delta = float(getattr(config, "lambda_delta", 0.0) or 0.0)
    if lambda_delta > 0:
        tags.append(f"delta{_tag_float(lambda_delta)}")
    lambda_direction = float(getattr(config, "lambda_direction", 0.0) or 0.0)
    if lambda_direction > 0:
        tags.append(f"dir{_tag_float(lambda_direction)}")
    lambda_latent = float(getattr(config, "lambda_latent", 0.0) or 0.0)
    if lambda_latent > 0:
        tags.append(f"z{_tag_float(lambda_latent)}")
    lambda_kinetic = float(getattr(config, "lambda_kinetic", 0.0) or 0.0)
    if lambda_kinetic > 0:
        tags.append(f"k{_tag_float(lambda_kinetic)}")
    lambda_lat_disp = float(getattr(config, "lambda_lat_disp", 0.0) or 0.0)
    if lambda_lat_disp > 0:
        tags.append(f"ld{_tag_float(lambda_lat_disp)}")
    if bool(getattr(config, "latent_disp_use_mag_ratio", False)):
        tags.append("otrati")
    if bool(getattr(config, "latent_disp_fullpop_ot", False)):
        tags.append("fullpot")
    if bool(getattr(config, "latent_disp_exclude_ema", False)):
        tags.append("nodispnorm")
    ablation = getattr(config, "ablation_flags", None) or {}
    if ablation.get("no_hjb"):
        tags.append("ablate-nohjb")
    if ablation.get("no_residual_drift"):
        tags.append("ablate-noresidual")
    if ablation.get("no_cell_type_embedding"):
        tags.append("ablate-notypeembed")
    if ablation.get("adjacent_only"):
        tags.append("ablate-adjacentonly")
    if ablation.get("no_state_momentum"):
        tags.append("ablate-nostatemom")
    pair_keys = getattr(config, "pair_group_keys", None) or []
    # Avoid duplicating profile_tag "nactpair" already present in the base folder name.
    if pair_keys and "patient_id" not in pair_keys:
        tags.append("grouppair")
    elif pair_keys and "patient_id" in pair_keys:
        # Only add when profile_tag did not already encode nactpair.
        pass
    if getattr(config, "use_stage_embedding", False):
        # Stage conditioning is default for nact-pair HGSOC; omit to keep names short.
        pass
    return tags


def build_training_checkpoint_dir(spec: DatasetSpec, config: Config) -> str:
    """Checkpoint directory for a specific training run (may include experiment tags)."""
    base = checkpoint_dir(
        spec,
        epochs=getattr(config, "epochs", None),
        lambda_hjb=float(config.lambda_hjb),
        lambda_recon=float(config.lambda_recon),
    )
    tags = training_checkpoint_tags(config)
    if not tags:
        return base
    return f"{base}_{'_'.join(tags)}"


def get_dataset_interpretation_metadata(dataset_name: str) -> Dict[str, str]:
    """Conservative interpretation guidance saved with training_summary.json."""
    key = dataset_name.replace("GSE225948_", "GSE225948_")
    if key in ("HGSOC",):
        return {
            "interpretation_level": "moderate",
            "potential_claim_strength": "weak_to_moderate",
        }
    if key in ("GSE155622",):
        return {
            "interpretation_level": "moderate",
            "potential_claim_strength": "moderate_global_weak_local",
        }
    if key in ("GSE225948_Brain", "Brain"):
        return {
            "interpretation_level": "exploratory",
            "potential_claim_strength": "weak",
        }
    if key in ("GSE141259",):
        return {
            "interpretation_level": "moderate",
            "potential_claim_strength": "moderate_global_weak_local",
        }
    return {
        "interpretation_level": "moderate",
        "potential_claim_strength": "weak",
    }


def save_training_summary(
    save_dir: str,
    *,
    dataset: str,
    profile: Optional[str],
    spec: DatasetSpec,
    config: Config,
    metrics: Dict[str, Any],
    val_split_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write training_summary.json under the checkpoint directory."""
    interp = get_dataset_interpretation_metadata(dataset)
    payload: Dict[str, Any] = {
        "dataset": dataset,
        "profile": profile or (spec.profile_tag if spec.profile_tag else None),
        "val_mode": config.val_mode,
        "val_split": val_split_info or {},
        "checkpoint_metric": getattr(config, "checkpoint_metric", "pcc"),
        "early_stop_metric": getattr(config, "early_stop_metric", "pcc"),
        "best_epoch": int(metrics.get("best_epoch", 0) or 0),
        "best_val_pcc": float(metrics.get("best_val_pcc", float("nan"))),
        "best_val_mse": float(metrics.get("best_val_mse", float("nan"))),
        "interpretation_level": interp["interpretation_level"],
        "potential_claim_strength": interp["potential_claim_strength"],
        "ablation_flags": dict(getattr(config, "ablation_flags", {}) or {}),
        "use_loss_normalization": bool(getattr(config, "use_loss_normalization", False)),
        "use_adjacent_transitions": bool(config.use_adjacent_transitions),
        "use_anchor_transitions": bool(config.use_anchor_transitions),
        "use_residual_drift": bool(getattr(config, "use_residual_drift", True)),
        "residual_drift_mode": getattr(config, "residual_drift_mode", "velocity"),
        "momentum_loss_type": getattr(config, "momentum_loss_type", "velocity"),
        "use_cell_type_embedding": bool(getattr(config, "use_cell_type_embedding", True)),
        "use_state_momentum": bool(getattr(config, "use_state_momentum", True)),
        "momentum_parameterization": (
            "state_dependent_p_theta(z,t)"
            if bool(getattr(config, "use_state_momentum", True))
            else "global_ema_batch_mean"
        ),
        "potential_time_mode": getattr(config, "potential_time_mode", "time_varying"),
        "potential_time_correction_scale": float(
            getattr(config, "potential_time_correction_scale", 0.1)
        ),
        "use_hamiltonian_flow": bool(getattr(config, "use_hamiltonian_flow", True)),
        "hamiltonian_damping_gamma": float(getattr(config, "hamiltonian_damping_gamma", 0.1)),
        "use_total_drift_hjb": bool(getattr(config, "use_total_drift_hjb", False)),
        "lambda_hjb": float(config.lambda_hjb),
        "lambda_hjb_note": "checkpoint alias for lambda_energy",
        "lambda_energy": float(config.lambda_hjb),
        "lambda_recon": float(config.lambda_recon),
        "recon_mmd_mix_ratio": RECON_MMD_MIX_RATIO,
        "inertia_momentum_mix": INERTIA_MOMENTUM_MIX,
        "lambda_latent": float(getattr(config, "lambda_latent", 0.0) or 0.0),
        "lambda_kinetic": float(getattr(config, "lambda_kinetic", 0.0) or 0.0),
        "lambda_lat_disp": float(getattr(config, "lambda_lat_disp", 0.0) or 0.0),
        "latent_consistency_note": (
            "lambda_latent: OT( integrate(encode(x_t))@t+1 , encode(x_{t+1}).detach() ); "
            "lambda_kinetic: INERTIA_MOMENTUM_MIX * L_momentum + L_kinetic (fixed mix); "
            "lambda_lat_disp: OT-coupled per-cell latent displacement matching"
        ),
        "reconstruction_mode": getattr(config, "reconstruction_mode", "mse_mmd"),
        "loss_formulation": (
            "L_OT + lambda_recon * L_recon + lambda_energy * L_energy "
            "+ lambda_kinetic * (INERTIA_MOMENTUM_MIX * L_momentum + L_kinetic) "
            "+ lambda_latent * L_latent + lambda_lat_disp * L_lat_disp "
            "+ optional lambda_density/delta/direction; "
            "mse_mmd: L_recon = lambda_recon * (L_pair_mse + RECON_MMD_MIX_RATIO * L_MMD)"
        ),
        "L_energy": "|dU/dt + 0.5 ||grad U||^2|",
        "L_energy_total_drift": "|dU/dt + 0.5 ||-grad U + r_theta||^2|",
        "L_momentum": (
            "velocity: ||p_final - (z_pred - z_curr)/dt||^2; "
            "force: ||(p_final - p_init)/dt + grad U + gamma*p_init - r_force||^2"
        ),
        "lambda_delta": float(getattr(config, "lambda_delta", 0.0) or 0.0),
        "lambda_direction": float(getattr(config, "lambda_direction", 0.0) or 0.0),
        "lambda_density": float(getattr(config, "lambda_density", 0.0) or 0.0),
        "use_density_regularization": bool(getattr(config, "use_density_regularization", False)),
        "density_align_stationary": bool(getattr(config, "density_align_stationary", True)),
        "density_use_latent_batch": bool(getattr(config, "density_use_latent_batch", True)),
        "density_within_cell_type": bool(getattr(config, "density_within_cell_type", True)),
        "lambda_residual_balance": float(getattr(config, "lambda_residual_balance", 0.0) or 0.0),
        "residual_ratio_target": float(getattr(config, "residual_ratio_target", 0.55) or 0.55),
        "homeostasis_ref_time": getattr(config, "homeostasis_ref_time", None),
        "compute_plasticity_scores": bool(getattr(config, "compute_plasticity_scores", True)),
        "pair_group_keys": list(getattr(config, "pair_group_keys", None) or []),
        "patient_key": getattr(config, "patient_key", "patient_id"),
        "stage_cond_key": getattr(config, "stage_cond_key", None),
        "use_stage_embedding": bool(getattr(config, "use_stage_embedding", False)),
        "n_stages": getattr(config, "n_stages", None),
        "density_basis": getattr(config, "density_basis", "X_pca"),
        "density_n_pcs": int(getattr(config, "density_n_pcs", 20)),
        "density_bandwidth": getattr(config, "density_bandwidth", None),
        "checkpoint_dir": save_dir,
    }
    out = Path(save_dir) / "training_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return out


def resolve_checkpoint_root(save_dir: str | Path) -> Path:
    """Return checkpoint root even if save_dir points at validation/ or figures/."""
    p = Path(save_dir).expanduser().resolve()
    if p.name == "validation":
        return p.parent
    return p


def setup_validation_layout(save_dir: str | Path) -> Dict[str, Path]:
    """Create validation/{tables,figures,reports} under checkpoint root (no double nesting)."""
    p = Path(save_dir).expanduser().resolve()
    if p.name == "validation":
        base = p
    else:
        base = resolve_checkpoint_root(save_dir) / "validation"
    paths = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "reports": base / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


# Adopted landscape checkpoints (must match output_file/_adopted.py).
GSE141259_RECOMMENDED_CHECKPOINT_REL = (
    "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)

RECOMMENDED_CHECKPOINT_DIRS: Dict[str, str] = {
    "HGSOC": (
        "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
    ),
    "GSE155622": (
        "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
    ),
    "GSE225948_Brain": "GSE225948_Brain_checkpoints_3000_3000_512_0.05_recon0.1",
    "GSE141259": (
        "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
    ),
}

# Default analysis scopes per dataset (main-9 primary for GSE155622; neuron-4 supplementary).
RECOMMENDED_ANALYSIS_SCOPES: Dict[str, Dict[str, Any]] = {
    "HGSOC": {
        "cell_types": ["EOC", "Immune", "Stromal"],
        "scorecard": "final_validation_scorecard.csv",
    },
    "GSE155622": {
        "primary": {
            "label": "main-9",
            "cell_type_column": "celltype",
            "cell_types": [
                "Fibroblast",
                "Immune",
                "Neuron",
                "RBC",
                "Satellite",
                "Schwann",
                "VEC",
                "VECC",
                "VSMC",
            ],
            "scorecard": "final_validation_scorecard_main9.csv",
        },
        "supplementary": {
            "label": "neuron-4",
            "cell_type_column": "neuron_subtype",
            "cell_types": ["Myelinated", "SNI-induced", "Non_peptidergic", "Peptidergic"],
            "scorecard": "final_validation_scorecard_neuron4.csv",
        },
    },
    "GSE225948_Brain": {
        "cell_types": ["BAM", "Bc", "DC", "EC", "Gran", "MdC", "Mg", "NK", "Tc"],
        "scorecard": "final_validation_scorecard.csv",
    },
    "GSE141259": {
        "cell_types": list(
            (
                "macrophages",
                "alv_epithelium",
                "dendritic_cells",
                "T_cells",
                "endothelial_cells",
                "monocytes",
                "B_cells",
                "club_cells",
                "ciliated_cells",
                "fibroblasts",
                "mesothelia_cells",
                "granulocytes",
                "goblet_cells",
                "NK_cells",
                "smooth_muscle_cells",
            )
        ),
        "scorecard": "final_disease_remodeling_scorecard.csv",
    },
}

# Backward-compatible alias (scripts may still import PHASE2_CHECKPOINT_DIRS).
PHASE2_CHECKPOINT_DIRS = RECOMMENDED_CHECKPOINT_DIRS


def get_save_dir(spec: DatasetSpec) -> str:
    """Return the adopted checkpoint for known datasets; else legacy path."""
    rel = RECOMMENDED_CHECKPOINT_DIRS.get(spec.name)
    if rel:
        return str((PROJECT_ROOT / rel).resolve())
    return checkpoint_dir(spec)


def resolve_checkpoint_dir(spec: DatasetSpec, override: str | None = None) -> str:
    """Resolve checkpoint root; optional override may be absolute or project-relative."""
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())
    return get_save_dir(spec)


def recommended_checkpoint_dir(dataset_name: str) -> str:
    """Absolute path to the adopted checkpoint for a dataset key."""
    rel = RECOMMENDED_CHECKPOINT_DIRS[dataset_name]
    return str((PROJECT_ROOT / rel).resolve())


def phase2_checkpoint_dir(dataset_name: str) -> str:
    """Backward-compatible alias for recommended_checkpoint_dir."""
    return recommended_checkpoint_dir(dataset_name)


def setup_analysis_dirs(save_dir: str) -> str:
    figure_dir = ensure_figure_dir(save_dir)
    setup_scanpy_figdir(save_dir)
    return figure_dir


def apply_recommended_landscape_options(config: Config) -> Config:
    """Quasi-stationary potential + density alignment + interpretation-friendly defaults."""
    config.potential_time_mode = "quasi_stationary"
    config.potential_time_correction_scale = 0.05
    config.use_density_regularization = True
    config.lambda_density = 0.01
    config.density_align_stationary = True
    config.density_use_latent_batch = True
    config.density_within_cell_type = True
    config.density_knn_k = 10
    config.lambda_residual_balance = 0.01
    config.residual_ratio_target = 0.55
    config.compute_plasticity_scores = True
    config.homeostasis_ref_time = getattr(config, "homeostasis_ref_time", None)
    return config


def apply_hamiltonian_loss_options(config: Config) -> Config:
    """Four-term objective: L_OT(z)+λ_ae L_ae+λ_d L_density+λ_H L_H."""
    config.loss_recipe = "hamiltonian"
    config.use_loss_normalization = False
    config.lambda_recon = 0.0
    config.lambda_hjb = 0.0
    config.lambda_reg = 0.0
    config.lambda_latent = 0.0
    config.lambda_lat_disp = 0.0
    config.lambda_residual_balance = 0.0
    config.lambda_kinetic = 0.0
    config.lambda_delta = 0.0
    config.lambda_direction = 0.0
    config.lambda_ae = 0.1
    config.lambda_density = 0.1
    config.lambda_hamiltonian = 0.1
    config.use_density_regularization = True
    config.density_align_stationary = True
    config.density_use_latent_batch = True
    config.use_residual_drift = False
    config.residual_drift_mode = "none"
    config.latent_disp_detach_potential = False
    config.reconstruction_mode = "mse"
    config.checkpoint_metric = "loss"
    config.early_stop_metric = "loss"
    return config


def apply_recommended_latent_flow_options(config: Config) -> Config:
    """Latent-flow stabilization, OT displacement supervision, and mixed recon loss."""
    config.lambda_latent = 0.5
    config.lambda_kinetic = 0.2
    config.kinetic_terminal_beta = 2.0
    config.lambda_lat_disp = 1.0
    config.latent_disp_ot_coupling = True
    config.latent_disp_ot_blur = 0.05
    config.latent_disp_detach_potential = True
    config.reconstruction_mode = "mse_mmd"
    config.lambda_recon = 0.01
    config.early_stop_metric = "pcc_then_mse"
    config.checkpoint_metric = "pcc_then_mse"
    config.early_stop_patience = 200
    return config


def apply_recommended_capacity_options(config: Config, spec: DatasetSpec) -> Config:
    """Wider/deeper autoencoder defaults for high-dimensional gene inputs."""
    if getattr(spec, "hidden_dim", None) is not None:
        config.hidden_dim = int(spec.hidden_dim)
    if getattr(spec, "n_layers", None) is not None:
        config.n_layers = int(spec.n_layers)
    if getattr(spec, "dropout", None) is not None:
        config.dropout = float(spec.dropout)
    return config


def apply_train_config(spec: DatasetSpec) -> Config:
    config = Config()
    config.use_hvg = spec.use_hvg
    config.epochs = spec.epochs
    config.n_top_genes = spec.n_top_genes
    config.min_genes = spec.min_genes
    config.lr = spec.lr
    config.batch_size = spec.batch_size
    config.residual_drift_mode = "velocity"
    config.momentum_loss_type = "velocity"
    apply_recommended_latent_flow_options(config)
    apply_recommended_landscape_options(config)
    apply_recommended_capacity_options(config, spec)
    config.lambda_delta = 0.0
    config.lambda_direction = 0.0
    config.use_loss_normalization = True
    hjb = spec.lambda_hjb if spec.lambda_hjb is not None else 0.02
    config.lambda_hjb = hjb
    config.lambda_reg = hjb
    # spec.lambda_recon is used for checkpoint directory naming only (see checkpoint_dir).
    config.use_hamiltonian_flow = True
    config.hamiltonian_damping_gamma = getattr(config, "hamiltonian_damping_gamma", 0.1)
    config.momentum_ema = getattr(config, "momentum_ema", 0.9)
    config.early_stop_metric = spec.early_stop_metric
    config.checkpoint_metric = spec.checkpoint_metric
    config.checkpoint_pcc_tie_epsilon = spec.checkpoint_pcc_tie_epsilon
    config.early_stop_patience = spec.early_stop_patience
    config.early_stop_min_delta = spec.early_stop_min_delta
    config.use_adjacent_transitions = True
    config.use_anchor_transitions = True
    config.val_mode = spec.val_mode
    config.val_ratio = spec.val_ratio
    config.temporal_group_key = spec.temporal_group_key
    config.data_path = resolve_data_path(spec)
    config.start_time = spec.start_time
    config.train_time = list(spec.train_time)
    if spec.name == "HGSOC":
        apply_hgsoc_nact_paired_options(config)
    config.show_figures = False
    config.plot_style = get_plot_style(spec)
    config.use_loss_normalization = getattr(config, "use_loss_normalization", True)
    config.loss_norm_momentum = getattr(config, "loss_norm_momentum", 0.98)
    config.loss_norm_eps = getattr(config, "loss_norm_eps", 1e-8)
    config.use_residual_drift = getattr(config, "use_residual_drift", True)
    config.use_cell_type_embedding = getattr(config, "use_cell_type_embedding", True)
    config.use_state_momentum = getattr(config, "use_state_momentum", True)
    config.potential_time_mode = getattr(config, "potential_time_mode", "quasi_stationary")
    config.potential_time_correction_scale = getattr(
        config, "potential_time_correction_scale", 0.05
    )
    config.ablation_flags = dict(getattr(config, "ablation_flags", None) or {})
    return config


def apply_gse141259_conservative_train_options(config: Config) -> Config:
    """Apply final conservative GSE141259 training flags (Hamiltonian flow + density)."""
    config.use_loss_normalization = True
    config.use_hamiltonian_flow = True
    config.use_density_regularization = True
    config.lambda_density = 0.01
    config.density_basis = "X_pca"
    config.density_n_pcs = 20
    config.early_stop_metric = "pcc_then_mse"
    config.checkpoint_metric = "pcc_then_mse"
    return config


def gse141259_recommended_checkpoint_rel() -> str:
    """
    Relative checkpoint dir for GSE141259 conservative training (profile l3_genes5000).

    Override at runtime via ``resolve_checkpoint_dir(..., checkpoint_dir=...)`` when missing.
    """
    config = apply_train_config(GSE141259_L3_GENES5000)
    apply_gse141259_conservative_train_options(config)
    return Path(build_training_checkpoint_dir(GSE141259_L3_GENES5000, config)).name


def format_train_config_summary(spec: DatasetSpec, config: Config) -> str:
    """Human-readable training hyperparameters for logs."""
    hjb = float(config.lambda_hjb)
    profile = f" profile={spec.profile_tag}" if spec.profile_tag else ""
    return (
        f"dataset={spec.name}{profile} | lr={config.lr:g} batch={config.batch_size} | "
        f"hidden_dim={config.hidden_dim} n_layers={config.n_layers} dropout={config.dropout:g} | "
        f"reconstruction_mode={getattr(config, 'reconstruction_mode', 'mse_mmd')} "
        f"lambda_recon={config.lambda_recon:g} recon_mmd_mix={RECON_MMD_MIX_RATIO:g} "
        f"lambda_energy={hjb:g} lambda_kinetic={getattr(config, 'lambda_kinetic', 0.0):g} "
        f"inertia_momentum_mix={INERTIA_MOMENTUM_MIX:g} "
        f"lambda_delta={getattr(config, 'lambda_delta', 0.0):g} "
        f"lambda_direction={getattr(config, 'lambda_direction', 0.0):g} "
        f"lambda_latent={getattr(config, 'lambda_latent', 0.0):g} "
        f"lambda_lat_disp={getattr(config, 'lambda_lat_disp', 0.0):g} "
        f"lambda_density={getattr(config, 'lambda_density', 0.0)} "
        f"density_latent_batch={getattr(config, 'density_use_latent_batch', True)} "
        f"lambda_residual_balance={getattr(config, 'lambda_residual_balance', 0.0)} "
        f"pair_group={getattr(config, 'pair_group_keys', None)} "
        f"stage_embed={getattr(config, 'use_stage_embedding', False)} | "
        f"hamiltonian_flow={getattr(config, 'use_hamiltonian_flow', True)} "
        f"damping_gamma={getattr(config, 'hamiltonian_damping_gamma', 0.1):g} | "
        f"residual_drift_mode={getattr(config, 'residual_drift_mode', 'velocity')} "
        f"momentum_loss_type={getattr(config, 'momentum_loss_type', 'velocity')} "
        f"state_momentum={getattr(config, 'use_state_momentum', True)} "
        f"potential_time_mode={getattr(config, 'potential_time_mode', 'quasi_stationary')} | "
        f"density_basis={getattr(config, 'density_basis', 'X_pca')} "
        f"density_n_pcs={getattr(config, 'density_n_pcs', 20)}"
    )


def merge_checkpoint_obs(
    adata,
    save_dir: str,
    columns: Optional[Sequence[str]] = None,
):
    obs_path = Path(save_dir) / "obs.csv"
    if not obs_path.exists():
        raise FileNotFoundError(
            f"Missing {obs_path}. Run the training script for this dataset first."
        )
    obs = pd.read_csv(obs_path, index_col=0)
    default_cols = (
        "potential",
        "potential_stationary",
        "potential_relative_type",
        "potential_deviation",
        "plasticity_score",
        "stability_score",
        "pseudotime",
        "diffusion_eff",
        "hjb_residual",
        "residual_ratio",
        "annotation",
    )
    use_cols = tuple(columns) if columns else default_cols
    overlap = adata.obs_names.intersection(obs.index)
    if len(overlap) == 0:
        raise ValueError(
            "No overlapping cell barcodes between AnnData and checkpoint obs.csv."
        )
    n_missing = adata.n_obs - len(overlap)
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} cells in AnnData have no checkpoint obs row; "
            "model fields will be NaN for those cells.",
            UserWarning,
            stacklevel=2,
        )
    aligned = obs.reindex(adata.obs_names)
    for col in use_cols:
        if col in aligned.columns:
            adata.obs[col] = aligned[col].values
    return adata


def downstream_umap_preprocess(
    adata,
    n_top_genes: int = 3000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    harmony_key: Optional[str] = None,
    use_rep: Optional[str] = None,
):
    if "log1p" not in adata.uns and float(adata.X.max()) > 50:
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        counts_view = adata.copy()
        sc.pp.highly_variable_genes(counts_view, n_top_genes=n_top_genes, flavor="seurat_v3")
        adata.var["highly_variable"] = counts_view.var["highly_variable"]
        adata = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata)
    else:
        if n_top_genes and n_top_genes < adata.n_vars:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
            adata = adata[:, adata.var.highly_variable].copy()
        if "log1p" not in adata.uns:
            sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
            sc.pp.log1p(adata)

    sc.tl.pca(adata, n_comps=n_pcs)
    if harmony_key is not None:
        harmony_integrate_adata(adata, harmony_key, max_iter_harmony=20)
        use_rep = use_rep or "X_harmony"
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
    sc.tl.umap(adata)
    return adata


def add_umap_3d(adata):
    sc.tl.umap(adata, n_components=3)
    adata.obsm["X_umap_3d"] = adata.obsm["X_umap"].copy()
    sc.tl.umap(adata, n_components=2)
    return adata


HGSOC_PATIENT_STAGE = {
    "EOC1005": "IVA",
    "EOC136": "IVA",
    "EOC153": "IVA",
    "EOC227": "IVA",
    "EOC3": "IVA",
    "EOC349": "IVB",
    "EOC372": "IIIC",
    "EOC443": "IVA",
    "EOC540": "IIIC",
    "EOC733": "IVA",
    "EOC87": "IIIC",
}


HGSOC_TREATMENT_TIME = {
    "treatment-naive": 0.0,
    "treatment_naive": 0.0,
    "pre-NACT": 0.0,
    "pre_NACT": 0.0,
    "naive": 0.0,
    "post-NACT": 1.0,
    "post_NACT": 1.0,
    "post-nact": 1.0,
}


def apply_hgsoc_nact_paired_options(config: Config) -> Config:
    """Patient-paired NACT protocol: time=chemo phase; stage=conditioner; patient hold-out val."""
    config.train_time = [0.0, 1.0]
    config.start_time = 0.0
    config.use_adjacent_transitions = True
    config.use_anchor_transitions = False
    config.pair_group_keys = ["patient_id"]
    config.patient_key = "patient_id"
    config.stage_cond_key = "stage_code"
    config.use_stage_embedding = True
    config.n_stages = 3
    config.val_mode = "patients"
    config.val_ratio = float(getattr(config, "val_ratio", 0.18) or 0.18)
    config.homeostasis_ref_time = 0.0
    config.temporal_group_key = "treatment_phase"
    return config


def prepare_hgsoc_for_training(adata, config: Config):
    """HGSOC for NACT-paired training.

    - ``time``: treatment-naive (0) → post-NACT (1)
    - ``stage`` / ``stage_code``: clinical stage conditioner (IIIC/IVA/IVB), not time
    - pairing: same ``patient_id`` × cell type (configured via ``pair_group_keys``)
    """
    apply_hgsoc_nact_paired_options(config)
    adata.obs["stage"] = (
        adata.obs["patient_id"]
        .map(HGSOC_PATIENT_STAGE)
        .astype("category")
        .cat.set_categories(["IIIC", "IVA", "IVB"], ordered=True)
    )
    adata.obs["stage_code"] = adata.obs["stage"].cat.codes.astype(int)
    if "treatment_phase" not in adata.obs.columns:
        raise KeyError(
            "HGSOC NACT-paired training requires obs['treatment_phase'] "
            "(treatment-naive / post-NACT)."
        )
    phase = adata.obs["treatment_phase"].astype(str)
    time_vals = phase.map(HGSOC_TREATMENT_TIME)
    unknown = sorted(set(phase[time_vals.isna()].unique()))
    if unknown:
        raise ValueError(
            f"Unrecognized treatment_phase values for HGSOC NACT pairing: {unknown}. "
            f"Known: {sorted(HGSOC_TREATMENT_TIME)}"
        )
    adata.obs[config.time_key] = time_vals.astype(float)
    adata.obs["annotation"] = adata.obs["cell_type"].astype("category")
    adata.obs[config.cell_type_key] = adata.obs["cell_type"].astype("category").cat.codes
    config.n_stages = int(adata.obs["stage_code"].max()) + 1
    sort_indices = adata.obs.sort_values(
        by=["patient_id", config.time_key, config.cell_type_key]
    ).index
    adata = adata[sort_indices, :].copy()
    print(
        "HGSOC protocol: NACT-paired | time=treatment_phase "
        f"(n_naive={(adata.obs[config.time_key] == 0).sum()}, "
        f"n_post={(adata.obs[config.time_key] == 1).sum()}) | "
        "stage=conditioner | pair_group=patient_id×cell_type",
        flush=True,
    )
    return adata


def annotate_hgsoc_stage(adata):
    adata.obs["stage"] = (
        adata.obs["patient_id"]
        .map(HGSOC_PATIENT_STAGE)
        .astype("category")
        .cat.set_categories(["IIIC", "IVA", "IVB"], ordered=True)
    )
    return adata


GSE155622_CONDITION_TIME = {
    "Control": 0,
    "SNI 6h": 0.25,
    "SNI 24h": 1,
    "SNI 2d": 2,
    "SNI 7d": 7,
    "SNI 14d": 14,
}

GSE155622_STAGE_ORDER = [
    "Control",
    "SNI 6h",
    "SNI 24h",
    "SNI 2d",
    "SNI 7d",
    "SNI 14d",
]

# adata.obs['celltype'] — nine major DRG compartments (Control → SNI 14d LAP).
GSE155622_MAIN_CELL_TYPES = (
    "Fibroblast",
    "Immune",
    "Neuron",
    "RBC",
    "Satellite",
    "Schwann",
    "VEC",
    "VECC",
    "VSMC",
)

GSE155622_MAIN_CELL_LAP_PATHS = {
    ct: ("Control", "SNI 14d") for ct in GSE155622_MAIN_CELL_TYPES
}

# Default canonical LAP endpoints per neuron_subtype (override with --start/--end on CLI).
GSE155622_NEURON_LAP_PATHS = {
    "Myelinated": ("Control", "SNI 14d"),
    "Peptidergic": ("Control", "SNI 14d"),
    "Non_peptidergic": ("Control", "SNI 14d"),
    "SNI-induced": ("SNI 6h", "SNI 14d"),
}

# *_analysis.py canonical LAP + potential-derivative targets
GSE155622_ANALYSIS_CELL_TYPES = (
    "Myelinated",
    "Peptidergic",
    "Non_peptidergic",
    "SNI-induced",
)

HGSOC_ANALYSIS_CELL_TYPES = ("EOC", "Immune", "Stromal")
# LAP along NACT axis (treatment-naive → post-NACT), not clinical stage.
HGSOC_CELLTYPE_LAP_PATHS = {
    ct: ("treatment-naive", "post-NACT") for ct in HGSOC_ANALYSIS_CELL_TYPES
}

# adata.obs['parent'] — all annotated types in GSE225948_Brain
GSE225948_PARENT_CELL_TYPES = (
    "BAM",
    "Bc",
    "DC",
    "EC",
    "Epi",
    "Gran",
    "MC",
    "MaC",
    "MdC",
    "Mg",
    "NK",
    "OD",
    "Tc",
)

# parent types with proportion >= 0.1% — used by *_analysis.py and *_LAP.py
GSE225948_ANALYSIS_CELL_TYPES = (
    "BAM",
    "Bc",
    "DC",
    "EC",
    "Gran",
    "MdC",
    "Mg",
    "NK",
    "Tc",
)

GSE225948_CELLTYPE_LAP_PATHS = {ct: ("Sham", "D14") for ct in GSE225948_ANALYSIS_CELL_TYPES}

# adata.obs['metacelltype'] — major lung compartments (D0 → D28 LAP).
GSE141259_ANALYSIS_CELL_TYPES = (
    "macrophages",
    "alv_epithelium",
    "dendritic_cells",
    "T_cells",
    "endothelial_cells",
    "monocytes",
    "B_cells",
    "club_cells",
    "ciliated_cells",
    "fibroblasts",
    "mesothelia_cells",
    "granulocytes",
    "goblet_cells",
    "NK_cells",
    "smooth_muscle_cells",
)

GSE141259_CELLTYPE_LAP_PATHS = {
    ct: ("D0", "D28") for ct in GSE141259_ANALYSIS_CELL_TYPES
}

GSE155622_NEURON_SUBTYPE = {
    "Atf3/Fam19a4": "Myelinated",
    "Atf3/Gfra3/Gal": "SNI-induced",
    "Atf3/Mrgprd": "SNI-induced",
    "Atf3/S100b/Gal": "SNI-induced",
    "Mrgpra3": "Non_peptidergic",
    "Mrgpra3/Mrgprb4": "Non_peptidergic",
    "Mrgprd/Gm7271": "Non_peptidergic",
    "Mrgprd/Lpar3": "Non_peptidergic",
    "Nppb": "Non_peptidergic",
    "S100b/Baiap2l1": "Myelinated",
    "S100b/Ntrk3/Gfra1": "Myelinated",
    "S100b/Prokr2": "Myelinated",
    "S100b/Smr2": "Myelinated",
    "S100b/Wnt7a": "Myelinated",
    "Cldn9": "Peptidergic",
    "Th/Fam19a4": "Peptidergic",
    "Zcchc12/Dcn": "Peptidergic",
    "Zcchc12/Rxfp1": "Peptidergic",
    "Zcchc12/Sstr2": "Peptidergic",
    "Zcchc12/Trpm8": "Peptidergic",
    "Fibroblast": "Fibroblast",
    "Immune": "Immune",
    "RBC": "RBC",
    "Satellite": "Satellite",
    "Schwann": "Schwann",
    "VEC": "VEC",
    "VECC": "VECC",
    "VSMC": "VSMC",
}


def prepare_gse155622_for_training(adata, config: Config):
    adata.obs["stage"] = adata.obs["condition"].map(GSE155622_CONDITION_TIME).astype("category")
    adata.obs[config.time_key] = adata.obs["stage"].astype("float")
    adata.obs["annotation"] = adata.obs["celltype"].astype("category")
    adata.obs[config.cell_type_key] = adata.obs["celltype"].astype("category").cat.codes
    sort_indices = adata.obs.sort_values(by=[config.time_key, config.cell_type_key]).index
    return adata[sort_indices, :].copy()


def annotate_gse155622_from_checkpoint(adata):
    if "condition" in adata.obs.columns:
        adata.obs["stage"] = adata.obs["condition"]
    adata.obs["stage"] = pd.Categorical(
        adata.obs["stage"], categories=GSE155622_STAGE_ORDER, ordered=True
    )
    if "celltype_2" in adata.obs.columns:
        adata.obs["neuron_subtype"] = (
            adata.obs["celltype_2"].map(GSE155622_NEURON_SUBTYPE).astype("category")
        )
    return adata


GSE225948_TREATMENT_TIME = {"Sham": 0, "D02": 2, "D14": 14}
GSE225948_TREATMENT_ORDER = ["Sham", "D02", "D14"]


def prepare_gse225948_for_training(adata, config: Config):
    adata.obs[config.time_key] = (
        adata.obs["treatment"]
        .map(GSE225948_TREATMENT_TIME)
        .astype("category")
        .cat.set_categories(list(config.train_time), ordered=True)
    )
    adata.obs["annotation"] = adata.obs["parent"].astype("category")
    adata.obs[config.cell_type_key] = adata.obs["parent"].astype("category").cat.codes
    sort_indices = adata.obs.sort_values(by=[config.time_key, config.cell_type_key]).index
    return adata[sort_indices, :].copy()


def annotate_gse225948(adata):
    adata.obs["treatment"] = pd.Categorical(
        adata.obs["treatment"], categories=GSE225948_TREATMENT_ORDER, ordered=True
    )
    adata.obs["stage"] = adata.obs["treatment"].astype("category")
    if "parent" in adata.obs.columns:
        adata.obs["parent"] = adata.obs["parent"].astype("category")
    return adata


def _gse141259_stage_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).map({d: f"D{d}" for d in GSE141259_STAGE_DAYS})
    text = series.astype(str)
    if text.str.match(r"^\d+$").all():
        return text.astype(int).map({d: f"D{d}" for d in GSE141259_STAGE_DAYS})
    return text


def prepare_gse141259_for_training(adata, config: Config):
    adata = adata.copy()
    if "metacelltype" in adata.obs.columns:
        adata = adata[adata.obs["metacelltype"].astype(str) != "unassigned"].copy()
    adata.obs["stage"] = (
        _gse141259_stage_labels(adata.obs["stage"])
        .astype("category")
        .cat.set_categories(list(GSE141259_STAGE_LABELS), ordered=True)
    )
    adata.obs[config.time_key] = adata.obs["stage"].map(GSE141259_STAGE_LABEL_TIME).astype(float)
    adata.obs["annotation"] = adata.obs["metacelltype"].astype("category")
    adata.obs[config.cell_type_key] = adata.obs["metacelltype"].astype("category").cat.codes
    sort_indices = adata.obs.sort_values(by=[config.time_key, config.cell_type_key]).index
    return adata[sort_indices, :].copy()


def annotate_gse141259(adata):
    adata.obs["stage"] = pd.Categorical(
        _gse141259_stage_labels(adata.obs["stage"]),
        categories=list(GSE141259_STAGE_LABELS),
        ordered=True,
    )
    adata.obs["metacelltype"] = adata.obs["metacelltype"].astype("category")
    return adata


def filter_cell_type(adata, column: str, value: str):
    """Subset AnnData to one cell type; validates column and label."""
    if column not in adata.obs.columns:
        raise KeyError(f"Cell type column {column!r} not in adata.obs")
    mask = adata.obs[column].astype(str) == str(value)
    if mask.sum() == 0:
        available = sorted(adata.obs[column].astype(str).unique())
        raise ValueError(f"Cell type {value!r} not found in {column}. Available: {available}")
    return adata[mask].copy()


def print_cell_type_table(adata, column: str, title: str = "") -> pd.Series:
    if column not in adata.obs.columns:
        raise KeyError(f"Column {column!r} not in adata.obs")
    counts = adata.obs[column].value_counts()
    header = title or f"Cell types ({column})"
    print(f"\n=== {header} ===")
    for ct, n in counts.items():
        print(f"  {ct}: {n}")
    return counts


def parse_cell_type_cli(argv=None):
    """
    Parse --list-cell-types / --cell-type / --cell-type-column from script argv.

    Returns (list_only: bool, cell_type: Optional[str], column: Optional[str]).
    """
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    list_only = "--list-cell-types" in argv
    cell_type = None
    column = None
    for i, arg in enumerate(argv):
        if arg == "--cell-type" and i + 1 < len(argv):
            cell_type = argv[i + 1]
        if arg == "--cell-type-column" and i + 1 < len(argv):
            column = argv[i + 1]
    return list_only, cell_type, column
