"""LDH-scRNA transport variant (temporary): lat_disp + latent + recon.

Use this ONLY for Energy / W1 / MMD / OT comparisons against PRESCIENT /
MIOFlow / WOT-inspired. Do NOT replace the publication landscape checkpoints.

Design (aligned with first-order marginal transporters):
  * Dominate with OT-coupled displacement + latent consistency + recon
  * Turn landscape terms OFF (HJB / density / residual-balance / strong kinetic)
  * Select checkpoints by total val loss (proxy for OT-dominated objective;
    native Energy/OT checkpointing is not yet in train_model.py)

Paper / landscape main model should keep the adopted DatasetSpec defaults
(pcc_then_mse, lambda_energy/density/kinetic on).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TAG = "transportOT_v2"

# ---------------------------------------------------------------------------
# Transport variant — concrete lambdas
# ---------------------------------------------------------------------------
#
#   L ≈ L_ot + λ_recon L_recon + λ_latent L_latent + λ_lat_disp L_lat_disp
#       (+ tiny kinetic only for numerical stability)
#
COMMON_OVERRIDES: Dict[str, Any] = {
    # --- main transport terms (like PRESCIENT / MIOFlow / WOT) ---
    "lambda_lat_disp": 3.0,  # strongest: OT-coupled per-cell displacement
    "lambda_latent": 1.5,  # next-time latent population match
    "lambda_recon": 0.05,  # keep decode usable for pca/gene scoring
    # --- landscape / physics OFF ---
    "lambda_energy": 0.0,  # HJB / energy regularizer
    "lambda_density": 0.0,
    "lambda_residual_balance": 0.0,
    "lambda_kinetic": 0.01,  # near-zero; 0 can be unstable on long Δt
    # --- selection (OT-proxy) ---
    "checkpoint_metric": "loss",
    "early_stop_metric": "loss",
    "early_stop_patience": 400,  # avoid lung's epoch-14 collapse
    "use_loss_normalization": True,
    "latent_disp_ot_coupling": True,
    "latent_disp_fullpop_ot": False,  # batch OT; full-pop OOMs on lung
    "no_density_regularization": True,
    "checkpoint_suffix_tag": TAG,
}

# Reference: landscape / paper main model (DO NOT use for OT table)
LANDSCAPE_MAIN_MODEL: Dict[str, Any] = {
    "lambda_lat_disp": 1.0,
    "lambda_latent": 0.5,
    "lambda_recon": 0.01,
    "lambda_energy": 0.05,  # dataset-dependent; ~0.02–0.05
    "lambda_density": 0.01,
    "lambda_kinetic": 0.2,
    "checkpoint_metric": "pcc_then_mse",
    "early_stop_metric": "pcc_then_mse",
}

DATASET_RECIPES: Dict[str, Dict[str, Any]] = {
    "GSE155622": {
        "profile": None,
        "val_mode": "time_extrapolate",
        "epochs": None,
        "extra_cli": [
            "--loss-normalization",
            "--no-density-regularization",
            "--latent-disp-ot-coupling",
            "--no-latent-disp-fullpop-ot",
        ],
    },
    "GSE141259": {
        "profile": "l3_genes5000",
        "val_mode": "time_extrapolate",
        "epochs": None,
        "extra_cli": [
            "--loss-normalization",
            "--no-density-regularization",
            "--latent-disp-ot-coupling",
            "--no-latent-disp-fullpop-ot",
        ],
    },
    "HGSOC": {
        "profile": None,
        "val_mode": "patients",
        "epochs": None,
        "extra_cli": [
            "--loss-normalization",
            "--no-density-regularization",
            "--latent-disp-ot-coupling",
            "--no-latent-disp-fullpop-ot",
        ],
    },
}


def checkpoint_dir_for(dataset: str) -> Path:
    return HERE / "checkpoints" / TAG / dataset


def results_dir_for(score_space: str) -> Path:
    return HERE / "results" / TAG / score_space
