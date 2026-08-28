#!/usr/bin/env python
"""Matched temporal null: time/pairing shuffle only, clean-holdout evaluation.

Protocol (canonical paper null):
  - Preserve expression ↔ cell ↔ cell-type coupling
  - Pain/lung: jointly permute temporal metadata across cells
  - HGSOC: shuffle treatment_phase within each patient (pairing null)
  - Fixed cell subset (max_cells=5000, subsample seed=42)
  - From-scratch retrain, n_epochs=500, batch_size=128 (micro-batch 32 if needed)
  - n_replicates=4
  - All metrics evaluated on the original unshuffled subset (clean holdout)

Summary table: checkpoint methods_enhancement/physical_retrain_controls_*.csv
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from plot_utils import PALETTE, configure_headless, style_axis
from methods_enhancement_utils import fig_path, methods_outdir, result_path, write_output_file_index
from methods_model_utils import (
    TemporalSDENetwork,
    _apply_training_summary,
    _infer_n_input_genes,
    _load_state_dict_compat,
    load_training_stack,
    measure_physical_scorecard,
)

configure_headless()

from celltype_analysis import DATASET_REGISTRY
from dataset_pipeline import apply_train_config, recommended_checkpoint_dir


def shuffle_temporal_block(adata, rng: np.random.Generator):
    """Jointly permute temporal metadata; keep X and cell-type codes intact."""
    out = adata.copy()
    cols = [
        c
        for c in ("time", "stage", "condition", "treatment_phase", "stage_code")
        if c in out.obs.columns
    ]
    if not cols:
        return out
    perm = rng.permutation(out.n_obs)
    for col in cols:
        out.obs[col] = np.asarray(out.obs[col])[perm]
    if "time" in out.obs.columns:
        out.obs["time"] = out.obs["time"].astype(float)
    return out


def shuffle_patient_pairing(adata, rng: np.random.Generator):
    """Within each patient, shuffle treatment_phase (destroys naive↔post pairing)."""
    out = adata.copy()
    if "patient_id" not in out.obs.columns or "treatment_phase" not in out.obs.columns:
        return shuffle_temporal_block(out, rng)
    for pid in out.obs["patient_id"].astype(str).unique():
        m = out.obs["patient_id"].astype(str) == pid
        phases = out.obs.loc[m, "treatment_phase"].astype(str).to_numpy(copy=True)
        rng.shuffle(phases)
        out.obs.loc[m, "treatment_phase"] = phases
    # Keep numeric time axis consistent with treatment_phase when present.
    if "time" in out.obs.columns and "treatment_phase" in out.obs.columns:
        phase = out.obs["treatment_phase"].astype(str).str.lower()
        time_map = {}
        for p in phase.unique():
            if "naive" in p or "pre" in p or p in {"0", "primary"}:
                time_map[p] = 0.0
            elif "post" in p or "interval" in p or p in {"1"}:
                time_map[p] = 1.0
        if time_map:
            out.obs["time"] = phase.map(time_map).astype(float).to_numpy()
    return out


def matched_shuffle_fn(dataset_key: str) -> Callable:
    if dataset_key == "HGSOC":
        return shuffle_patient_pairing
    return shuffle_temporal_block


def matched_shuffle_mode(dataset_key: str) -> str:
    return "pairing_matched" if dataset_key == "HGSOC" else "temporal_matched"


def _null_adata_cache_path(checkpoint_dir: Path, max_cells: int) -> Path:
    return methods_outdir(checkpoint_dir) / f"_cache_null_adata_mc{int(max_cells)}_rs42.h5ad"


def load_or_build_null_stack(
    dataset_key: str,
    checkpoint_dir: Path,
    *,
    max_cells: int,
    device: Optional[str],
):
    """Load pretrained model + fixed cell subset; cache adata to skip 15GB raw rereads."""
    cache = _null_adata_cache_path(checkpoint_dir, max_cells)
    profile = DATASET_REGISTRY[dataset_key]
    config = apply_train_config(profile.spec)
    config = _apply_training_summary(config, Path(checkpoint_dir))
    n_in = _infer_n_input_genes(Path(checkpoint_dir))
    if n_in is not None:
        config.n_top_genes = n_in
        config.use_hvg = True
    config.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config.show_figures = False

    if cache.is_file():
        print(f"[{dataset_key}] loading cached adata {cache} ...", flush=True)
        adata = ad.read_h5ad(cache)
        print(
            f"[{dataset_key}] cache hit n_cells={adata.n_obs} n_genes={adata.n_vars}",
            flush=True,
        )
    else:
        print(
            f"[{dataset_key}] cache miss — reading raw + building panel "
            f"(will write {cache.name}) ...",
            flush=True,
        )
        # Build on CPU to avoid CUDA interaction during multi-GB I/O.
        _, adata, config_built = load_training_stack(
            dataset_key, checkpoint_dir, device="cpu", max_cells=max_cells
        )
        config = config_built
        config.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(cache)
        print(
            f"[{dataset_key}] wrote cache {cache} ({cache.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )

    model = TemporalSDENetwork(config, adata)
    ckpt_path = Path(checkpoint_dir) / "best_model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=config.device)
    _load_state_dict_compat(model, state)
    model = model.to(config.device).eval()
    return model, adata, config


def train_null_eval_clean(
    *,
    dataset_key: str,
    checkpoint_dir: Path,
    clean_model,
    clean_adata,
    clean_config,
    shuffle_fn: Callable,
    n_epochs: int,
    seed: int,
    batch_size: int,
    real_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Train from scratch on shuffled labels; score on clean (unshuffled) adata."""
    import copy

    from train_model import SDETrainer, TemporalSDENetwork

    rng = np.random.default_rng(seed)
    shuffled = shuffle_fn(clean_adata, rng)

    config_ft = copy.deepcopy(clean_config)
    config_ft.epochs = int(n_epochs)
    config_ft.seed = int(seed)
    config_ft.batch_size = int(batch_size)
    # Effective batch=128 under ~1GB free VRAM: accumulate 32-wide micro-batches.
    if int(batch_size) > 32:
        config_ft.micro_batch_size = 32
    config_ft.early_stop_patience = max(int(n_epochs), 9999)
    config_ft.skip_final_evaluation = True
    config_ft.show_figures = False
    # Validation is unused for null (patience disabled); skip to avoid OOM under orphan VRAM.
    config_ft.skip_epoch_validation = True

    model = TemporalSDENetwork(config_ft, shuffled).to(config_ft.device)
    with tempfile.TemporaryDirectory(prefix="temporal_null_") as tmp:
        trainer = SDETrainer(model, shuffled, config_ft, tmp)
        trainer.train()
        # Evaluate on original, unshuffled cells / holdout protocol.
        # Cap eval batch to 32 to avoid OOM on the scorecard forward pass.
        eval_bs = min(int(batch_size), 32)
        metrics = measure_physical_scorecard(
            trainer.model, clean_adata, clean_config, batch_size=eval_bs
        )

    metrics["label"] = "temporal_null_retrain"
    metrics["initialization"] = "random_from_scratch"
    metrics["batch_size"] = int(batch_size)
    metrics["micro_batch_size"] = int(
        getattr(config_ft, "micro_batch_size", batch_size) or batch_size
    )
    metrics["eval_on"] = "clean_unshuffled"
    real_s = real_metrics.get("spearman_U_neglogKDE", np.nan)
    metrics["real_spearman"] = real_s
    metrics["collapse_ratio"] = (
        metrics["spearman_U_neglogKDE"] / real_s if abs(real_s) > 1e-6 else np.nan
    )
    if np.isfinite(real_metrics.get("holdout_pcc", np.nan)):
        rp = real_metrics["holdout_pcc"]
        metrics["holdout_pcc_collapse_ratio"] = (
            metrics.get("holdout_pcc", np.nan) / rp if abs(rp) > 1e-6 else np.nan
        )
        rd = real_metrics.get("valley_depth_p90_p10", np.nan)
        metrics["valley_depth_collapse_ratio"] = (
            metrics.get("valley_depth_p90_p10", np.nan) / rd
            if np.isfinite(rd) and abs(rd) > 1e-6
            else np.nan
        )
    return metrics


def run_matched_temporal_null(
    dataset_key: str,
    checkpoint_dir: Path,
    *,
    n_replicates: int = 5,
    n_epochs: int = 200,
    max_cells: int = 2500,
    batch_size: int = 32,
    seed: int = 42,
    device: Optional[str] = None,
) -> pd.DataFrame:
    out = methods_outdir(checkpoint_dir)
    mode = matched_shuffle_mode(dataset_key)
    shuffle_fn = matched_shuffle_fn(dataset_key)

    print(
        f"[{dataset_key}] loading stack max_cells={max_cells} device={device or 'auto'}...",
        flush=True,
    )
    model, adata, config = load_or_build_null_stack(
        dataset_key, checkpoint_dir, max_cells=max_cells, device=device
    )
    config.batch_size = int(batch_size)
    print(
        f"[{dataset_key}] loaded n_cells={adata.n_obs} n_genes={adata.n_vars}; "
        f"scoring real pretrained...",
        flush=True,
    )

    eval_bs = min(int(batch_size), 32)
    real = measure_physical_scorecard(model, adata, config, batch_size=eval_bs)
    print(
        f"[{dataset_key}] real Spearman={real.get('spearman_U_neglogKDE', float('nan')):.4f} "
        f"PCC={real.get('holdout_pcc', float('nan')):.4f}",
        flush=True,
    )
    real.update(
        {
            "dataset": dataset_key,
            "control": "real_pretrained",
            "replicate": -1,
            "n_epochs": 0,
            "shuffle_mode": mode,
            "initialization": "pretrained",
            "batch_size": batch_size,
            "eval_on": "clean_unshuffled",
            "n_cells": int(adata.n_obs),
        }
    )
    rows: List[dict] = [real]

    for rep in range(n_replicates):
        print(
            f"[{dataset_key}] matched temporal null rep {rep + 1}/{n_replicates} "
            f"mode={mode} epochs={n_epochs} cells={adata.n_obs} batch={batch_size}",
            flush=True,
        )
        m = train_null_eval_clean(
            dataset_key=dataset_key,
            checkpoint_dir=checkpoint_dir,
            clean_model=model,
            clean_adata=adata,
            clean_config=config,
            shuffle_fn=shuffle_fn,
            n_epochs=n_epochs,
            seed=seed + rep,
            batch_size=batch_size,
            real_metrics=real,
        )
        rows.append(
            {
                "dataset": dataset_key,
                "control": "shuffle_retrain",
                "replicate": rep,
                "n_epochs": n_epochs,
                "shuffle_mode": mode,
                "initialization": "random_from_scratch",
                "batch_size": batch_size,
                "eval_on": "clean_unshuffled",
                "n_cells": int(adata.n_obs),
                "pearson_U_neglogKDE": m.get("pearson_U_neglogKDE", np.nan),
                "spearman_U_neglogKDE": m.get("spearman_U_neglogKDE", np.nan),
                "holdout_pcc": m.get("holdout_pcc", np.nan),
                "holdout_mse": m.get("holdout_mse", np.nan),
                "potential_std": m.get("potential_std", np.nan),
                "valley_depth_p90_p10": m.get("valley_depth_p90_p10", np.nan),
                "deep_valley_fraction": m.get("deep_valley_fraction", np.nan),
                "real_spearman": m.get("real_spearman", np.nan),
                "collapse_ratio": m.get("collapse_ratio", np.nan),
                "holdout_pcc_collapse_ratio": m.get("holdout_pcc_collapse_ratio", np.nan),
                "valley_depth_collapse_ratio": m.get("valley_depth_collapse_ratio", np.nan),
            }
        )

    df = pd.DataFrame(rows)
    tag = f"{dataset_key}_{mode}_e{n_epochs}_mc{int(max_cells)}_bs{int(batch_size)}"
    df.to_csv(result_path(out, f"physical_retrain_controls_{tag}.csv"), index=False)

    null = df[df["control"] == "shuffle_retrain"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    specs = [
        ("spearman_U_neglogKDE", "Spearman(U, −log KDE)", real.get("spearman_U_neglogKDE", np.nan)),
        ("holdout_pcc", "Holdout PCC (clean)", real.get("holdout_pcc", np.nan)),
        ("valley_depth_p90_p10", "Valley depth", real.get("valley_depth_p90_p10", np.nan)),
    ]
    for ax, (col, ylab, real_val) in zip(axes, specs):
        if not null.empty and col in null.columns:
            ax.boxplot(
                [null[col].dropna().to_numpy()],
                labels=[mode],
                patch_artist=True,
                boxprops=dict(facecolor=PALETTE[1], alpha=0.55),
            )
        if np.isfinite(real_val):
            ax.axhline(real_val, color=PALETTE[5], ls="--", lw=2, label=f"real={real_val:.3f}")
        ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=7)
        style_axis(ax, grid_axis="y")
    fig.suptitle(
        f"{dataset_key}: matched temporal null ({mode}) | {n_epochs}ep × {n_replicates}rep | "
        f"cells={adata.n_obs}, batch={batch_size}, eval=clean"
    )
    fig.tight_layout()
    fig.savefig(fig_path(out, f"physical_retrain_controls_{tag}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    def _collapse(col: str, real_val: float) -> float:
        if null.empty or col not in null.columns or not np.isfinite(real_val) or abs(real_val) < 1e-6:
            return np.nan
        return float(null[col].median() / real_val)

    summary = {
        "dataset": dataset_key,
        "shuffle_mode": mode,
        "initialization": "random_from_scratch",
        "n_epochs": n_epochs,
        "n_replicates": n_replicates,
        "max_cells": max_cells,
        "n_cells_used": int(adata.n_obs),
        "batch_size": batch_size,
        "micro_batch_size": 32 if int(batch_size) > 32 else int(batch_size),
        "eval_on": "clean_unshuffled",
        "real_spearman": float(real.get("spearman_U_neglogKDE", np.nan)),
        "null_median_spearman": float(null["spearman_U_neglogKDE"].median()) if not null.empty else np.nan,
        "null_spearman_q25": float(null["spearman_U_neglogKDE"].quantile(0.25)) if not null.empty else np.nan,
        "null_spearman_q75": float(null["spearman_U_neglogKDE"].quantile(0.75)) if not null.empty else np.nan,
        "collapse_ratio": _collapse("spearman_U_neglogKDE", float(real.get("spearman_U_neglogKDE", np.nan))),
        "real_holdout_pcc": float(real.get("holdout_pcc", np.nan)),
        "null_median_holdout_pcc": float(null["holdout_pcc"].median()) if not null.empty else np.nan,
        "null_holdout_pcc_q25": float(null["holdout_pcc"].quantile(0.25)) if not null.empty else np.nan,
        "null_holdout_pcc_q75": float(null["holdout_pcc"].quantile(0.75)) if not null.empty else np.nan,
        "holdout_pcc_collapse_ratio": _collapse("holdout_pcc", float(real.get("holdout_pcc", np.nan))),
        "real_valley_depth": float(real.get("valley_depth_p90_p10", np.nan)),
        "null_median_valley_depth": float(null["valley_depth_p90_p10"].median()) if not null.empty else np.nan,
        "valley_depth_collapse_ratio": _collapse(
            "valley_depth_p90_p10", float(real.get("valley_depth_p90_p10", np.nan))
        ),
    }
    (result_path(out, f"physical_retrain_controls_{tag}_summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_output_file_index(out, dataset_key=dataset_key)
    print(json.dumps(summary, indent=2), flush=True)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="Matched temporal null (clean-holdout eval)")
    p.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), required=True)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--n-replicates", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--max-cells", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args(argv)
    ckpt = Path(args.checkpoint_dir or recommended_checkpoint_dir(args.dataset))
    df = run_matched_temporal_null(
        args.dataset,
        ckpt,
        n_replicates=args.n_replicates,
        n_epochs=args.n_epochs,
        max_cells=args.max_cells,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
