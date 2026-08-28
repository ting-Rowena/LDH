"""Load trained TemporalSDENetwork and re-encode latents for methods analyses."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from celltype_analysis import DATASET_REGISTRY, load_annotated_adata
from dataset_pipeline import (
    apply_train_config,
    prepare_gse141259_for_training,
    prepare_gse155622_for_training,
    prepare_hgsoc_for_training,
)
from train_model import (
    Config,
    SDETrainer,
    TemporalDataProcessor,
    TemporalSDENetwork,
    _load_state_dict_compat,
    ensure_cell_type_codes,
    validate_potential_logp_consistency,
)

PREPARE_FN = {
    "GSE155622": prepare_gse155622_for_training,
    "GSE141259": prepare_gse141259_for_training,
    "HGSOC": prepare_hgsoc_for_training,
}


def _apply_training_summary(config: Config, checkpoint_dir: Path) -> Config:
    summary_path = checkpoint_dir / "training_summary.json"
    if not summary_path.is_file():
        return config
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "potential_time_mode",
        "potential_time_correction_scale",
        "use_hamiltonian_flow",
        "use_state_momentum",
        "use_residual_drift",
        "residual_drift_mode",
        "momentum_loss_type",
        "use_density_regularization",
        "lambda_density",
        "density_align_stationary",
        "density_use_latent_batch",
        "lambda_latent",
        "lambda_kinetic",
        "lambda_recon",
        "val_mode",
    ):
        if key in summary:
            setattr(config, key, summary[key])
    return config


def _infer_n_input_genes(checkpoint_dir: Path) -> Optional[int]:
    ckpt_path = checkpoint_dir / "best_model.pth"
    if not ckpt_path.is_file():
        return None
    state = torch.load(ckpt_path, map_location="cpu")
    w = state.get("gene_encoder.0.weight")
    return int(w.shape[1]) if w is not None else None


def inject_genes_into_panel(
    gene_list: Sequence[str],
    adata_raw,
    force_genes: Sequence[str],
) -> List[str]:
    """
    Swap lowest-mean panel genes with target genes present in raw but absent from panel.

    Keeps fixed input width for a frozen checkpoint while allowing KO genes outside HVG.
    """
    from analysis_protocol_utils import resolve_genes

    panel = list(gene_list)
    missing = [g for g in resolve_genes(adata_raw.var_names, force_genes) if g not in panel]
    if not missing:
        return panel

    import scipy.sparse as sp

    X = adata_raw.X.toarray() if sp.issparse(adata_raw.X) else np.asarray(adata_raw.X)
    present_idx = {g: panel.index(g) for g in panel if g in adata_raw.var_names}
    means = []
    for g in panel:
        if g in present_idx:
            j = list(adata_raw.var_names).index(g)
            means.append((float(np.mean(X[:, j])), g))
        else:
            means.append((float("inf"), g))
    means.sort(key=lambda x: x[0])
    drop_order = [g for _, g in means if g not in missing]

    out = panel.copy()
    for gene in missing:
        if not drop_order:
            break
        slot = drop_order.pop(0)
        j = out.index(slot)
        out[j] = gene
    return out


def shuffle_expression_rows(adata, rng: np.random.Generator):
    """Permute expression rows — destroys cell–expression coupling, keeps marginals."""
    import scipy.sparse as sp

    out = adata.copy()
    perm = rng.permutation(out.n_obs)
    if sp.issparse(out.X):
        out.X = out.X[perm]
    else:
        out.X = np.asarray(out.X)[perm]
    return out


def compose_shuffle_fns(*fns: Callable) -> Callable:
    def _composed(adata, rng):
        out = adata
        for fn in fns:
            out = fn(out, rng)
        return out

    return _composed


def _training_gene_list(checkpoint_dir: Path, adata_raw, config: Config) -> list:
    """Persist the HVG panel used at training time (deterministic re-selection)."""
    path = Path(checkpoint_dir) / "training_var_names.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    panel = TemporalDataProcessor(adata_raw.copy(), config).process().var_names.tolist()
    path.write_text(json.dumps(panel), encoding="utf-8")
    return panel


def _adata_for_training_panel(adata_raw, config: Config, gene_list: Sequence[str]):
    """Normalize/log raw data on the fixed training panel, preserving model input width."""
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    from anndata import AnnData, concat

    present = [g for g in gene_list if g in adata_raw.var_names]
    adata = adata_raw[:, present].copy()
    missing = [g for g in gene_list if g not in adata_raw.var_names]
    if missing:
        warnings.warn(
            f"Training panel genes missing from raw adata; filling zeros: {missing[:5]}",
            UserWarning,
        )
        zeros = sp.csr_matrix((adata_raw.n_obs, len(missing)), dtype=np.float32)
        filler = AnnData(
            X=zeros,
            obs=adata_raw.obs.copy(),
            var=pd.DataFrame(index=pd.Index(missing, name=adata_raw.var_names.name)),
        )
        adata = concat([adata, filler], axis=1, join="outer", merge="same")

    adata = adata[:, list(gene_list)].copy()
    if "log1p" not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata)
    ensure_cell_type_codes(adata, config.cell_type_key)
    return adata


def _knockdown_raw_counts(adata_raw, genes: Sequence[str], factor: float = 0.0):
    """Zero target genes on raw counts before HVG selection (in-place on copy)."""
    from analysis_protocol_utils import resolve_genes

    out = adata_raw.copy()
    resolved = resolve_genes(out.var_names, genes)
    if not resolved:
        return out, []
    import scipy.sparse as sp

    X = out.X.toarray() if sp.issparse(out.X) else np.asarray(out.X)
    for g in resolved:
        j = list(out.var_names).index(g)
        X[:, j] = X[:, j] * float(factor)
    out.X = X
    return out, resolved


def load_training_stack(
    dataset_key: str,
    checkpoint_dir: Path,
    *,
    device: str = None,
    max_cells: Optional[int] = None,
    knockdown_genes: Optional[Sequence[str]] = None,
    knockdown_factor: float = 0.0,
    force_genes: Optional[Sequence[str]] = None,
):
    """
    Return (model, adata_train_ready, config) with checkpoint weights loaded.

    adata is prepared the same way as training (HVG filter + time / cell_type codes).
    ``knockdown_factor`` < 1 scales genes down (KO); > 1 overexpresses.
    """
    profile = DATASET_REGISTRY[dataset_key]
    config = apply_train_config(profile.spec)
    config = _apply_training_summary(config, Path(checkpoint_dir))
    n_in = _infer_n_input_genes(Path(checkpoint_dir))
    if n_in is not None:
        config.n_top_genes = n_in
        config.use_hvg = True
    config.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config.show_figures = False

    adata_raw = load_annotated_adata(profile, str(checkpoint_dir))
    prep = PREPARE_FN.get(dataset_key)
    if prep is not None:
        adata_raw = prep(adata_raw, config)
    gene_list = _training_gene_list(Path(checkpoint_dir), adata_raw, config)
    if force_genes:
        gene_list = inject_genes_into_panel(gene_list, adata_raw, force_genes)
    if knockdown_genes:
        adata_raw, _ = _knockdown_raw_counts(
            adata_raw, knockdown_genes, factor=float(knockdown_factor)
        )
    adata = _adata_for_training_panel(adata_raw, config, gene_list)

    if max_cells and adata.n_obs > max_cells:
        import scanpy as sc
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42, copy=False)

    model = TemporalSDENetwork(config, adata)
    ckpt_path = Path(checkpoint_dir) / "best_model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=config.device)
    _load_state_dict_compat(model, state)
    model = model.to(config.device).eval()
    return model, adata, config


def reencode_latent(model, adata, config: Config, *, device: str = None) -> str:
    """Encode expression → X_latent / X_latent_pca on adata (in-place). Returns latent key."""
    from latent_embeddings import save_latent_embeddings_to_adata

    dev = device or config.device
    model = model.to(dev).eval()
    ensure_cell_type_codes(adata, config.cell_type_key)
    save_latent_embeddings_to_adata(
        model,
        adata,
        device=dev,
        cell_type_key=config.cell_type_key,
    )
    return "X_latent"


def measure_u_kde(
    model,
    adata,
    config: Config,
    *,
    max_cells: int = 3000,
    use_stationary: bool = True,
) -> Dict[str, float]:
    """Pearson/Spearman between model potential and −log KDE(z)."""
    if use_stationary and getattr(config, "potential_time_mode", "") == "quasi_stationary":
        # Custom: U0(z) vs KDE — matches density_align_stationary training target
        import scipy.sparse as sp

        model.eval()
        x_all = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        t_all = adata.obs[config.time_key].values.astype(float)
        ct_all = adata.obs[config.cell_type_key].values.astype(int)
        if len(x_all) > max_cells:
            idx = np.random.default_rng(0).choice(len(x_all), max_cells, replace=False)
            x_all, t_all, ct_all = x_all[idx], t_all[idx], ct_all[idx]

        z_list, u_list = [], []
        bs = int(config.batch_size)
        with torch.no_grad():
            for start in range(0, len(x_all), bs):
                end = min(start + bs, len(x_all))
                x = torch.tensor(x_all[start:end], dtype=torch.float32, device=config.device)
                ct = torch.tensor(ct_all[start:end], dtype=torch.long, device=config.device)
                z = model.encode(x, ct)
                u = model.stationary_potential(z).squeeze(-1).cpu().numpy()
                z_list.append(z.cpu().numpy())
                u_list.append(u)
        z_np = np.vstack(z_list)
        u_np = np.concatenate(u_list)
        from train_model import _potential_neglogp_arrays_from_latent
        from scipy.stats import spearmanr

        u_np, neg_log_p = _potential_neglogp_arrays_from_latent(z_np, u_np)
        if np.std(u_np) < 1e-12 or np.std(neg_log_p) < 1e-12:
            return {"pearson_U_neglogKDE": 0.0, "spearman_U_neglogKDE": 0.0}
        return {
            "pearson_U_neglogKDE": float(np.corrcoef(u_np, neg_log_p)[0, 1]),
            "spearman_U_neglogKDE": float(spearmanr(u_np, neg_log_p).correlation),
        }

    pearson, spearman = validate_potential_logp_consistency(
        model, adata, config, max_cells=max_cells
    )
    return {"pearson_U_neglogKDE": pearson, "spearman_U_neglogKDE": spearman}


def measure_holdout_pcc(
    model,
    adata,
    config: Config,
    *,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """Validation PCC/MSE on held-out transitions (same split as training)."""
    import tempfile

    config_eval = copy.deepcopy(config)
    config_eval.skip_final_evaluation = True
    config_eval.show_figures = False
    # Cap eval minibatch: training configs use 384–512 which OOMs on shared GPUs.
    if batch_size is not None:
        config_eval.batch_size = int(batch_size)
    else:
        config_eval.batch_size = min(int(getattr(config_eval, "batch_size", 32) or 32), 32)
    with tempfile.TemporaryDirectory(prefix="null_val_") as tmp:
        trainer = SDETrainer(model, adata, config_eval, tmp)
        val = trainer._validate()
    return {
        "holdout_pcc": float(val.get("pcc", np.nan)),
        "holdout_mse": float(val.get("mse", np.nan)),
    }


def measure_landscape_metrics(model, adata, config: Config, *, max_cells: int = 4000) -> Dict[str, float]:
    """Landscape summary: potential spread and deep-valley cell fraction."""
    import scipy.sparse as sp

    model.eval()
    x_all = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    ct_all = adata.obs[config.cell_type_key].values.astype(int)
    if len(x_all) > max_cells:
        idx = np.random.default_rng(0).choice(len(x_all), max_cells, replace=False)
        x_all, ct_all = x_all[idx], ct_all[idx]

    u_parts = []
    bs = int(config.batch_size)
    with torch.no_grad():
        for start in range(0, len(x_all), bs):
            end = min(start + bs, len(x_all))
            x = torch.tensor(x_all[start:end], dtype=torch.float32, device=config.device)
            ct = torch.tensor(ct_all[start:end], dtype=torch.long, device=config.device)
            z = model.encode(x, ct)
            u = model.stationary_potential(z).squeeze(-1).cpu().numpy()
            u_parts.append(u)
    u = np.concatenate(u_parts)
    if u.size < 20 or not np.isfinite(u).any():
        return {
            "potential_std": np.nan,
            "valley_depth_p90_p10": np.nan,
            "deep_valley_fraction": np.nan,
        }
    p10, p90 = np.nanpercentile(u, [10, 90])
    return {
        "potential_std": float(np.nanstd(u)),
        "valley_depth_p90_p10": float(p90 - p10),
        "deep_valley_fraction": float(np.mean(u <= p10)),
    }


def latent_delta_from_knockdown(
    model,
    adata,
    config: Config,
    genes: Sequence[str],
    *,
    factor: float = 0.0,
    max_cells: int = 2000,
    seed_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Mean latent shift z(KD) − z(WT) from encoder forward pass (principled KO direction).

    Normalized to unit length for use as rollout seed perturbation.
    """
    from analysis_protocol_utils import resolve_genes

    resolved = resolve_genes(adata.var_names, genes)
    if not resolved:
        return None, []
    import scipy.sparse as sp

    x_all = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    ct_all = adata.obs[config.cell_type_key].values.astype(int)
    mask = np.ones(len(x_all), dtype=bool) if seed_mask is None else np.asarray(seed_mask, dtype=bool)
    idx_cells = np.where(mask)[0]
    if len(idx_cells) > max_cells:
        idx_cells = np.random.default_rng(0).choice(idx_cells, max_cells, replace=False)
    if len(idx_cells) < 10:
        return None, resolved

    gene_idx = [list(adata.var_names).index(g) for g in resolved]
    x_sub = x_all[idx_cells].copy()
    x_kd = x_sub.copy()
    for j in gene_idx:
        x_kd[:, j] = x_kd[:, j] * float(factor)

    ct = torch.tensor(ct_all[idx_cells], dtype=torch.long, device=config.device)
    model.eval()
    with torch.no_grad():
        z_wt = model.encode(
            torch.tensor(x_sub, dtype=torch.float32, device=config.device), ct
        ).cpu().numpy()
        z_kd = model.encode(
            torch.tensor(x_kd, dtype=torch.float32, device=config.device), ct
        ).cpu().numpy()
    delta = np.nanmean(z_kd - z_wt, axis=0)
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm < 1e-8:
        return None, resolved
    return delta / norm, resolved


def gradient_ko_latent_direction(
    model,
    adata,
    config: Config,
    genes: Sequence[str],
    *,
    max_cells: int = 512,
    seed_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Encoder Jacobian direction: mean ∂z/∂x_g averaged over target gene columns.
    """
    from analysis_protocol_utils import resolve_genes

    resolved = resolve_genes(adata.var_names, genes)
    if not resolved:
        return None, []
    import scipy.sparse as sp

    x_all = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    ct_all = adata.obs[config.cell_type_key].values.astype(int)
    mask = np.ones(len(x_all), dtype=bool) if seed_mask is None else np.asarray(seed_mask, dtype=bool)
    idx_cells = np.where(mask)[0]
    if len(idx_cells) > max_cells:
        idx_cells = np.random.default_rng(0).choice(idx_cells, max_cells, replace=False)
    if len(idx_cells) < 10:
        return None, resolved

    gene_idx = [list(adata.var_names).index(g) for g in resolved]
    x = torch.tensor(x_all[idx_cells], dtype=torch.float32, device=config.device, requires_grad=True)
    ct = torch.tensor(ct_all[idx_cells], dtype=torch.long, device=config.device)
    model.eval()
    z = model.encode(x, ct)
    direction = np.zeros(z.shape[1], dtype=float)
    for j in gene_idx:
        for d in range(z.shape[1]):
            grad_x = torch.autograd.grad(z[:, d].sum(), x, retain_graph=True, create_graph=False)[0]
            direction[d] += float(grad_x[:, j].mean().detach().cpu())
    vec = direction
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-8:
        return None, resolved
    return vec / norm, resolved


def measure_physical_scorecard(
    model,
    adata,
    config: Config,
    *,
    max_cells_u_kde: int = 3000,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """Combined physical null metrics: U–KDE, holdout PCC, landscape spread."""
    cfg = copy.deepcopy(config)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    else:
        cfg.batch_size = min(int(getattr(cfg, "batch_size", 32) or 32), 32)
    out = measure_u_kde(model, adata, cfg, max_cells=max_cells_u_kde)
    out.update(measure_holdout_pcc(model, adata, cfg, batch_size=cfg.batch_size))
    out.update(measure_landscape_metrics(model, adata, cfg))
    return out


def finetune_shuffled_control(
    dataset_key: str,
    checkpoint_dir: Path,
    shuffle_fn,
    *,
    n_epochs: int = 40,
    max_cells: int = 2500,
    seed: int = 0,
    device: str = None,
    extended_metrics: bool = True,
    from_scratch: bool = False,
    training_stack=None,
    real_metrics_override: Optional[Dict[str, float]] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Shuffle metadata/expression and train a null control.

    ``from_scratch=True`` discards checkpoint weights and initializes a new
    TemporalSDENetwork.  ``False`` keeps the historical fine-tune-from-pretrained
    behavior for backward compatibility only.
    """
    if training_stack is None:
        model, adata, config = load_training_stack(
            dataset_key, checkpoint_dir, device=device, max_cells=max_cells
        )
    else:
        model, adata, config = training_stack

    measure_fn = measure_physical_scorecard if extended_metrics else measure_u_kde
    real_metrics = (
        dict(real_metrics_override)
        if real_metrics_override is not None
        else measure_fn(model, adata, config)
    )
    real_metrics["label"] = "real_pretrained"

    rng = np.random.default_rng(seed)
    shuffled = shuffle_fn(adata, rng)
    config_ft = copy.deepcopy(config)
    config_ft.epochs = int(n_epochs)
    config_ft.seed = int(seed)
    if batch_size is not None:
        config_ft.batch_size = int(batch_size)
    config_ft.early_stop_patience = max(int(n_epochs), 9999)
    config_ft.skip_final_evaluation = True
    config_ft.show_figures = False

    if from_scratch:
        # Re-initialize the full architecture so null training cannot inherit
        # dynamical weights from the real checkpoint.
        model = TemporalSDENetwork(config_ft, shuffled).to(config_ft.device)
    else:
        model = copy.deepcopy(model).to(config_ft.device)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="null_ft_") as tmp:
        trainer = SDETrainer(model, shuffled, config_ft, tmp)
        trainer.train()
        metrics = measure_fn(trainer.model, shuffled, config_ft)
    metrics["label"] = "shuffled_retrain"
    metrics["initialization"] = "random_from_scratch" if from_scratch else "pretrained_finetune"
    if batch_size is not None:
        metrics["batch_size"] = int(batch_size)
    real_s = real_metrics.get("spearman_U_neglogKDE", np.nan)
    metrics["real_spearman"] = real_s
    metrics["collapse_ratio"] = (
        metrics["spearman_U_neglogKDE"] / real_s if abs(real_s) > 1e-6 else np.nan
    )
    if extended_metrics and np.isfinite(real_metrics.get("holdout_pcc", np.nan)):
        metrics["holdout_pcc_collapse_ratio"] = (
            metrics.get("holdout_pcc", np.nan) / real_metrics["holdout_pcc"]
            if abs(real_metrics["holdout_pcc"]) > 1e-6
            else np.nan
        )
        r_depth = real_metrics.get("valley_depth_p90_p10", np.nan)
        metrics["valley_depth_collapse_ratio"] = (
            metrics.get("valley_depth_p90_p10", np.nan) / r_depth
            if np.isfinite(r_depth) and abs(r_depth) > 1e-6
            else np.nan
        )
    return {**real_metrics, **metrics}
