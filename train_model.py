"""
Hamiltonian Biological Flow Model for single-cell dynamics.

Architecture
------------
1. Latent encoder z = f_enc(x) + type_embed  (cell_type must be integer codes)
2. Potential U(z, t); momentum p (EMA of latent velocity or integrated state)
3. H(z,p,t) = U(z,t) + 0.5 p^T p;  dz/dt = p + r_theta by default;  dp/dt = -∇U - γp + noise
4. Unified loss: L_OT + λ_recon L_recon + λ_energy L_energy + λ_kinetic L_inertia + λ_latent L_latent + λ_lat_disp L_lat_disp (+ optional density/delta/direction)
5. Expression head: predicted_expression (NOT RNA velocity)
6. Validation on held-out cells / times / types / extrapolation pairs
"""

import math
import random
import time
import warnings
from typing import Optional

from plot_utils import (
    configure_headless,
    ensure_figure_dir,
    finish_figure,
    plot_embedding_panel,
    resolve_label_key,
    resolve_violin_groupby_key,
    save_figure,
    setup_scanpy_figdir,
    _adaptive_dpi,
)

configure_headless()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors
from torch import optim

try:
    import scvelo as scv
except ImportError:
    scv = None

import sde as sde
from hamiltonian_flow import (
    HamiltonianFlowFunc,
    energy_regularizer,
    integrate_hamiltonian_flow,
    momentum_regularizer,
    total_drift_energy_regularizer,
)

VELOCITY_LAYER_TITLE = "model-predicted expression delta, not RNA velocity"


class Config:
    def __init__(self):
        self.data_path = "./GSE155622_raw_UMI_counts_3.h5ad"
        self.time_key = "time"
        self.cell_type_key = "cell_type"
        self.temporal_group_key = "stage"
        self.hvg_flavor = "seurat_v3"
        self.min_genes = 400
        self.n_top_genes = 3000
        self.use_hvg = True
        self.start_time = 0
        self.train_time = [0, 1, 2, 3, 4, 5, 6, 7]

        self.hidden_dim = 512
        self.n_layers = 3
        self.dropout = 0.15
        self.time_emb_dim = 48
        self.sigma_type = "Mlp"
        self.sigma_min = 1e-3
        self.sigma_scale = 0.1
        self.activation = "softplus"

        self.epochs = 3000
        self.lr = 1e-5
        self.batch_size = 512
        # Optional micro-batching: sample ``batch_size`` pairs then optimize in
        # chunks of ``micro_batch_size`` with gradient accumulation so the
        # effective batch matches ``batch_size`` under tight GPU memory.
        self.micro_batch_size = None
        self.gradient_accumulation_steps = 1
        self._lambda_hjb = 0.02  # checkpoint alias for lambda_energy
        self.lambda_recon = 0.01
        self.lambda_delta = 0.0
        self.lambda_direction = 0.0
        self.loss_recipe = "legacy"
        self.lambda_ae = 0.0
        self.lambda_hamiltonian = 0.0
        self.use_hamiltonian_flow = True
        self.hamiltonian_damping_gamma = 0.1
        self.momentum_ema = 0.9
        # State-dependent initial momentum p_theta(z, t). When True, each cell carries
        # its own inertial momentum (well-defined second-order dynamics) instead of a
        # single global batch-mean EMA vector shared by every cell (legacy behaviour).
        self.use_state_momentum = True
        # Potential parameterization:
        #   "time_varying"     -> U(z, t) (legacy; not a true landscape, U drifts with t)
        #   "quasi_stationary" -> U(z, t) = U0(z) + eps * phi(z, t); the reported Waddington
        #                          quasi-potential landscape is the time-invariant U0(z), and
        #                          phi is an explicit small non-equilibrium correction.
        self.potential_time_mode = "quasi_stationary"
        self.potential_time_correction_scale = 0.05
        self.early_stop_metric = "pcc_then_mse"  # pcc | mse | loss | pcc_then_mse
        self.checkpoint_metric = "pcc_then_mse"  # pcc | mse | loss | pcc_then_mse
        self.checkpoint_pcc_tie_epsilon = 0.005
        self.early_stop_patience = 200
        self.early_stop_min_delta = 1e-4
        self.skip_final_evaluation = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 112

        # Transition training: adjacent t_i -> t_{i+1} plus optional anchor 0 -> t_j
        self.use_adjacent_transitions = True
        self.use_anchor_transitions = True

        # Validation: held-out cells | time | cell_type | time_extrapolate
        self.val_mode = "cells"
        self.val_ratio = 0.15
        self.val_time_point = None

        # Optional EMA loss normalization stabilizes multi-term distributional training.
        self.use_loss_normalization = True
        self.loss_norm_momentum = 0.98
        self.loss_norm_eps = 1e-8

        # Architecture / transition ablations (defaults preserve prior behaviour).
        self.use_residual_drift = True
        self.residual_drift_mode = "velocity"
        self.momentum_loss_type = "velocity"
        self.use_cell_type_embedding = True
        self.ablation_flags: dict = {}

        # When True, HJ regularizer uses total drift (-∇U + r_theta). Recommended with use_residual_drift.
        self.use_total_drift_hjb = False
        # Latent-consistency & inertia regularizers. The latent flow otherwise diverges
        # (integrate(encode(x_t))@t+1 lands far outside encode(x_{t+1}); see short-horizon
        # diagnostic). lambda_latent pulls the pushed-forward latent distribution onto the
        # encoded next-timepoint distribution (target detached). lambda_kinetic scales a
        # combined inertia term: INERTIA_MOMENTUM_MIX * L_momentum + L_kinetic (fixed mix).
        self.lambda_latent = 0.5
        self.lambda_kinetic = 0.2
        # Population-level latent displacement (OT-coupled per-cell or centroid fallback).
        self.lambda_lat_disp = 1.0
        # Overshoot control. kinetic_terminal_beta (>1) makes the kinetic penalty
        # emphasise the FINAL momentum ||p_final||^2 over the initial one, so the
        # flow is pushed to "stop at" the target instead of coasting through it
        # (the second-order inertia is the physical source of the ~1.4x latent
        # displacement overshoot). latent_disp_detach_potential runs a second,
        # potential-detached (z, p) integration whose z_pred feeds only the
        # population displacement/direction objective, gating those gradients out
        # of the quasi-potential U0 so the Waddington landscape (validated vs
        # -log KDE) is not distorted while the momentum/residual fields still learn
        # the direction and magnitude.
        self.kinetic_terminal_beta = 2.0
        self.latent_disp_detach_potential = True
        # OT-coupled per-cell displacement target. When True, the displacement /
        # direction losses supervise each source cell against its entropic-OT
        # barycentric image in z_next (per cell type) instead of the group centroid,
        # constraining per-cell direction AND magnitude (curbs the overshoot that the
        # centroid-only objective leaves uncontrolled). latent_disp_ot_blur is the
        # Sinkhorn entropic scale on the mean-normalised squared-distance cost.
        self.latent_disp_ot_coupling = True
        self.latent_disp_ot_blur = 0.05
        self.latent_disp_use_mag_ratio = False
        self.latent_disp_exclude_ema = False
        self.latent_disp_fullpop_ot = False

        self.reconstruction_mode = "mse_mmd"

        self.lambda_density = 0.01
        self.use_density_regularization = True
        # Align U0 (not U(z,t)) with density; use latent k-NN during training batches.
        self.density_align_stationary = True
        self.density_use_latent_batch = True
        self.density_within_cell_type = True
        self.density_knn_k = 10
        self.density_basis = "X_pca"
        self.density_n_pcs = 20
        self.density_bandwidth = None
        self.density_target_key = "density_neglogp"
        # Penalize residual drift dominating gradient flow (improves landscape interpretability).
        self.lambda_residual_balance = 0.01
        self.residual_ratio_target = 0.55
        # Reference time for homeostasis deviation at inference (None => earliest time).
        self.homeostasis_ref_time = None
        self.compute_plasticity_scores = True

        # Optional transition pairing beyond cell type (e.g. HGSOC patient-paired NACT).
        # When set, sample_transition_pairs only matches within these obs groups.
        self.pair_group_keys = None  # e.g. ["patient_id"]
        self.patient_key = "patient_id"
        # Clinical stage (or other) as embedding conditioner — not used as time.
        self.stage_cond_key = None  # integer codes in adata.obs
        self.use_stage_embedding = False
        self.n_stages = None

        self.color_palette = "tab20"
        self.umap_params = {"n_neighbors": 15, "min_dist": 0.3}
        self.plot_genes = 3
        self.sde_dt = 0.1
        self.keep_velocity_alias = True
        self.show_figures = False
        self.figure_subdir = "figures"
        self.plot_style = None

    @property
    def lambda_hjb(self):
        """Checkpoint-compatible alias for lambda_energy."""
        return self._lambda_hjb

    @lambda_hjb.setter
    def lambda_hjb(self, value):
        self._lambda_hjb = float(value)

    @property
    def lambda_energy(self):
        return self._lambda_hjb

    @lambda_energy.setter
    def lambda_energy(self, value):
        self._lambda_hjb = float(value)

    @property
    def lambda_reg(self):
        """Backward-compatible alias for lambda_energy (used in checkpoint dir names)."""
        return self._lambda_hjb

    @lambda_reg.setter
    def lambda_reg(self, value):
        self._lambda_hjb = float(value)


def warn_gradient_only_hj_regularizer(cfg) -> None:
    """Warn when gradient-only HJ regularizer omits residual drift."""
    if (
        getattr(cfg, "use_residual_drift", True)
        and getattr(cfg, "residual_drift_mode", "velocity") != "none"
        and not getattr(cfg, "use_total_drift_hjb", False)
        and float(getattr(cfg, "lambda_hjb", 0.0) or 0.0) > 0
    ):
        warnings.warn(
            "Gradient-only HJ-inspired regularizer ignores residual drift r_theta; "
            "set use_total_drift_hjb=True when use_residual_drift=True.",
            UserWarning,
            stacklevel=2,
        )


config = Config()
random.seed(config.seed)
torch.manual_seed(config.seed)
np.random.seed(config.seed)


def ensure_cell_type_codes(adata, key):
    """Ensure cell_type column is integer codes for nn.Embedding."""
    series = adata.obs[key]
    if pd.api.types.is_integer_dtype(series) and series.min() >= 0:
        adata.obs[key] = series.astype(int)
        return adata
    if pd.api.types.is_numeric_dtype(series):
        uniq = np.sort(series.unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        adata.obs[key] = series.map(mapping).astype(int)
        return adata
    codes = pd.Categorical(series).codes
    if (codes < 0).any():
        raise ValueError(f"{key} 含缺失类别，请先清洗 adata.obs['{key}']")
    adata.obs[key] = codes.astype(int)
    return adata


def get_training_transitions(cfg):
    """Build (t_curr, t_next) pairs: adjacent steps and/or anchor from start_time."""
    times = sorted(set([cfg.start_time] + list(cfg.train_time)))
    pairs = []
    if cfg.use_adjacent_transitions:
        for i in range(len(times) - 1):
            pairs.append((float(times[i]), float(times[i + 1])))
    if cfg.use_anchor_transitions:
        for t in times:
            t = float(t)
            if t != float(cfg.start_time):
                pairs.append((float(cfg.start_time), t))
    pairs = sorted(set(pairs))
    if not pairs:
        raise ValueError("无可用训练时间转移对，请检查 train_time / start_time 配置")
    return pairs


class TemporalDataProcessor:
    def __init__(self, adata, cfg=None):
        self.adata = adata
        self.cfg = cfg or config

    def process(self):
        sc.pp.filter_genes(self.adata, min_counts=3)
        flavor = getattr(self.cfg, "hvg_flavor", "seurat_v3")
        if self.cfg.use_hvg:
            if flavor == "seurat_v3":
                # seurat_v3 expects raw counts in X (before normalize/log1p)
                sc.pp.highly_variable_genes(
                    self.adata,
                    n_top_genes=self.cfg.n_top_genes,
                    flavor="seurat_v3",
                )
                self.adata = self.adata[:, self.adata.var.highly_variable]
                sc.pp.normalize_total(self.adata, target_sum=1e4, inplace=True)
                sc.pp.log1p(self.adata)
            else:
                sc.pp.normalize_total(self.adata, target_sum=1e4, inplace=True)
                sc.pp.log1p(self.adata)
                sc.pp.highly_variable_genes(
                    self.adata,
                    n_top_genes=self.cfg.n_top_genes,
                    flavor=flavor,
                )
                self.adata = self.adata[:, self.adata.var.highly_variable]
        else:
            sc.pp.normalize_total(self.adata, target_sum=1e4, inplace=True)
            sc.pp.log1p(self.adata)
            self._select_temporal_genes()
        ensure_cell_type_codes(self.adata, self.cfg.cell_type_key)
        return self.adata

    def _resolve_temporal_group_key(self):
        key = getattr(self.cfg, "temporal_group_key", None) or self.cfg.time_key
        if key in self.adata.obs.columns:
            return key
        for fallback in ("stage", "condition", "annotation", self.cfg.time_key):
            if fallback in self.adata.obs.columns:
                warnings.warn(
                    f"temporal_group_key '{key}' 不在 obs 中，回退使用 '{fallback}'",
                    UserWarning,
                    stacklevel=2,
                )
                return fallback
        raise ValueError(
            f"无法找到用于 temporal DEG 的分组列，请设置 config.temporal_group_key"
        )

    def _select_temporal_genes(self):
        group_key = self._resolve_temporal_group_key()
        sc.tl.rank_genes_groups(
            self.adata,
            groupby=group_key,
            method="t-test",
            n_genes=self.cfg.min_genes,
        )
        result = self.adata.uns["rank_genes_groups"]
        groups = result["names"].dtype.names
        df = pd.DataFrame(
            {group + "_names": result["names"][group] for group in groups}
        )
        var_names = np.unique(np.concatenate(df.values))
        self.adata.var["temporal"] = self.adata.var_names.isin(var_names)
        self.adata = self.adata[:, self.adata.var.temporal]


def _activation_module(name):
    if name == "relu":
        return nn.LeakyReLU()
    if name == "softplus":
        return nn.Softplus()
    if name == "tanh":
        return nn.Tanh()
    if name == "none":
        return nn.Identity()
    raise NotImplementedError(name)


def _build_deep_mlp(
    in_dim: int,
    out_dim: int,
    cfg,
    *,
    activation: nn.Module,
    use_layernorm: bool = True,
    final_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """Stack of ``cfg.n_layers`` linear blocks (width ``cfg.hidden_dim``)."""
    n_layers = max(1, int(getattr(cfg, "n_layers", 2)))
    width = int(cfg.hidden_dim)
    dropout = float(getattr(cfg, "dropout", 0.1))
    layers: list[nn.Module] = []
    if n_layers <= 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        dims = [in_dim] + [width] * (n_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_layernorm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


def _sinkhorn_plan(a_pts, b_pts, blur=0.05, n_iter=50):
    """Entropic-OT transport plan between two point clouds (uniform marginals).

    Returns the (n_a, n_b) coupling ``P`` for cost ``C_ij = ||a_i - b_j||^2``
    (normalised by its mean for scale stability). Intended to be called under
    ``torch.no_grad()`` — used to build a well-posed per-cell barycentric target
    for the latent displacement loss, so it does not need to be differentiable.
    """
    C = torch.cdist(a_pts, b_pts, p=2) ** 2
    C = C / (C.mean() + 1e-8)
    n, m = C.shape
    K = -C / max(blur, 1e-6)
    log_a = -math.log(n)
    log_b = -math.log(m)
    f = torch.zeros(n, device=C.device, dtype=C.dtype)
    g = torch.zeros(m, device=C.device, dtype=C.dtype)
    for _ in range(n_iter):
        f = log_a - torch.logsumexp(K + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(K + f.unsqueeze(1), dim=0)
    return torch.exp(f.unsqueeze(1) + K + g.unsqueeze(0))


class PotentialNetwork(nn.Module):
    """Latent potential U(z, t).

    Two parameterizations (``cfg.potential_time_mode``):

    - ``"time_varying"`` (legacy): a single MLP U(z, t). Convenient, but U is not a
      quasi-potential — its minima and barriers can drift arbitrarily with t, so it
      should not be interpreted as a Waddington energy landscape.
    - ``"quasi_stationary"``: U(z, t) = U0(z) + eps * phi(z, t). The reported landscape
      is the time-invariant quasi-potential U0(z); the small (eps<<1) correction phi(z, t)
      captures explicit non-equilibrium / time-dependent forcing. This makes the
      "landscape" claim well defined while retaining time dependence in the dynamics.
    """

    def __init__(self, latent_dim, cfg):
        super().__init__()
        act = _activation_module(cfg.activation)
        h = cfg.hidden_dim
        self.time_mode = getattr(cfg, "potential_time_mode", "time_varying")
        if self.time_mode not in {"time_varying", "quasi_stationary"}:
            raise ValueError(f"Unknown potential_time_mode={self.time_mode!r}")
        self.time_correction_scale = float(getattr(cfg, "potential_time_correction_scale", 0.1))
        if self.time_mode == "quasi_stationary":
            self.stationary_net = nn.Sequential(
                nn.Linear(latent_dim, h),
                act,
                nn.Linear(h, h),
                act,
                nn.Linear(h, 1),
            )
            self.time_correction_net = nn.Sequential(
                nn.Linear(latent_dim + 1, h),
                act,
                nn.Linear(h, 1),
            )
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim + 1, h),
                act,
                nn.Linear(h, h),
                act,
                nn.Linear(h, 1),
            )

    def stationary_potential(self, z):
        """Time-invariant quasi-potential U0(z) used for landscape reporting."""
        if self.time_mode == "quasi_stationary":
            return self.stationary_net(z)
        t0 = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        return self.net(torch.cat([z, t0], dim=-1))

    def forward(self, z, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if self.time_mode == "quasi_stationary":
            u0 = self.stationary_net(z)
            corr = self.time_correction_net(torch.cat([z, t], dim=-1))
            return u0 + self.time_correction_scale * corr
        return self.net(torch.cat([z, t], dim=-1))


class MomentumNetwork(nn.Module):
    """State-dependent initial momentum p_theta(z, t).

    Replaces the legacy global batch-mean EMA momentum (a single vector shared by every
    cell) with a per-cell momentum field, so the second-order (inertial) latent dynamics
    dz/dt = p + r_theta are well defined at each state rather than seeded by a global
    constant. Output is unconstrained (momentum has the units of a latent velocity).
    """

    def __init__(self, latent_dim, cfg):
        super().__init__()
        act = _activation_module(cfg.activation)
        h = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, h),
            act,
            nn.Linear(h, latent_dim),
        )

    def forward(self, z, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([z, t], dim=-1))


class ResidualDriftNetwork(nn.Module):
    """Non-conservative residual drift r_theta(z, t)."""

    def __init__(self, latent_dim, cfg):
        super().__init__()
        act = _activation_module(cfg.activation)
        h = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, h),
            act,
            nn.Linear(h, latent_dim),
            nn.Tanh(),
        )

    def forward(self, z, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([z, t], dim=-1))


class LatentSDEFunc(HamiltonianFlowFunc):
    """Backward-compatible alias for HamiltonianFlowFunc."""


class TemporalSDENetwork(nn.Module):
    def __init__(self, cfg, adata):
        super().__init__()
        self.config = cfg
        ensure_cell_type_codes(adata, cfg.cell_type_key)
        self.input_dim = adata.n_vars
        self.latent_dim = cfg.hidden_dim
        self.n_types = int(adata.obs[cfg.cell_type_key].max()) + 1

        self.gene_encoder = _build_deep_mlp(
            self.input_dim,
            self.latent_dim,
            cfg,
            activation=nn.ELU(),
            use_layernorm=True,
        )
        self.use_cell_type_embedding = bool(getattr(cfg, "use_cell_type_embedding", True))
        self.use_residual_drift = bool(getattr(cfg, "use_residual_drift", True))
        self.use_state_momentum = bool(getattr(cfg, "use_state_momentum", True))
        self.type_embed = (
            nn.Embedding(self.n_types, self.latent_dim) if self.use_cell_type_embedding else None
        )
        self.use_stage_embedding = bool(getattr(cfg, "use_stage_embedding", False))
        self.stage_cond_key = getattr(cfg, "stage_cond_key", None)
        n_stages = getattr(cfg, "n_stages", None)
        if self.use_stage_embedding:
            if n_stages is None and self.stage_cond_key and self.stage_cond_key in adata.obs.columns:
                n_stages = int(adata.obs[self.stage_cond_key].max()) + 1
            if n_stages is None or int(n_stages) < 1:
                raise ValueError("use_stage_embedding=True requires n_stages or stage_cond_key in adata.obs")
            self.n_stages = int(n_stages)
            self.stage_embed = nn.Embedding(self.n_stages, self.latent_dim)
        else:
            self.n_stages = None
            self.stage_embed = None
        self.potential_net = PotentialNetwork(self.latent_dim, cfg)
        self.residual_net = (
            ResidualDriftNetwork(self.latent_dim, cfg) if self.use_residual_drift else None
        )
        self.momentum_net = (
            MomentumNetwork(self.latent_dim, cfg) if self.use_state_momentum else None
        )
        self.sde_func = HamiltonianFlowFunc(
            self.potential_net, self.latent_dim, cfg, residual_net=self.residual_net
        )
        self.flow_func = self.sde_func
        # Legacy / diagnostic global momentum. Retained as a fallback when the
        # state-dependent momentum network is disabled and for downstream tools.
        self.register_buffer(
            "_momentum_ema",
            torch.zeros(1, self.latent_dim),
            persistent=False,
        )
        self.predictor = _build_deep_mlp(
            self.latent_dim + 1,
            self.input_dim,
            cfg,
            activation=nn.SiLU(),
            use_layernorm=True,
            final_activation=nn.Softplus(),
        )
        self.pseudotime_head = nn.Sequential(
            nn.Linear(self.latent_dim + 1, 1),
            nn.Softplus(),
        )
        self._time_grid = None

    def encode(self, x, cell_type, stage=None):
        z = self.gene_encoder(x)
        if self.type_embed is not None and self.use_cell_type_embedding:
            z = z + self.type_embed(cell_type.long())
        if self.stage_embed is not None and stage is not None:
            z = z + self.stage_embed(stage.long())
        return z

    def predict_expression(self, z, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.predictor(torch.cat([z, t], dim=-1))

    def potential(self, z, t):
        return self.potential_net(z, t)

    def stationary_potential(self, z):
        """Time-invariant quasi-potential U0(z) (equals U(z, 0) in time_varying mode)."""
        return self.potential_net.stationary_potential(z)

    def initial_momentum(self, z, t):
        """Per-cell initial momentum p_theta(z, t).

        Falls back to the global EMA momentum vector when the state-dependent momentum
        network is disabled (``use_state_momentum=False``, legacy behaviour).
        """
        if self.momentum_net is not None:
            return self.momentum_net(z, t)
        return self._momentum_ema.expand(z.shape[0], -1)

    def hj_regularizer(self, z, t):
        """L_energy; checkpoint field name kept as hjb_residual."""
        if getattr(self.config, "use_total_drift_hjb", False):
            return self.hj_total_drift_regularizer(z, t)
        return energy_regularizer(self.potential_net, z, t)

    def hj_total_drift_regularizer(self, z, t):
        """Total-drift-aware HJ regularizer, falling back to gradient-only when drift is disabled."""
        return total_drift_energy_regularizer(
            self.potential_net,
            self.sde_func.residual_drift,
            z,
            t,
            residual_drift_mode=getattr(self.config, "residual_drift_mode", "velocity"),
        )

    def momentum_loss(self, p_init, p_final, z_curr, z_pred, t_curr, t_next):
        dt = (t_next - t_curr).clamp(min=1e-6).unsqueeze(-1)
        mode = getattr(self.config, "momentum_loss_type", "velocity")
        mode = (mode or "velocity").lower()
        if mode not in {"velocity", "force", "both"}:
            raise ValueError(f"Unknown momentum_loss_type={mode!r}")

        velocity_target = (z_pred.detach() - z_curr) / dt
        loss_velocity = F.mse_loss(p_final, velocity_target)

        t_in = t_curr.unsqueeze(1) if t_curr.dim() == 1 else t_curr
        grad_u = self.sde_func.grad_potential(z_curr, t_in)
        residual_mode = getattr(self.config, "residual_drift_mode", "velocity")
        if getattr(self.config, "use_residual_drift", True) and residual_mode == "force":
            r_force = self.sde_func.residual_drift(z_curr, t_in)
        else:
            r_force = torch.zeros_like(z_curr)
        gamma = float(getattr(self.config, "hamiltonian_damping_gamma", 0.1))
        force_residual = (p_final - p_init) / dt + grad_u + gamma * p_init - r_force
        loss_force = torch.mean(force_residual ** 2)

        if mode == "velocity":
            return loss_velocity
        if mode == "force":
            return loss_force
        return 0.5 * (loss_velocity + loss_force)

    def hjb_residual(self, z, t):
        """Backward-compatible alias for :meth:`hj_regularizer`."""
        return self.hj_regularizer(z, t)

    def hjb_residual_total_drift(self, z, t):
        """Backward-compatible alias for :meth:`hj_total_drift_regularizer`."""
        return self.hj_total_drift_regularizer(z, t)

    def drift_decomposition(self, z, t, eps=1e-8):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        z_req = z.detach().clone().requires_grad_(True)
        pot = self.potential_net(z_req, t)
        grad_z = torch.autograd.grad(
            pot, z_req, torch.ones_like(pot), create_graph=True, retain_graph=False
        )[0]
        grad_drift = -grad_z
        residual_drift = self.sde_func.residual_drift(z, t)
        total_drift = grad_drift + residual_drift
        grad_norm = torch.linalg.norm(grad_drift, dim=1)
        residual_norm = torch.linalg.norm(residual_drift, dim=1)
        total_norm = torch.linalg.norm(total_drift, dim=1)
        residual_ratio = residual_norm / (grad_norm + residual_norm + eps)
        return {
            "grad_drift": grad_drift,
            "residual_drift": residual_drift,
            "total_drift": total_drift,
            "residual_ratio": residual_ratio,
            "grad_norm": grad_norm,
            "residual_norm": residual_norm,
            "total_norm": total_norm,
        }

    def _build_time_grid(self, device):
        times = sorted(set([self.config.start_time] + list(self.config.train_time)))
        self._time_grid = torch.tensor(times, dtype=torch.float32, device=device)
        return self._time_grid

    @staticmethod
    def _sorted_unique_times(ts, device, dtype=torch.float32):
        ts = ts.to(device=device, dtype=dtype)
        if ts.numel() == 0:
            return ts
        return torch.sort(ts.unique())[0]

    def integrate_latent(
        self,
        z0,
        ts,
        p0=None,
        return_momentum=False,
        detach_potential=False,
        add_noise=None,
        z_curr=None,
        z_target=None,
    ):
        ts = ts.to(dtype=z0.dtype, device=z0.device)
        if ts.numel() < 2:
            traj = z0.unsqueeze(0)
            p_out = p0 if p0 is not None else self._momentum_ema.expand(z0.shape[0], -1)
            return (traj, p_out) if return_momentum else traj
        if getattr(self.config, "use_hamiltonian_flow", True):
            if p0 is None:
                t0 = ts[0].reshape(1, 1).expand(z0.shape[0], 1)
                p0 = self.initial_momentum(z0, t0)
            z_anchor = z0 if z_curr is None else z_curr
            z_tgt = z_target.detach() if z_target is not None else None
            traj, p_final = integrate_hamiltonian_flow(
                self.flow_func,
                z0,
                p0,
                ts,
                dt=self.config.sde_dt,
                add_noise=self.training if add_noise is None else add_noise,
                detach_potential=detach_potential,
                z_curr=z_anchor,
                z_target=z_tgt,
            )
            return (traj, p_final) if return_momentum else traj
        traj = sde.sdeint_adjoint(
            self.sde_func,
            z0,
            ts,
            method="euler",
            dt=self.config.sde_dt,
            dt_min=1e-4,
            adjoint_method="euler",
            names={"drift": "f", "diffusion": "g"},
        )
        p_out = p0 if p0 is not None else torch.zeros_like(z0)
        return (traj, p_out) if return_momentum else traj

    def forward(
        self,
        x_curr,
        x_next,
        cell_type,
        t_curr,
        t_next,
        z_curr_override=None,
        ot_tgt_z=None,
        stage=None,
    ):
        device = x_curr.device
        if z_curr_override is not None:
            z_curr = z_curr_override
        else:
            z_curr = self.encode(x_curr, cell_type, stage=stage)
        z_next = self.encode(x_next, cell_type, stage=stage)

        if t_curr.dim() == 0:
            t_curr = t_curr.unsqueeze(0).expand(x_curr.shape[0])
        if t_next.dim() == 0:
            t_next = t_next.unsqueeze(0).expand(x_next.shape[0])

        pot = self.potential(z_curr, t_curr)
        pot_stationary = self.stationary_potential(z_curr).squeeze(-1)
        energy = self.hj_regularizer(z_curr, t_curr)
        r_theta = self.sde_func.residual_drift(z_curr, t_curr.unsqueeze(1))
        drift_decomp = self.drift_decomposition(z_curr, t_curr)

        p_init = self.initial_momentum(z_curr, t_curr)
        if self.training and getattr(self.config, "momentum_ema", 0.0) > 0 and z_curr_override is None:
            with torch.no_grad():
                dt_est = (t_next - t_curr).clamp(min=1e-6).unsqueeze(-1)
                vel = (z_next - z_curr) / dt_est
                ema = float(getattr(self.config, "momentum_ema", 0.9))
                self._momentum_ema.copy_(
                    ema * self._momentum_ema + (1.0 - ema) * vel.mean(dim=0, keepdim=True)
                )

        full_grid = self._build_time_grid(device)
        t0 = float(t_curr[0].item())
        t1 = float(t_next[0].item())
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        mask = (full_grid >= lo - 1e-6) & (full_grid <= hi + 1e-6)
        ts = full_grid[mask]
        if ts.numel() == 0 or abs(ts[0].item() - t0) > 1e-5:
            ts = torch.cat([torch.tensor([t0], device=device), ts])
        if abs(ts[-1].item() - t1) > 1e-5:
            ts = torch.cat([ts, torch.tensor([t1], device=device, dtype=ts.dtype)])
        ts = self._sorted_unique_times(ts, device, dtype=ts.dtype)

        z_target = ot_tgt_z if ot_tgt_z is not None else z_next
        traj, p_final = self.integrate_latent(
            z_curr,
            ts,
            p0=p_init,
            return_momentum=True,
            z_curr=z_curr,
            z_target=z_target,
        )
        idx_end = (ts - t1).abs().argmin()
        z_pred = traj[idx_end]
        mom = self.momentum_loss(p_init, p_final, z_curr, z_pred, t_curr, t_next)

        # Potential-gated prediction for the population displacement/direction loss.
        z_pred_disp = z_pred
        want_disp = float(getattr(self.config, "lambda_lat_disp", 0.0) or 0.0) > 0
        if (
            self.training
            and want_disp
            and bool(getattr(self.config, "latent_disp_detach_potential", False))
            and getattr(self.config, "use_hamiltonian_flow", True)
        ):
            traj_g, _ = self.integrate_latent(
                z_curr,
                ts,
                p0=p_init,
                return_momentum=True,
                detach_potential=True,
                add_noise=False,
                z_curr=z_curr,
                z_target=z_target,
            )
            z_pred_disp = traj_g[idx_end]

        expr_pred = self.predict_expression(z_pred, t_next)
        expr_curr = self.predict_expression(z_curr, t_curr)
        pseudotime = self.pseudotime_head(
            torch.cat([z_next, t_next.unsqueeze(1)], dim=1)
        ).squeeze(-1)

        return {
            "latent_trajectory": traj,
            "time_grid": ts,
            "z_curr": z_curr,
            "z_pred": z_pred,
            "z_pred_disp": z_pred_disp,
            "expr_pred": expr_pred,
            "expr_curr": expr_curr,
            "x_curr_input": x_curr,
            "x_next_input": x_next,
            "t_curr": t_curr,
            "t_next": t_next,
            "potential": pot.squeeze(-1),
            "potential_stationary": pot_stationary,
            "hjb_residual": energy.squeeze(-1),
            "hj_regularizer": energy.squeeze(-1),
            "energy": energy.squeeze(-1),
            "momentum": mom,
            "momentum_init": p_init,
            "momentum_state": p_final,
            "residual_drift": r_theta,
            "residual_ratio": drift_decomp["residual_ratio"].squeeze(-1),
            "grad_norm": drift_decomp["grad_norm"].squeeze(-1),
            "residual_norm": drift_decomp["residual_norm"].squeeze(-1),
            "total_norm": drift_decomp["total_norm"].squeeze(-1),
            "pseudotime": pseudotime,
            "z_next_emb": z_next,
        }


class ValidationSplit:
    """Held-out indices for true validation (not training-batch self-report)."""

    def __init__(self, adata, cfg):
        self.cfg = cfg
        self.train_mask = np.ones(adata.n_obs, dtype=bool)
        self.val_mask = np.zeros(adata.n_obs, dtype=bool)
        self.val_transitions = []
        self.split_info: dict = {}
        self._split(adata)
        self._log_split_summary()

    def _split(self, adata):
        obs = adata.obs
        n = adata.n_obs
        rng = np.random.RandomState(self.cfg.seed)
        time_key = self.cfg.time_key
        all_times = sorted(set(obs[time_key].astype(float).unique()))
        self.split_info = {
            "val_mode": self.cfg.val_mode,
            "all_times": [float(t) for t in all_times],
            "n_cells_total": int(n),
        }

        if self.cfg.val_mode == "cells":
            idx = np.arange(n)
            tr, va = train_test_split(idx, test_size=self.cfg.val_ratio, random_state=self.cfg.seed)
            self.train_mask[va] = False
            self.val_mask[va] = True
            self.split_info.update(
                {
                    "holdout_time": None,
                    "train_times": [float(t) for t in all_times],
                    "val_times": [float(t) for t in all_times],
                    "n_train_cells": int(self.train_mask.sum()),
                    "n_val_cells": int(self.val_mask.sum()),
                }
            )

        elif self.cfg.val_mode == "time":
            times = np.asarray(all_times, dtype=float)
            if len(times) < 2:
                warnings.warn(
                    f"val_mode=time 仅有 {len(times)} 个时间点，无法 hold-out 整点验证；"
                    "回退为 cells 式随机划分。",
                    UserWarning,
                    stacklevel=2,
                )
                idx = np.arange(n)
                tr, va = train_test_split(idx, test_size=self.cfg.val_ratio, random_state=self.cfg.seed)
                self.train_mask[va] = False
                self.val_mask[va] = True
                self.split_info["fallback"] = "cells_random"
            else:
                if self.cfg.val_time_point is not None:
                    hold = float(self.cfg.val_time_point)
                else:
                    hold = float(times[-1])
                if hold not in set(times.tolist()):
                    warnings.warn(
                        f"val_time_point={hold} 不在数据时间点 {times.tolist()} 中，"
                        f"改用最后时间点 {float(times[-1])}。",
                        UserWarning,
                        stacklevel=2,
                    )
                    hold = float(times[-1])
                t_arr = obs[time_key].astype(float).values
                self.val_mask = t_arr == hold
                self.train_mask = ~self.val_mask
                train_times = sorted(set(t_arr[self.train_mask].tolist()))
                self.split_info.update(
                    {
                        "holdout_time": hold,
                        "train_times": [float(t) for t in train_times],
                        "val_times": [hold],
                        "n_train_cells": int(self.train_mask.sum()),
                        "n_val_cells": int(self.val_mask.sum()),
                    }
                )
                if self.val_mask.sum() == 0:
                    warnings.warn(
                        f"val_mode=time hold-out t={hold} 无细胞，验证指标可能为 NaN。",
                        UserWarning,
                        stacklevel=2,
                    )

        elif self.cfg.val_mode == "cell_type":
            types = obs[self.cfg.cell_type_key].unique()
            n_hold = max(1, int(len(types) * self.cfg.val_ratio))
            hold_types = set(rng.choice(types, size=n_hold, replace=False))
            self.val_mask = obs[self.cfg.cell_type_key].isin(hold_types).values
            self.train_mask = ~self.val_mask
            self.split_info.update(
                {
                    "holdout_cell_types": sorted(str(t) for t in hold_types),
                    "n_train_cells": int(self.train_mask.sum()),
                    "n_val_cells": int(self.val_mask.sum()),
                }
            )

        elif self.cfg.val_mode == "patients":
            patient_key = getattr(self.cfg, "patient_key", "patient_id")
            if patient_key not in obs.columns:
                raise ValueError(
                    f"val_mode=patients requires obs column {patient_key!r}"
                )
            patients = np.asarray(obs[patient_key].astype(str).unique())
            n_hold = max(1, int(round(len(patients) * float(self.cfg.val_ratio))))
            n_hold = min(n_hold, max(1, len(patients) - 1))
            hold_patients = set(rng.choice(patients, size=n_hold, replace=False))
            self.val_mask = obs[patient_key].astype(str).isin(hold_patients).values
            self.train_mask = ~self.val_mask
            self.split_info.update(
                {
                    "holdout_patients": sorted(hold_patients),
                    "train_patients": sorted(
                        set(patients.tolist()) - hold_patients
                    ),
                    "train_times": [float(t) for t in all_times],
                    "val_times": [float(t) for t in all_times],
                    "n_train_cells": int(self.train_mask.sum()),
                    "n_val_cells": int(self.val_mask.sum()),
                }
            )

        elif self.cfg.val_mode == "time_extrapolate":
            times = list(all_times)
            if len(times) < 2:
                warnings.warn(
                    "time_extrapolate 至少需要 2 个时间点；回退为 cells 随机划分。",
                    UserWarning,
                    stacklevel=2,
                )
                idx = np.arange(n)
                tr, va = train_test_split(idx, test_size=self.cfg.val_ratio, random_state=self.cfg.seed)
                self.train_mask[va] = False
                self.val_mask[va] = True
                self.split_info["fallback"] = "cells_random"
            else:
                if len(times) < 3:
                    warnings.warn(
                        f"time_extrapolate 仅有 {len(times)} 个时间点，外推验证可能不稳定；"
                        f"train={times[:-1]} val={[times[-1]]}。",
                        UserWarning,
                        stacklevel=2,
                    )
                split_at = max(1, int(len(times) * (1 - self.cfg.val_ratio)))
                if split_at >= len(times):
                    split_at = len(times) - 1
                train_times = set(times[:split_at])
                val_times = set(times[split_at:])
                t_arr = obs[time_key].astype(float).values
                self.train_mask = np.array([t in train_times for t in t_arr])
                self.val_mask = np.array([t in val_times for t in t_arr])
                all_pairs = get_training_transitions(self.cfg)
                self.val_transitions = [
                    (a, b) for a, b in all_pairs if a in train_times and b in val_times
                ]
                self.split_info.update(
                    {
                        "holdout_time": None,
                        "train_times": [float(t) for t in sorted(train_times)],
                        "val_times": [float(t) for t in sorted(val_times)],
                        "n_train_cells": int(self.train_mask.sum()),
                        "n_val_cells": int(self.val_mask.sum()),
                        "val_transition_pairs": list(self.val_transitions),
                    }
                )
                if not self.val_transitions:
                    warnings.warn(
                        "time_extrapolate: 无跨越 train→val 时间的转移对，ValPCC/ValMSE 可能为 NaN。",
                        UserWarning,
                        stacklevel=2,
                    )
                if self.val_mask.sum() == 0:
                    warnings.warn(
                        "time_extrapolate: 验证时间窗无细胞，ValPCC/ValMSE 可能为 NaN。",
                        UserWarning,
                        stacklevel=2,
                    )
        else:
            raise ValueError(f"未知 val_mode: {self.cfg.val_mode}")

        if self.cfg.val_mode == "time" and self.split_info.get("holdout_time") is not None:
            hold = float(self.split_info["holdout_time"])
            all_pairs = get_training_transitions(self.cfg)
            self.val_transitions = [
                (a, b) for a, b in all_pairs if np.isclose(float(b), hold)
            ]
            self.split_info["val_transition_pairs"] = list(self.val_transitions)
        elif self.cfg.val_mode != "time_extrapolate" or not self.val_transitions:
            self.val_transitions = get_training_transitions(self.cfg)
            if "val_transition_pairs" not in self.split_info:
                self.split_info["val_transition_pairs"] = list(self.val_transitions)

    def _log_split_summary(self):
        info = self.split_info
        print(
            f"Validation split | mode={info.get('val_mode')} | "
            f"train_cells={info.get('n_train_cells', '?')} "
            f"val_cells={info.get('n_val_cells', '?')}",
            flush=True,
        )
        if info.get("holdout_time") is not None:
            print(f"  hold-out time: {info['holdout_time']}", flush=True)
        if info.get("train_times") is not None:
            print(f"  train times: {info.get('train_times')}", flush=True)
        if info.get("val_times") is not None:
            print(f"  val times: {info.get('val_times')}", flush=True)
        if info.get("val_transition_pairs"):
            print(f"  val transition pairs: {info['val_transition_pairs']}", flush=True)
        if info.get("holdout_patients"):
            print(f"  hold-out patients: {info['holdout_patients']}", flush=True)
        if info.get("fallback"):
            print(f"  WARNING: split fallback={info['fallback']}", flush=True)

    def filter_pairs(self, pairs, adata, for_validation=False):
        obs = adata.obs
        mask = self.val_mask if for_validation else self.train_mask
        filtered = []
        for curr_i, next_i in pairs:
            if mask[curr_i] and mask[next_i]:
                filtered.append((curr_i, next_i))
        return filtered


def gaussian_mmd_loss(x_pred, x_target, sigmas=(1, 2, 4, 8, 16)):
    """Multi-scale Gaussian-kernel MMD between predicted and target expression batches."""
    x_pred = x_pred.reshape(x_pred.shape[0], -1)
    x_target = x_target.reshape(x_target.shape[0], -1)
    xx = torch.cdist(x_pred, x_pred, p=2)
    yy = torch.cdist(x_target, x_target, p=2)
    xy = torch.cdist(x_pred, x_target, p=2)
    loss = torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * float(sigma) ** 2)
        k_xx = torch.exp(-gamma * xx ** 2).mean()
        k_yy = torch.exp(-gamma * yy ** 2).mean()
        k_xy = torch.exp(-gamma * xy ** 2).mean()
        loss = loss + k_xx + k_yy - 2.0 * k_xy
    return loss / max(len(sigmas), 1)


LOSS_PART_KEYS = (
    "ot",
    "recon",
    "energy",
    "momentum",
    "density",
    "delta",
    "direction",
    "latent",
    "kinetic",
    "lat_disp",
    "residual_balance",
    "hamiltonian",
)
LOSS_PART_KEYS_LEGACY = ("ot", "recon", "hjb", "pseudo", "residual", "lat_dir")

# Fixed internal mix ratios for the simplified loss (see reconstruction_loss_weights).
RECON_MMD_MIX_RATIO = 5.0  # mse_mmd: L_recon = lambda_recon * (L_mse + ratio * L_mmd)
INERTIA_MOMENTUM_MIX = 0.025  # L_inertia = mix * L_momentum + L_kinetic; scaled by lambda_kinetic


def reconstruction_loss_weights(cfg) -> tuple[float, float]:
    """Return (w_mse, w_mmd) multipliers for pair-MSE and MMD reconstruction terms."""
    legacy_mse = getattr(cfg, "_legacy_lambda_pair_mse", None)
    legacy_mmd = getattr(cfg, "_legacy_lambda_mmd", None)
    if legacy_mse is not None or legacy_mmd is not None:
        return (
            float(legacy_mse if legacy_mse is not None else 0.0),
            float(legacy_mmd if legacy_mmd is not None else 0.0),
        )
    mode = getattr(cfg, "reconstruction_mode", "mse_mmd")
    scale = float(getattr(cfg, "lambda_recon", 0.01) or 0.0)
    if mode == "mse":
        return scale, 0.0
    if mode == "mmd_only":
        return 0.0, scale
    return scale, scale * RECON_MMD_MIX_RATIO


def build_reconstruction_loss(cfg, pair_mse_loss, mmd_loss):
    w_mse, w_mmd = reconstruction_loss_weights(cfg)
    return w_mse * pair_mse_loss + w_mmd * mmd_loss


def _mean_loss_parts(parts_list):
    if not parts_list:
        return {k: float("nan") for k in LOSS_PART_KEYS}
    out = {}
    for k in LOSS_PART_KEYS:
        vals = [p[k].item() for p in parts_list if k in p]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


class LossNormEMA:
    """EMA scale estimates for optional per-term loss normalization."""

    def __init__(self, momentum: float = 0.98, eps: float = 1e-8):
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.ema = {k: None for k in LOSS_PART_KEYS}

    def update(self, parts: dict):
        for key in parts:
            val = parts[key]
            x = float(val.item() if torch.is_tensor(val) else val)
            if self.ema[key] is None:
                self.ema[key] = x
            else:
                self.ema[key] = self.momentum * self.ema[key] + (1.0 - self.momentum) * x

    def normalize(self, parts: dict) -> dict:
        out = {}
        for key in parts:
            scale = self.ema[key]
            if scale is None:
                out[key] = parts[key]
            else:
                out[key] = parts[key] / (scale + self.eps)
        return out

    def as_dict(self) -> dict:
        return {k: (None if v is None else float(v)) for k, v in self.ema.items()}


class SDETrainer:
    def __init__(self, model, adata, cfg, save_dir):
        ensure_cell_type_codes(adata, cfg.cell_type_key)
        warn_gradient_only_hj_regularizer(cfg)
        self.model = model.to(cfg.device)
        self.adata = adata
        self.config = cfg
        self.save_dir = save_dir
        self.ot_loss = SamplesLoss("sinkhorn", p=2, blur=0.005, scaling=0.5, debias=True)
        # Latent-space OT for the latent-consistency regularizer (blur tuned for latent scale;
        # sinkhorn keeps non-vanishing gradients even when z_pred has diverged far away).
        self.latent_ot_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True)
        if getattr(cfg, "loss_recipe", "legacy") == "hamiltonian":
            cfg.use_loss_normalization = False
        self.val_split = ValidationSplit(adata, cfg)
        self.transitions = get_training_transitions(cfg)
        self.train_transitions = self._resolve_train_transitions()
        self.loss_norm_ema = LossNormEMA(
            momentum=float(getattr(cfg, "loss_norm_momentum", 0.98)),
            eps=float(getattr(cfg, "loss_norm_eps", 1e-8)),
        )
        self._ot_tgt_cache = None
        if bool(getattr(cfg, "latent_disp_fullpop_ot", False)) and bool(
            getattr(cfg, "latent_disp_ot_coupling", False)
        ):
            self._build_fullpop_ot_cache()
        self.best_val_loss = float("inf")
        self.best_val_pcc = float("-inf")
        self.best_val_mse = float("inf")
        self.best_val_mse_at_best_pcc = float("inf")
        self.early_stop_best_pcc = float("-inf")
        self.early_stop_best_mse = float("inf")
        self.early_stop_best_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.metrics = {
            "train_loss": [],
            "train_mse": [],
            "train_pcc": [],
            "val_loss": [],
            "val_mse": [],
            "val_pcc": [],
            "hjb": [],
            "residual_norm": [],
            "potential_logp_pearson": None,
            "potential_logp_spearman": None,
            "ph": None,
            "bdr": None,
            "params": sum(p.numel() for p in model.parameters()),
            "train_time": 0.0,
        }
        lambda_density = float(getattr(cfg, "lambda_density", 0.0) or 0.0)
        if getattr(cfg, "use_density_regularization", False) or lambda_density > 0:
            from density_regularization import (
                attach_density_target_to_adata,
                attach_density_target_within_type,
            )

            if getattr(cfg, "density_use_latent_batch", True):
                print(
                    f"Density regularization: lambda={lambda_density} "
                    f"mode=latent_batch within_type={getattr(cfg, 'density_within_cell_type', True)} "
                    f"align_stationary={getattr(cfg, 'density_align_stationary', True)}",
                    flush=True,
                )
            elif getattr(cfg, "density_within_cell_type", False):
                attach_density_target_within_type(
                    self.adata,
                    basis=getattr(cfg, "density_basis", "X_pca"),
                    target_key=getattr(cfg, "density_target_key", "density_neglogp"),
                    cell_type_key=cfg.cell_type_key,
                    n_pcs=int(getattr(cfg, "density_n_pcs", 20)),
                    bandwidth=getattr(cfg, "density_bandwidth", None),
                )
            else:
                attach_density_target_to_adata(
                    self.adata,
                    basis=getattr(cfg, "density_basis", "X_pca"),
                    target_key=getattr(cfg, "density_target_key", "density_neglogp"),
                    n_pcs=int(getattr(cfg, "density_n_pcs", 20)),
                    bandwidth=getattr(cfg, "density_bandwidth", None),
                )
            cfg.use_density_regularization = True
            if not getattr(cfg, "density_use_latent_batch", True):
                print(
                    f"Density regularization: lambda={lambda_density} "
                    f"basis={getattr(cfg, 'density_basis', 'X_pca')} "
                    f"n_pcs={getattr(cfg, 'density_n_pcs', 20)}",
                    flush=True,
                )

    def _build_fullpop_ot_cache(self):
        """Precompute per-cell OT barycentric targets on full train populations."""
        print("Building full-population OT displacement cache...", flush=True)
        blur = float(getattr(self.config, "latent_disp_ot_blur", 0.05) or 0.05)
        train_mask = self.val_split.train_mask
        obs = self.adata.obs
        tvals = obs[self.config.time_key].astype(float).values
        device = self.config.device
        cache = {}
        was_training = self.model.training
        self.model.eval()

        stage_key = getattr(self.config, "stage_cond_key", None)

        def _encode_indices(idxs):
            if len(idxs) == 0:
                return torch.zeros((0, self.model.latent_dim), device=device)
            if sp.issparse(self.adata.X):
                x = torch.tensor(self.adata.X[idxs].toarray(), dtype=torch.float32, device=device)
            else:
                x = torch.tensor(self.adata.X[idxs], dtype=torch.float32, device=device)
            ct = torch.tensor(
                obs[self.config.cell_type_key].values[idxs], dtype=torch.long, device=device
            )
            stage = None
            if stage_key and stage_key in obs.columns:
                stage = torch.tensor(obs[stage_key].values[idxs], dtype=torch.long, device=device)
            with torch.no_grad():
                return self.model.encode(x, ct, stage=stage)

        group_keys = list(getattr(self.config, "pair_group_keys", None) or [])
        for t_curr, t_next in self.train_transitions:
            key = (float(t_curr), float(t_next))
            cache[key] = {}
            if group_keys:
                group_frame = obs[group_keys].astype(str)
                combos = set()
                for i in np.where(train_mask)[0]:
                    combos.add(
                        tuple(group_frame.iloc[i].tolist())
                        + (obs[self.config.cell_type_key].values[i],)
                    )
                group_iters = sorted(combos)
            else:
                group_iters = [
                    (ct,) for ct in np.unique(obs[self.config.cell_type_key].values)
                ]
            for combo in group_iters:
                *gvals, ct = combo if group_keys else (combo[0],)
                if group_keys:
                    g_mask = np.ones(len(obs), dtype=bool)
                    for gk, gv in zip(group_keys, gvals):
                        g_mask &= obs[gk].astype(str).values == gv
                    ct_mask = obs[self.config.cell_type_key].values == ct
                    base = g_mask & ct_mask
                else:
                    base = obs[self.config.cell_type_key].values == ct
                curr_mask = base & train_mask & np.isclose(tvals, float(t_curr))
                next_mask = base & train_mask & np.isclose(tvals, float(t_next))
                curr_idx = np.where(curr_mask)[0]
                if len(curr_idx) < 1 or next_mask.sum() < 1:
                    continue
                zc = _encode_indices(curr_idx)
                next_idx = np.where(next_mask)[0]
                zn = _encode_indices(next_idx)
                with torch.no_grad():
                    plan = _sinkhorn_plan(zc.detach(), zn.detach(), blur=blur)
                    row = plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    tgt = (plan @ zn) / row
                for i, obs_i in enumerate(curr_idx):
                    cache[key][int(obs_i)] = tgt[i].detach().cpu()
            print(
                f"  OT cache {t_curr}->{t_next}: {len(cache[key])} source cells",
                flush=True,
            )
        self._ot_tgt_cache = cache
        if was_training:
            self.model.train()
        print("Full-population OT cache ready.", flush=True)

    def _lookup_ot_tgt(self, t_curr, t_next, curr_idx):
        if self._ot_tgt_cache is None:
            return None
        key = (float(t_curr), float(t_next))
        trans = self._ot_tgt_cache.get(key)
        if not trans:
            return None
        try:
            rows = [trans[int(i)] for i in curr_idx]
        except KeyError:
            return None
        return torch.stack(rows, dim=0).to(self.config.device)

    def _resolve_train_transitions(self):
        """Filter transition pairs so training never samples hold-out / val-only times."""
        info = self.val_split.split_info
        pairs = list(self.transitions)
        if self.config.val_mode == "time" and info.get("holdout_time") is not None:
            hold = float(info["holdout_time"])
            pairs = [
                (a, b)
                for a, b in pairs
                if not np.isclose(float(a), hold) and not np.isclose(float(b), hold)
            ]
        elif self.config.val_mode == "time_extrapolate" and info.get("train_times"):
            train_times = {float(t) for t in info["train_times"]}
            pairs = [
                (a, b)
                for a, b in pairs
                if float(a) in train_times and float(b) in train_times
            ]
        return pairs

    def _resolve_val_transitions(self):
        info = self.val_split.split_info
        if (
            self.config.val_mode == "time_extrapolate"
            and self.val_split.val_transitions
        ):
            return list(self.val_split.val_transitions)
        if self.config.val_mode == "time" and info.get("holdout_time") is not None:
            hold = float(info["holdout_time"])
            return [(a, b) for a, b in self.transitions if np.isclose(float(b), hold)]
        return list(self.transitions)

    def _build_rollout_chains(self):
        return build_adjacent_rollout_chains(self.config)

    def _effective_micro_batch_size(self):
        """Return micro-batch size for gradient accumulation under memory pressure."""
        bs = max(1, int(self.config.batch_size))
        micro = getattr(self.config, "micro_batch_size", None)
        if micro is None or int(micro) <= 0:
            return bs
        return max(1, min(bs, int(micro)))

    def _train_rollout_chain(self, time_chain, optimizer):
        """Multi-step BPTT rollout with exponentially decayed per-step losses."""
        chains = sample_rollout_chains(
            self.adata,
            time_chain,
            self.config.batch_size,
            self.config,
            cell_mask=self.val_split.train_mask,
        )
        if not chains:
            return None

        alpha = float(getattr(self.config, "rollout_decay_alpha", 0.5) or 0.5)
        z_roll = None
        total_loss = torch.zeros((), device=self.config.device)
        weight = 1.0
        agg_parts = []
        last_out = None
        last_batch = None

        for step_k, (t_curr, t_next) in enumerate(time_chain):
            curr_idx = [c[step_k] for c in chains]
            next_idx = [c[step_k + 1] for c in chains]
            batch = _load_batch(self.adata, curr_idx, next_idx, self.config)
            ot_tgt_z = self._lookup_ot_tgt(t_curr, t_next, curr_idx)
            out = _model_forward_from_batch(
                self.model, batch, z_curr_override=z_roll, ot_tgt_z=ot_tgt_z
            )
            loss, parts, norm_parts = self._compute_loss(
                out,
                batch["x_next"],
                batch["t_next"],
                batch["cell_type"],
                update_ema=(step_k == 0),
                density_curr=batch.get("density_curr"),
                ot_tgt_z=ot_tgt_z,
            )
            total_loss = total_loss + weight * loss
            agg_parts.append(parts)
            if norm_parts is not None:
                if not hasattr(self, "_epoch_train_norm_parts"):
                    self._epoch_train_norm_parts = []
                self._epoch_train_norm_parts.append(norm_parts)
            weight *= alpha
            z_roll = torch.nan_to_num(out["z_pred"], nan=0.0, posinf=1e4, neginf=-1e4)
            last_out = out
            last_batch = batch

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return {
            "loss": float(total_loss.item()),
            "parts": agg_parts[-1] if agg_parts else {},
            "out": last_out,
            "batch": last_batch,
        }

    def train(self):
        import os

        os.makedirs(self.save_dir, exist_ok=True)
        start_time = time.time()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.9)
        split = self.val_split.split_info
        meta_cols = {
            "ValMode": self.config.val_mode,
            "TrainTimes": str(split.get("train_times", "")),
            "ValTimes": str(split.get("val_times", "")),
            "HoldoutTime": split.get("holdout_time", ""),
        }
        base_columns = [
            "Epoch",
            "TrainLoss",
            "TrainMSE",
            "TrainPCC",
            "TrainOT",
            "TrainRecon",
            "TrainEnergy",
            "TrainMomentum",
            "TrainDensity",
            "TrainDeltaLoss",
            "TrainDirectionLoss",
            "ValLoss",
            "ValMSE",
            "ValPCC",
            "ValOT",
            "ValRecon",
            "ValEnergy",
            "ValMomentum",
            "ValDensity",
            "ValDeltaLoss",
            "ValDirectionLoss",
            "Energy",
            "PairMSELoss",
            "MMDLoss",
            "ReconstructionMode",
            "ResidualDriftMode",
            "MomentumLossType",
            "LambdaRecon",
            "ReconMmdMixRatio",
            "ValMode",
            "TrainTimes",
            "ValTimes",
            "HoldoutTime",
        ]
        if float(getattr(self.config, "lambda_density", 0.0) or 0.0) > 0:
            base_columns.extend(
                [
                    "DensityLoss",
                    "LambdaDensity",
                    "DensityBasis",
                    "DensityNPCs",
                ]
            )
        if getattr(self.config, "use_loss_normalization", False):
            base_columns.extend(
                [
                    "TrainOTNorm",
                    "TrainReconNorm",
                    "TrainEnergyNorm",
                    "TrainMomentumNorm",
                    "TrainDensityNorm",
                    "TrainDeltaNorm",
                    "TrainDirectionNorm",
                    "ValOTNorm",
                    "ValReconNorm",
                    "ValEnergyNorm",
                    "ValMomentumNorm",
                    "ValDensityNorm",
                    "ValDeltaNorm",
                    "ValDirectionNorm",
                    "EMA_OT",
                    "EMA_Recon",
                    "EMA_Energy",
                    "EMA_Momentum",
                    "EMA_Density",
                    "EMA_Delta",
                    "EMA_Direction",
                ]
            )
            print(
                "Loss normalization: ENABLED (EMA-scaled OT/recon/energy/momentum/density/delta/direction)",
                flush=True,
            )
        epoch_csv = pd.DataFrame(columns=base_columns)

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss, train_mse, train_pcc, energy_vals, mom_vals, train_parts = [], [], [], [], [], []
            pair_mse_vals, mmd_vals, density_vals = [], [], []

            use_rollout = bool(getattr(self.config, "use_multi_step_rollout", False))
            if use_rollout:
                rollout_chains = self._build_rollout_chains()
                for time_chain in rollout_chains:
                    result = self._train_rollout_chain(time_chain, optimizer)
                    if result is None:
                        continue
                    parts = result["parts"]
                    out = result["out"]
                    batch = result["batch"]
                    train_mse.append(parts["recon"].item())
                    energy_vals.append(parts["energy"].item())
                    mom_vals.append(parts["momentum"].item())
                    if "pair_mse_loss" in parts:
                        pair_mse_vals.append(float(parts["pair_mse_loss"]))
                    if "mmd_loss" in parts:
                        mmd_vals.append(float(parts["mmd_loss"]))
                    if "density" in parts:
                        density_vals.append(float(parts["density"].detach()))
                    train_parts.append(parts)
                    train_pcc.append(_batch_pcc(out["expr_pred"], batch["x_next"]))
                    epoch_loss.append(result["loss"])
                # Single-step transitions not covered by rollout chains (e.g. anchor jumps).
                rollout_edges = {
                    (float(a), float(b))
                    for chain in rollout_chains
                    for a, b in chain
                }
                extra_transitions = [
                    (a, b)
                    for a, b in self.train_transitions
                    if (float(a), float(b)) not in rollout_edges
                ]
            else:
                extra_transitions = list(self.train_transitions)

            for t_curr, t_next in extra_transitions:
                pairs = sample_transition_pairs(
                    self.adata,
                    t_curr,
                    t_next,
                    self.config.batch_size,
                    self.config,
                    cell_mask=self.val_split.train_mask,
                )
                if not pairs:
                    continue
                micro = self._effective_micro_batch_size()
                n_pairs = len(pairs)
                n_micro = int(np.ceil(n_pairs / float(micro)))
                optimizer.zero_grad()
                chunk_losses = []
                last_parts = None
                last_out = None
                last_batch = None
                last_norm_parts = None
                for mi in range(n_micro):
                    chunk = pairs[mi * micro : (mi + 1) * micro]
                    if not chunk:
                        continue
                    curr_idx, next_idx = zip(*chunk)
                    batch = _load_batch(self.adata, curr_idx, next_idx, self.config)
                    ot_tgt_z = self._lookup_ot_tgt(t_curr, t_next, curr_idx)
                    out = _model_forward_from_batch(self.model, batch, ot_tgt_z=ot_tgt_z)
                    loss, parts, norm_parts = self._compute_loss(
                        out,
                        batch["x_next"],
                        batch["t_next"],
                        batch["cell_type"],
                        update_ema=(mi == 0),
                        density_curr=batch.get("density_curr"),
                        ot_tgt_z=ot_tgt_z,
                    )
                    # Scale so accumulated grads match a single effective batch step.
                    (loss / float(n_micro)).backward()
                    chunk_losses.append(float(loss.detach().item()))
                    last_parts = parts
                    last_out = out
                    last_batch = batch
                    last_norm_parts = norm_parts
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                if last_parts is None:
                    continue
                train_mse.append(last_parts["recon"].item())
                energy_vals.append(last_parts["energy"].item())
                mom_vals.append(last_parts["momentum"].item())
                if "pair_mse_loss" in last_parts:
                    pair_mse_vals.append(float(last_parts["pair_mse_loss"]))
                if "mmd_loss" in last_parts:
                    mmd_vals.append(float(last_parts["mmd_loss"]))
                if "density" in last_parts:
                    density_vals.append(float(last_parts["density"].detach()))
                train_parts.append(last_parts)
                if last_norm_parts is not None:
                    if not hasattr(self, "_epoch_train_norm_parts"):
                        self._epoch_train_norm_parts = []
                    self._epoch_train_norm_parts.append(last_norm_parts)
                train_pcc.append(_batch_pcc(last_out["expr_pred"], last_batch["x_next"]))
                epoch_loss.append(float(np.mean(chunk_losses)) if chunk_losses else 0.0)

            val_metrics = self._validate()
            train_norm_avg = _mean_loss_parts(
                getattr(self, "_epoch_train_norm_parts", [])
            )
            self._epoch_train_norm_parts = []
            val_metrics["train_norm_parts"] = train_norm_avg

            n_steps = max(len(epoch_loss), 1)
            avg_train = float(np.mean(epoch_loss)) if epoch_loss else 0.0
            avg_train_mse = float(np.mean(train_mse)) if train_mse else 0.0
            avg_train_pcc = float(np.mean(train_pcc)) if train_pcc else 0.0
            avg_energy = float(np.mean(energy_vals)) if energy_vals else 0.0
            avg_momentum = float(np.mean(mom_vals)) if mom_vals else 0.0
            train_part_avg = _mean_loss_parts(train_parts)
            val_part_avg = val_metrics.get("parts", {k: float("nan") for k in LOSS_PART_KEYS})

            self.metrics["train_loss"].append(avg_train)
            self.metrics["train_mse"].append(avg_train_mse)
            self.metrics["train_pcc"].append(avg_train_pcc)
            self.metrics["val_loss"].append(val_metrics["loss"])
            self.metrics["val_mse"].append(val_metrics["mse"])
            self.metrics["val_pcc"].append(val_metrics["pcc"])
            self.metrics["hjb"].append(avg_energy)
            self.metrics["energy"] = self.metrics.get("energy", [])
            self.metrics["energy"].append(avg_energy)
            self.metrics["momentum"] = self.metrics.get("momentum", [])
            self.metrics["momentum"].append(avg_momentum)
            self.metrics["residual_norm"].append(float("nan"))
            scheduler.step()

            self._update_early_stop(val_metrics)
            ckpt_saved = self._checkpoint_if_improved(val_metrics, epoch + 1)

            ema_msg = ""
            if getattr(self.config, "use_loss_normalization", False):
                ema = self.loss_norm_ema.as_dict()

                def _fmt_ema(v):
                    return f"{v:.2e}" if v is not None else "n/a"

                ema_msg = (
                    f" | EMA_OT {_fmt_ema(ema['ot'])} EMA_Recon {_fmt_ema(ema['recon'])} "
                    f"EMA_Energy {_fmt_ema(ema.get('energy'))}"
                )
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"TrainLoss {avg_train:.4f} TrainMSE {avg_train_mse:.4f} TrainPCC {avg_train_pcc:.4f} | "
                f"ValLoss {val_metrics['loss']:.4f} ValMSE {val_metrics['mse']:.4f} ValPCC {val_metrics['pcc']:.4f} | "
                f"L_energy {avg_energy:.4e} L_momentum {avg_momentum:.4e}{ema_msg}"
                + (" *" if ckpt_saved else "")
            )
            row = [
                epoch + 1,
                avg_train,
                avg_train_mse,
                avg_train_pcc,
                train_part_avg["ot"],
                train_part_avg["recon"],
                train_part_avg["energy"],
                train_part_avg["momentum"],
                train_part_avg.get("density", float("nan")),
                train_part_avg.get("delta", float("nan")),
                train_part_avg.get("direction", float("nan")),
                val_metrics["loss"],
                val_metrics["mse"],
                val_metrics["pcc"],
                val_part_avg["ot"],
                val_part_avg["recon"],
                val_part_avg["energy"],
                val_part_avg["momentum"],
                val_part_avg.get("density", float("nan")),
                val_part_avg.get("delta", float("nan")),
                val_part_avg.get("direction", float("nan")),
                avg_energy,
                float(np.mean(pair_mse_vals)) if pair_mse_vals else float("nan"),
                float(np.mean(mmd_vals)) if mmd_vals else float("nan"),
                getattr(self.config, "reconstruction_mode", "mse"),
                getattr(self.config, "residual_drift_mode", "velocity"),
                getattr(self.config, "momentum_loss_type", "velocity"),
                float(getattr(self.config, "lambda_recon", 0.01) or 0.0),
                RECON_MMD_MIX_RATIO,
                meta_cols["ValMode"],
                meta_cols["TrainTimes"],
                meta_cols["ValTimes"],
                meta_cols["HoldoutTime"],
            ]
            if float(getattr(self.config, "lambda_density", 0.0) or 0.0) > 0:
                row.extend(
                    [
                        float(np.mean(density_vals)) if density_vals else float("nan"),
                        float(getattr(self.config, "lambda_density", 0.0)),
                        getattr(self.config, "density_basis", "X_pca"),
                        int(getattr(self.config, "density_n_pcs", 20)),
                    ]
                )
            if getattr(self.config, "use_loss_normalization", False):
                train_norm = val_metrics.get("train_norm_parts", {})
                val_norm = val_metrics.get("norm_parts", {})
                ema = self.loss_norm_ema.as_dict()
                row.extend(
                    [
                        train_norm.get("ot", float("nan")),
                        train_norm.get("recon", float("nan")),
                        train_norm.get("energy", float("nan")),
                        train_norm.get("momentum", float("nan")),
                        train_norm.get("density", float("nan")),
                        train_norm.get("delta", float("nan")),
                        train_norm.get("direction", float("nan")),
                        val_norm.get("ot", float("nan")),
                        val_norm.get("recon", float("nan")),
                        val_norm.get("energy", float("nan")),
                        val_norm.get("momentum", float("nan")),
                        val_norm.get("density", float("nan")),
                        val_norm.get("delta", float("nan")),
                        val_norm.get("direction", float("nan")),
                        ema.get("ot", float("nan")),
                        ema.get("recon", float("nan")),
                        ema.get("energy", float("nan")),
                        ema.get("momentum", float("nan")),
                        ema.get("density", float("nan")),
                        ema.get("delta", float("nan")),
                        ema.get("direction", float("nan")),
                    ]
                )
            epoch_csv.loc[epoch] = row

            patience = int(getattr(self.config, "early_stop_patience", 0) or 0)
            if patience > 0 and self.epochs_without_improvement >= patience:
                metric = getattr(self.config, "early_stop_metric", "loss")
                print(
                    f"Early stopping at epoch {epoch + 1}: no Val{metric.upper()} improvement "
                    f"for {patience} epochs (checkpoint epoch {self.best_epoch}, "
                    f"metric={getattr(self.config, 'checkpoint_metric', 'pcc')})."
                )
                break

        epoch_csv.to_csv(f"{self.save_dir}/Loss_epoch.csv", index=False)
        self.metrics["train_time"] = time.time() - start_time
        self.metrics["best_epoch"] = self.best_epoch
        self.metrics["best_val_pcc"] = self.best_val_pcc
        self.metrics["best_val_mse"] = self.best_val_mse
        self.metrics["best_val_loss"] = self.best_val_loss
        self.metrics["checkpoint_metric"] = getattr(self.config, "checkpoint_metric", "pcc")
        self.metrics["val_mode"] = self.config.val_mode
        self.metrics["val_split"] = dict(self.val_split.split_info)
        self.metrics["use_loss_normalization"] = bool(
            getattr(self.config, "use_loss_normalization", False)
        )
        if not getattr(self.config, "skip_final_evaluation", False):
            self.evaluate_model()
        else:
            print(
                f"\n=== 跳过最终评估 (smoke test) | best epoch {self.best_epoch} | "
                f"best ValPCC {self.best_val_pcc:.4f} ==="
            )

    def _update_early_stop(self, val_metrics):
        metric = getattr(self.config, "early_stop_metric", "pcc")
        min_delta = float(getattr(self.config, "early_stop_min_delta", 0.0) or 0.0)
        pcc = float(val_metrics.get("pcc", float("nan")))
        mse = float(val_metrics.get("mse", float("nan")))
        loss = float(val_metrics.get("loss", float("nan")))
        improved = False

        if metric in ("pcc", "pcc_then_mse"):
            if np.isfinite(pcc) and pcc > self.early_stop_best_pcc + min_delta:
                self.early_stop_best_pcc = pcc
                improved = True
        elif metric == "mse":
            if np.isfinite(mse) and mse < self.early_stop_best_mse - min_delta:
                self.early_stop_best_mse = mse
                improved = True
        elif metric == "loss":
            if np.isfinite(loss) and loss < self.early_stop_best_loss - min_delta:
                self.early_stop_best_loss = loss
                improved = True

        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def _checkpoint_if_improved(self, val_metrics, epoch):
        metric = getattr(self.config, "checkpoint_metric", "pcc")
        min_delta = float(getattr(self.config, "early_stop_min_delta", 0.0) or 0.0)
        tie_eps = float(getattr(self.config, "checkpoint_pcc_tie_epsilon", 0.005) or 0.005)
        pcc = float(val_metrics.get("pcc", float("nan")))
        mse = float(val_metrics.get("mse", float("nan")))
        loss = float(val_metrics.get("loss", float("nan")))
        save = False

        if metric == "pcc":
            if np.isfinite(pcc) and pcc > self.best_val_pcc + min_delta:
                self.best_val_pcc = pcc
                save = True
        elif metric == "mse":
            if np.isfinite(mse) and mse < self.best_val_mse - min_delta:
                self.best_val_mse = mse
                save = True
        elif metric == "loss":
            if np.isfinite(loss) and loss < self.best_val_loss - min_delta:
                self.best_val_loss = loss
                save = True
        elif metric == "pcc_then_mse":
            if np.isfinite(pcc) and pcc > self.best_val_pcc + min_delta:
                self.best_val_pcc = pcc
                self.best_val_mse_at_best_pcc = mse if np.isfinite(mse) else self.best_val_mse_at_best_pcc
                save = True
            elif (
                np.isfinite(pcc)
                and np.isfinite(mse)
                and pcc >= self.best_val_pcc - tie_eps
                and mse < self.best_val_mse_at_best_pcc - min_delta
            ):
                self.best_val_mse_at_best_pcc = mse
                save = True

        if save:
            self.best_epoch = int(epoch)
            if np.isfinite(mse):
                self.best_val_mse = min(self.best_val_mse, mse)
            torch.save(self.model.state_dict(), f"{self.save_dir}/best_model.pth")
        return save

    def _validate(self):
        if bool(getattr(self.config, "skip_epoch_validation", False)):
            empty_parts = {k: float("nan") for k in LOSS_PART_KEYS}
            return {
                "loss": float("nan"),
                "mse": float("nan"),
                "pcc": float("nan"),
                "parts": empty_parts,
                "norm_parts": {},
            }
        self.model.eval()
        losses, mses, pccs, val_parts, val_norm_parts = [], [], [], [], []
        transitions = self._resolve_val_transitions()
        micro = self._effective_micro_batch_size()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for t_curr, t_next in transitions:
            if self.config.val_mode in ("time", "time_extrapolate"):
                pairs = sample_transition_pairs(
                    self.adata,
                    t_curr,
                    t_next,
                    self.config.batch_size,
                    self.config,
                    curr_cell_mask=self.val_split.train_mask,
                    next_cell_mask=self.val_split.val_mask,
                )
            else:
                pairs = sample_transition_pairs(
                    self.adata,
                    t_curr,
                    t_next,
                    self.config.batch_size,
                    self.config,
                    cell_mask=self.val_split.val_mask,
                )
            if not pairs:
                continue
            for mi in range(0, len(pairs), micro):
                chunk = pairs[mi : mi + micro]
                curr_idx, next_idx = zip(*chunk)
                batch = _load_batch(self.adata, curr_idx, next_idx, self.config)
                ot_tgt_z = self._lookup_ot_tgt(t_curr, t_next, curr_idx)
                out = _model_forward_from_batch(self.model, batch, ot_tgt_z=ot_tgt_z)
                loss, parts, norm_parts = self._compute_loss(
                    out,
                    batch["x_next"],
                    batch["t_next"],
                    batch["cell_type"],
                    update_ema=False,
                    density_curr=batch.get("density_curr"),
                    ot_tgt_z=ot_tgt_z,
                )
                losses.append(loss.item())
                mses.append(parts["recon"].item())
                val_parts.append(parts)
                if norm_parts is not None:
                    val_norm_parts.append(norm_parts)
                pccs.append(_batch_pcc(out["expr_pred"], batch["x_next"]))
        self.model.train()
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "mse": float(np.mean(mses)) if mses else float("nan"),
            "pcc": float(np.mean(pccs)) if pccs else float("nan"),
            "parts": _mean_loss_parts(val_parts),
            "norm_parts": _mean_loss_parts(val_norm_parts) if val_norm_parts else {},
        }

    def _compute_pseudotime_loss(self, pseudotime, true_time, cell_type, z_emb):
        loss = F.mse_loss(pseudotime, true_time)
        for ct in cell_type.unique():
            mask = cell_type == ct
            if mask.sum() > 1:
                loss = loss + 0.1 * F.mse_loss(
                    pseudotime[mask], pseudotime[mask].mean().detach()
                )
        if z_emb is not None and z_emb.shape[0] > 2:
            with torch.no_grad():
                dist = torch.cdist(z_emb.detach(), z_emb.detach())
                dist_n = dist / (dist.max() + 1e-8)
            pseudo = torch.cdist(pseudotime.unsqueeze(1), pseudotime.unsqueeze(1))
            pseudo_n = pseudo / (pseudo.max() + 1e-8)
            loss = loss + 0.5 * F.mse_loss(pseudo_n, dist_n)
        return loss

    def _safe_ot_loss(self, pred, target):
        """Sinkhorn OT with guards against NaN/Inf, tiny batches, and huge diameters."""
        if pred.shape[0] < 2:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        max_d = float(getattr(self.config, "ot_max_pairwise_dist", 500.0) or 500.0)
        with torch.no_grad():
            d_pred = torch.cdist(pred, pred).max()
            d_tgt = torch.cdist(target, target).max()
            if (
                not torch.isfinite(d_pred)
                or not torch.isfinite(d_tgt)
                or float(d_pred) > max_d
                or float(d_tgt) > max_d
            ):
                return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return self.ot_loss(pred, target)

    def _safe_latent_ot_loss(self, z_pred, z_tgt):
        return self._safe_ot_loss(z_pred, z_tgt)

    def _density_loss(self, out, cell_type, density_curr):
        device = out["expr_pred"].device
        dtype = out["expr_pred"].dtype
        density_loss = torch.zeros((), device=device, dtype=dtype)
        if float(getattr(self.config, "lambda_density", 0.0) or 0.0) <= 0:
            return density_loss
        z_for_density = out.get("z_curr")
        if z_for_density is None:
            return density_loss
        z_det = z_for_density.detach()
        if bool(getattr(self.config, "density_align_stationary", True)):
            pot_for_density = self.model.stationary_potential(z_det).squeeze(-1)
        else:
            t_curr = out.get("t_curr")
            pot_for_density = (
                self.model.potential(z_det, t_curr).squeeze(-1)
                if t_curr is not None
                else out.get("potential_stationary", out.get("potential"))
            )
        if pot_for_density is None:
            return density_loss
        if bool(getattr(self.config, "density_use_latent_batch", False)):
            from density_regularization import density_regularization_loss_latent_batch

            return density_regularization_loss_latent_batch(
                pot_for_density,
                z_det,
                cell_type,
                within_type=bool(getattr(self.config, "density_within_cell_type", True)),
                k=int(getattr(self.config, "density_knn_k", 10) or 10),
            )
        if density_curr is not None:
            from density_regularization import density_regularization_loss

            return density_regularization_loss(pot_for_density, density_curr)
        return density_loss

    def _ot_barycentric_target(self, z_curr, z_next, cell_type):
        target = torch.empty_like(z_curr)
        if cell_type is not None and torch.is_tensor(cell_type) and cell_type.numel() == z_curr.shape[0]:
            groups = [cell_type == c for c in torch.unique(cell_type)]
        else:
            groups = [torch.ones(z_curr.shape[0], dtype=torch.bool, device=z_curr.device)]
        blur = float(getattr(self.config, "latent_disp_ot_blur", 0.05) or 0.05)
        with torch.no_grad():
            for mask in groups:
                if int(mask.sum()) < 2:
                    target[mask] = z_next[mask]
                    continue
                plan = _sinkhorn_plan(z_curr[mask].detach(), z_next[mask].detach(), blur=blur)
                row = plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
                target[mask] = (plan @ z_next[mask].detach()) / row
        return target.detach()

    def _hamiltonian_consistency_loss(self, out, cell_type):
        z_curr = out.get("z_curr")
        z_next = out.get("z_next_emb")
        p_curr = out.get("momentum_init")
        t_curr = out.get("t_curr")
        t_next = out.get("t_next")
        if any(x is None for x in (z_curr, z_next, p_curr, t_curr, t_next)):
            ref = out["expr_pred"]
            return torch.zeros((), device=ref.device, dtype=ref.dtype)
        target = self._ot_barycentric_target(z_curr, z_next, cell_type)
        dt = (t_next - t_curr).clamp_min(1e-6).unsqueeze(-1)
        position_residual = p_curr - (target - z_curr) / dt
        p_next_observed = self.model.initial_momentum(target, t_next)
        z_force = z_curr if z_curr.requires_grad else z_curr.detach().requires_grad_(True)
        u0 = self.model.stationary_potential(z_force)
        grad_u0 = torch.autograd.grad(
            u0, z_force, torch.ones_like(u0), create_graph=True, retain_graph=True
        )[0]
        gamma = float(getattr(self.config, "hamiltonian_damping_gamma", 0.1) or 0.0)
        force_residual = (p_next_observed - p_curr) / dt + grad_u0 + gamma * p_curr
        return 0.5 * (position_residual.pow(2).mean() + force_residual.pow(2).mean())

    def _compute_hamiltonian_loss(self, out, x_next, t_next, cell_type, density_curr=None):
        expr_pred = out["expr_pred"]
        if not torch.isfinite(expr_pred).all():
            expr_pred = torch.nan_to_num(expr_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        z_pred = out.get("z_pred")
        z_next = out.get("z_next_emb")
        ot = (
            self._safe_latent_ot_loss(z_pred, z_next.detach())
            if z_pred is not None and z_next is not None
            else torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype)
        )
        ae_terms = []
        if out.get("expr_curr") is not None and out.get("x_curr_input") is not None:
            ae_terms.append(F.mse_loss(out["expr_curr"], out["x_curr_input"]))
        if z_next is not None:
            ae_terms.append(F.mse_loss(self.model.predict_expression(z_next, t_next), x_next))
        ae = (
            torch.stack(ae_terms).mean()
            if ae_terms
            else torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype)
        )
        density = self._density_loss(out, cell_type, density_curr)
        hamiltonian = self._hamiltonian_consistency_loss(out, cell_type)
        lambda_ae = float(getattr(self.config, "lambda_ae", 0.1) or 0.0)
        lambda_density = float(getattr(self.config, "lambda_density", 0.0) or 0.0)
        lambda_h = float(getattr(self.config, "lambda_hamiltonian", 0.0) or 0.0)
        total = ot + lambda_ae * ae + lambda_density * density + lambda_h * hamiltonian
        parts = {k: torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype) for k in LOSS_PART_KEYS}
        parts.update(
            {
                "ot": ot,
                "recon": ae,
                "density": density,
                "hamiltonian": hamiltonian,
                "pair_mse_loss": F.mse_loss(expr_pred, x_next).detach(),
                "reconstruction_mode": "ae_current_next",
            }
        )
        return total, parts, None

    def _compute_loss(self, out, x_next, t_next, cell_type, update_ema: bool = True, density_curr=None, ot_tgt_z=None):
        if getattr(self.config, "loss_recipe", "legacy") == "hamiltonian":
            return self._compute_hamiltonian_loss(
                out, x_next, t_next, cell_type, density_curr=density_curr
            )
        expr_pred = out["expr_pred"]
        if not torch.isfinite(expr_pred).all():
            expr_pred = torch.nan_to_num(expr_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        ot = self._safe_ot_loss(expr_pred, x_next)
        pair_mse_loss = F.mse_loss(expr_pred, x_next)
        mmd_loss = gaussian_mmd_loss(expr_pred, x_next)
        recon = build_reconstruction_loss(self.config, pair_mse_loss, mmd_loss)

        energy = out.get("energy", out.get("hjb_residual")).mean()
        momentum = out.get("momentum")
        if momentum is None:
            momentum = torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype)
        elif torch.is_tensor(momentum) and momentum.dim() > 0:
            momentum = momentum.mean()

        density_loss = torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype)
        lambda_density = float(getattr(self.config, "lambda_density", 0.0) or 0.0)
        if lambda_density > 0:
            z_for_density = out.get("z_curr")
            if z_for_density is not None:
                z_det = z_for_density.detach()
                if bool(getattr(self.config, "density_align_stationary", True)):
                    pot_for_density = self.model.stationary_potential(z_det).squeeze(-1)
                else:
                    t_curr = out.get("t_curr")
                    if t_curr is None:
                        pot_for_density = out.get("potential_stationary", out.get("potential"))
                    else:
                        pot_for_density = self.model.potential(z_det, t_curr).squeeze(-1)
                if pot_for_density is not None:
                    if bool(getattr(self.config, "density_use_latent_batch", False)):
                        from density_regularization import (
                            density_regularization_loss_latent_batch,
                        )

                        density_loss = density_regularization_loss_latent_batch(
                            pot_for_density,
                            z_det,
                            cell_type,
                            within_type=bool(
                                getattr(self.config, "density_within_cell_type", True)
                            ),
                            k=int(getattr(self.config, "density_knn_k", 10) or 10),
                        )
                    elif density_curr is not None:
                        from density_regularization import density_regularization_loss

                        density_loss = density_regularization_loss(
                            pot_for_density, density_curr
                        )

        residual_balance_loss = torch.zeros(
            (), device=expr_pred.device, dtype=expr_pred.dtype
        )
        lambda_residual_balance = float(
            getattr(self.config, "lambda_residual_balance", 0.0) or 0.0
        )
        if lambda_residual_balance > 0 and out.get("residual_drift") is not None:
            target_ratio = float(
                getattr(self.config, "residual_ratio_target", 0.55) or 0.55
            )
            r = out["residual_drift"]
            zg = out["z_curr"].detach().requires_grad_(True)
            u0 = self.model.stationary_potential(zg).sum()
            gu = torch.autograd.grad(u0, zg, create_graph=True)[0]
            ratio = r.norm(dim=1).mean() / (gu.norm(dim=1).mean() + 1e-8)
            excess = F.relu(ratio - target_ratio)
            residual_balance_loss = excess.pow(2)

        expr_curr = out["expr_curr"]
        x_curr = out.get("x_curr_input")
        if x_curr is None:
            x_curr = out.get("x_curr")
        pred_delta = expr_pred - expr_curr
        true_delta = x_next - x_curr
        delta_mse = F.mse_loss(pred_delta, true_delta)
        direction_loss = (1.0 - F.cosine_similarity(pred_delta, true_delta, dim=1)).mean()

        # Latent-consistency: the pushed-forward latent population should match the encoded
        # next-timepoint population (target detached so the FLOW is corrected, not the encoder
        # collapsed). Directly penalizes the latent overshoot/divergence.
        lambda_latent = float(getattr(self.config, "lambda_latent", 0.0) or 0.0)
        lambda_kinetic = float(getattr(self.config, "lambda_kinetic", 0.0) or 0.0)
        z_curr = out.get("z_curr")
        z_pred = out.get("z_pred")
        z_next_emb = out.get("z_next_emb")
        if lambda_latent > 0 and z_pred is not None and z_next_emb is not None:
            latent_loss = self._safe_latent_ot_loss(z_pred, z_next_emb.detach())
        else:
            latent_loss = torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype)
        # Kinetic penalty on momentum magnitude damps the second-order overshoot.
        # kinetic_terminal_beta (>1) up-weights the FINAL momentum so the flow is
        # penalised for still carrying velocity at the target (i.e. coasting past
        # it), which is the direct lever on the latent displacement overshoot.
        p_init_out = out.get("momentum_init")
        p_final_out = out.get("momentum_state")
        beta = float(getattr(self.config, "kinetic_terminal_beta", 1.0) or 1.0)
        if lambda_kinetic > 0 and p_init_out is not None:
            kin_init = p_init_out.pow(2).mean()
            if p_final_out is not None:
                kin_final = p_final_out.pow(2).mean()
                kinetic = (kin_init + beta * kin_final) / (1.0 + beta)
            else:
                kinetic = kin_init
        else:
            kinetic = torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype)

        # Latent displacement supervision (per cell type). Two variants:
        #   * centroid (default): match mean(z_pred)-mean(z_curr) to the observed
        #     mean(z_next)-mean(z_curr). Only constrains the first moment, so it can
        #     leave the per-cell magnitude uncontrolled (the residual overshoot).
        #   * OT-coupled (latent_disp_ot_coupling=True): within each cell type solve
        #     an entropic OT between z_curr and z_next, take the barycentric image
        #     zhat_next(i)=sum_j P_ij z_next(j) as a WELL-POSED per-cell target, and
        #     supervise z_pred(i) -> sg(zhat_next(i)). This constrains direction AND
        #     magnitude per cell, directly targeting the displacement-norm overshoot,
        #     and is robust to the random within-type pairing because the target is
        #     rebuilt from the OT plan rather than the arbitrary pairing.
        lambda_lat_disp = float(getattr(self.config, "lambda_lat_disp", 0.0) or 0.0)
        lat_disp = torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype)
        # Use the potential-gated prediction (if produced) so these gradients shape
        # the momentum / residual fields but not the U0 landscape; falls back to the
        # standard z_pred when gating is off.
        z_disp = out.get("z_pred_disp")
        if z_disp is None:
            z_disp = z_pred
        use_ot_coupling = bool(getattr(self.config, "latent_disp_ot_coupling", False))
        use_mag_ratio = bool(getattr(self.config, "latent_disp_use_mag_ratio", False))
        if (
            lambda_lat_disp > 0
            and z_curr is not None
            and z_disp is not None
            and z_next_emb is not None
        ):
            if cell_type is not None and torch.is_tensor(cell_type) and cell_type.numel() == z_curr.shape[0]:
                groups = [(cell_type == c) for c in torch.unique(cell_type)]
            else:
                groups = [torch.ones(z_curr.shape[0], dtype=torch.bool, device=z_curr.device)]
            disp_terms = []
            blur = float(getattr(self.config, "latent_disp_ot_blur", 0.05) or 0.05)
            if use_ot_coupling:
                if ot_tgt_z is not None and ot_tgt_z.shape == z_curr.shape:
                    zc_all = z_curr
                    zp_all = z_disp
                    tgt_all = ot_tgt_z.detach()
                    pred_step = zp_all - zc_all
                    true_step = (tgt_all - zc_all).detach()
                    if use_mag_ratio:
                        pred_norm = pred_step.norm(dim=1)
                        true_norm = true_step.norm(dim=1).clamp_min(1e-8)
                        excess = F.relu(pred_norm / true_norm - 1.0)
                        disp_terms.append(excess.pow(2).mean())
                    else:
                        disp_terms.append((zp_all - tgt_all).pow(2).mean())
                else:
                    for m in groups:
                        if int(m.sum()) < 2:
                            continue
                        zc = z_curr[m]
                        zp = z_disp[m]
                        zn = z_next_emb[m].detach()
                        with torch.no_grad():
                            plan = _sinkhorn_plan(zc.detach(), zn, blur=blur)
                            row = plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
                            tgt = (plan @ zn) / row
                        pred_step = zp - zc
                        true_step = (tgt - zc).detach()
                        if use_mag_ratio:
                            pred_norm = pred_step.norm(dim=1)
                            true_norm = true_step.norm(dim=1).clamp_min(1e-8)
                            excess = F.relu(pred_norm / true_norm - 1.0)
                            disp_terms.append(excess.pow(2).mean())
                        else:
                            disp_terms.append((zp - tgt).pow(2).mean())
            else:
                for m in groups:
                    if int(m.sum()) < 2:
                        continue
                    mu_curr = z_curr[m].mean(dim=0)
                    mu_pred = z_disp[m].mean(dim=0)
                    mu_next = z_next_emb[m].mean(dim=0).detach()
                    pred_disp = mu_pred - mu_curr
                    true_disp = mu_next - mu_curr.detach()
                    disp_terms.append((pred_disp - true_disp).pow(2).mean())
            if disp_terms:
                lat_disp = torch.stack(disp_terms).mean()

        parts = {
            "ot": ot,
            "recon": recon,
            "energy": energy,
            "momentum": momentum,
            "density": density_loss,
            "delta": delta_mse,
            "direction": direction_loss,
            "latent": latent_loss,
            "kinetic": kinetic,
            "lat_disp": lat_disp,
            "residual_balance": residual_balance_loss,
            "hjb": energy,
            "pseudo": torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype),
            "residual": torch.zeros((), device=expr_pred.device, dtype=expr_pred.dtype),
            "pair_mse_loss": pair_mse_loss.detach(),
            "mmd_loss": mmd_loss.detach(),
        }
        norm_parts = None
        lambda_energy = float(getattr(self.config, "lambda_energy", self.config.lambda_hjb) or 0.0)
        lambda_kinetic = float(getattr(self.config, "lambda_kinetic", 0.0) or 0.0)
        lambda_delta = float(getattr(self.config, "lambda_delta", 0.0) or 0.0)
        lambda_direction = float(getattr(self.config, "lambda_direction", 0.0) or 0.0)
        lambda_residual_balance = float(
            getattr(self.config, "lambda_residual_balance", 0.0) or 0.0
        )
        legacy_momentum = getattr(self.config, "_legacy_lambda_momentum", None)
        norm_keys = list(LOSS_PART_KEYS)
        if getattr(self.config, "latent_disp_exclude_ema", False):
            norm_keys = [k for k in norm_keys if k != "lat_disp"]
        if getattr(self.config, "use_loss_normalization", False):
            if update_ema:
                self.loss_norm_ema.update({k: parts[k] for k in norm_keys})
            scaled = self.loss_norm_ema.normalize({k: parts[k] for k in norm_keys})
            norm_parts = {k: scaled[k].detach() for k in norm_keys}
            total = scaled["ot"] + scaled["recon"]
            if lambda_energy > 0:
                total = total + lambda_energy * scaled["energy"]
            if lambda_density > 0:
                total = total + lambda_density * scaled["density"]
            if lambda_delta > 0:
                total = total + lambda_delta * scaled["delta"]
            if lambda_direction > 0:
                total = total + lambda_direction * scaled["direction"]
            if lambda_latent > 0:
                total = total + lambda_latent * scaled["latent"]
            if lambda_kinetic > 0:
                if legacy_momentum is not None:
                    total = total + float(legacy_momentum) * scaled["momentum"]
                    total = total + lambda_kinetic * scaled["kinetic"]
                else:
                    total = total + lambda_kinetic * (
                        INERTIA_MOMENTUM_MIX * scaled["momentum"] + scaled["kinetic"]
                    )
            if lambda_lat_disp > 0:
                lat_disp_term = (
                    lat_disp
                    if getattr(self.config, "latent_disp_exclude_ema", False)
                    else scaled["lat_disp"]
                )
                total = total + lambda_lat_disp * lat_disp_term
            if lambda_residual_balance > 0:
                total = total + lambda_residual_balance * scaled["residual_balance"]
        else:
            total = ot + recon
            if lambda_energy > 0:
                total = total + lambda_energy * energy
            if lambda_density > 0:
                total = total + lambda_density * density_loss
            if lambda_delta > 0:
                total = total + lambda_delta * delta_mse
            if lambda_direction > 0:
                total = total + lambda_direction * direction_loss
            if lambda_latent > 0:
                total = total + lambda_latent * latent_loss
            if lambda_kinetic > 0:
                if legacy_momentum is not None:
                    total = total + float(legacy_momentum) * momentum
                    total = total + lambda_kinetic * kinetic
                else:
                    total = total + lambda_kinetic * (
                        INERTIA_MOMENTUM_MIX * momentum + kinetic
                    )
            if lambda_lat_disp > 0:
                total = total + lambda_lat_disp * lat_disp
            if lambda_residual_balance > 0:
                total = total + lambda_residual_balance * residual_balance_loss

        parts["reconstruction_mode"] = getattr(self.config, "reconstruction_mode", "mse_mmd")
        return total, parts, norm_parts

    def evaluate_model(self):
        print("\n=== 开始最终评估 ===")
        self.adata = predict(
            model=self.model, adata=self.adata, config=self.config, save_dir=self.save_dir
        )
        pearson_r, spearman_r = validate_potential_logp_consistency(
            self.model, self.adata, self.config
        )
        self.metrics["potential_logp_pearson"] = pearson_r
        self.metrics["potential_logp_spearman"] = spearman_r
        self.metrics["ph"] = self._calculate_potential_barrier()
        self.metrics["bdr"] = self._calculate_bifurcation_rate()

        print("\n===== 评估报告 =====")
        print(f"• 参数量: {self.metrics['params'] / 1e6:.2f}M")
        print(f"• 总训练时间: {self.metrics['train_time'] / 60:.2f} 分钟")
        ckpt_metric = getattr(self.config, "checkpoint_metric", "pcc")
        stop_metric = getattr(self.config, "early_stop_metric", "pcc")
        print(
            f"• 最佳 checkpoint: epoch {self.metrics.get('best_epoch', 0)} "
            f"(checkpoint_metric={ckpt_metric}, early_stop={stop_metric})"
        )
        print(f"• 最佳训练 MSE: {min(self.metrics['train_mse']):.4f}")
        print(f"• 最佳验证 MSE: {min(self.metrics['val_mse']):.4f}")
        print(f"• 最佳验证 PCC: {max(self.metrics['val_pcc']):.4f}")
        print(f"• 选模 ValPCC: {self.metrics.get('best_val_pcc', float('nan')):.4f}")
        print(f"• 选模 ValMSE: {self.metrics.get('best_val_mse', float('nan')):.4f}")
        if ckpt_metric == "loss":
            print(f"• 选模 ValLoss: {self.metrics.get('best_val_loss', float('nan')):.4f}")
        print(f"• 平均 L_energy: {np.nanmean(self.metrics.get('energy', self.metrics['hjb'])):.4e}")
        print(f"• U vs -log KDE(z) Pearson: {pearson_r:.4f}")
        print(f"• U vs -log KDE(z) Spearman: {spearman_r:.4f}")
        if abs(pearson_r) < 0.3:
            warnings.warn(
                "Potential 与 latent KDE 的 -log p 相关性较低；"
                "U(z,t) 可能只是网络内部正则项，不宜直接解释为物理势能景观。",
                UserWarning,
                stacklevel=2,
            )
        print(f"• 势能屏障高度: {self.metrics['ph']:.4f}")
        print(f"• 分岔检测率: {self.metrics['bdr']:.4f}")

    def _calculate_potential_barrier(self):
        idx = np.random.choice(len(self.adata), min(1000, len(self.adata)), replace=False)
        potentials = self.adata.obs["potential"].values[idx]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(potentials.reshape(-1, 1))
        centers = sorted(kmeans.cluster_centers_.flatten())
        return centers[-1] - centers[0]

    def _calculate_bifurcation_rate(self):
        pseudotime = self.adata.obs["pseudotime"].values
        gene_indices = np.random.choice(
            self.adata.n_vars, min(100, self.adata.n_vars), replace=False
        )
        expr = self.adata.X[:, gene_indices]
        expr = expr.toarray() if sp.issparse(expr) else expr
        peak_counts = np.zeros(len(pseudotime))
        for i in range(expr.shape[1]):
            peaks, _ = find_peaks(expr[:, i], prominence=np.std(expr[:, i]))
            peak_counts[peaks] += 1
        hist, _ = np.histogram(pseudotime, bins=10, weights=peak_counts)
        threshold = np.percentile(hist, 75)
        return np.sum(hist > threshold) / 10


def _batch_pcc(pred, true):
    p = pred.detach().cpu().numpy().ravel()
    t = true.detach().cpu().numpy().ravel()
    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def _load_batch(adata, current_idx, next_idx, cfg):
    idx_c, idx_n = np.array(current_idx), np.array(next_idx)
    if sp.issparse(adata.X):
        x_curr = torch.tensor(adata.X[idx_c].toarray()).float()
        x_next = torch.tensor(adata.X[idx_n].toarray()).float()
    else:
        x_curr = torch.tensor(adata.X[idx_c]).float()
        x_next = torch.tensor(adata.X[idx_n]).float()
    t_curr = torch.tensor(adata.obs[cfg.time_key].values[idx_c]).float()
    t_next = torch.tensor(adata.obs[cfg.time_key].values[idx_n]).float()
    ct = torch.tensor(adata.obs[cfg.cell_type_key].values[idx_c]).long()
    device = cfg.device
    batch = {
        "x_curr": x_curr.to(device),
        "x_next": x_next.to(device),
        "t_curr": t_curr.to(device),
        "t_next": t_next.to(device),
        "cell_type": ct.to(device),
    }
    stage_key = getattr(cfg, "stage_cond_key", None)
    if stage_key and stage_key in adata.obs.columns:
        batch["stage"] = torch.tensor(adata.obs[stage_key].values[idx_c]).long().to(device)
    density_key = getattr(cfg, "density_target_key", "density_neglogp")
    if density_key in adata.obs.columns and float(getattr(cfg, "lambda_density", 0.0) or 0.0) > 0:
        density_curr = torch.tensor(
            pd.to_numeric(adata.obs[density_key].values[idx_c], errors="coerce").astype(float)
        ).float()
        batch["density_curr"] = density_curr.to(device)
    return batch


def _model_forward_from_batch(model, batch, *, z_curr_override=None, ot_tgt_z=None):
    """Call TemporalSDENetwork.forward with optional stage conditioner from batch."""
    return model(
        batch["x_curr"],
        batch["x_next"],
        batch["cell_type"],
        batch["t_curr"],
        batch["t_next"],
        z_curr_override=z_curr_override,
        ot_tgt_z=ot_tgt_z,
        stage=batch.get("stage"),
    )


def sample_transition_pairs(
    adata,
    t_curr,
    t_next,
    batch_size,
    cfg,
    cell_mask=None,
    curr_cell_mask=None,
    next_cell_mask=None,
):
    """Sample matched cell pairs for a specific time transition.

    Default: random same-cell-type pairing across the cohort.
    With ``cfg.pair_group_keys`` (e.g. ``["patient_id"]``): only pair within each
    group × cell type (HGSOC patient-paired NACT).
    """
    pairs = []
    obs = adata.obs
    tvals = obs[cfg.time_key].astype(float).values
    ct_vals = obs[cfg.cell_type_key].values
    group_keys = list(getattr(cfg, "pair_group_keys", None) or [])

    def _apply_masks(base_mask, *, is_curr: bool):
        mask = base_mask
        if cell_mask is not None:
            mask = mask & cell_mask
        if is_curr and curr_cell_mask is not None:
            mask = mask & curr_cell_mask
        if (not is_curr) and next_cell_mask is not None:
            mask = mask & next_cell_mask
        return mask

    if group_keys:
        missing = [k for k in group_keys if k not in obs.columns]
        if missing:
            raise KeyError(f"pair_group_keys missing from obs: {missing}")
        group_frame = obs[group_keys].astype(str)
        combo_rows = []
        for i in np.where(
            np.isclose(tvals, float(t_curr)) | np.isclose(tvals, float(t_next))
        )[0]:
            combo_rows.append(tuple(group_frame.iloc[i].tolist()) + (ct_vals[i],))
        for combo in sorted(set(combo_rows)):
            *group_vals, ct = combo
            g_mask = np.ones(len(obs), dtype=bool)
            for key, val in zip(group_keys, group_vals):
                g_mask &= obs[key].astype(str).values == val
            ct_mask = ct_vals == ct
            curr_mask = _apply_masks(
                g_mask & ct_mask & np.isclose(tvals, float(t_curr)), is_curr=True
            )
            next_mask = _apply_masks(
                g_mask & ct_mask & np.isclose(tvals, float(t_next)), is_curr=False
            )
            curr_idx = np.where(curr_mask)[0]
            next_pool = np.where(next_mask)[0]
            if len(curr_idx) > 0 and len(next_pool) > 0:
                next_idx = np.random.choice(next_pool, len(curr_idx))
                pairs.extend(zip(curr_idx, next_idx))
    else:
        for ct in np.unique(ct_vals):
            ct_mask = ct_vals == ct
            curr_mask = _apply_masks(
                ct_mask & np.isclose(tvals, float(t_curr)), is_curr=True
            )
            next_mask = _apply_masks(
                ct_mask & np.isclose(tvals, float(t_next)), is_curr=False
            )
            curr_idx = np.where(curr_mask)[0]
            if len(curr_idx) > 0 and next_mask.sum() > 0:
                next_idx = np.random.choice(np.where(next_mask)[0], len(curr_idx))
                pairs.extend(zip(curr_idx, next_idx))
    if len(pairs) == 0:
        return []
    pairs = random.sample(pairs, min(batch_size, len(pairs)))
    return pairs


def build_adjacent_rollout_chains(cfg):
    """Build multi-step adjacent chains [t_i->t_{i+1}->...] for BPTT rollout training."""
    times = sorted(set([float(cfg.start_time)] + [float(t) for t in cfg.train_time]))
    horizon = int(getattr(cfg, "rollout_horizon", 3) or 3)
    chains = []
    for start in range(len(times) - 1):
        chain = [
            (times[j], times[j + 1])
            for j in range(start, min(len(times) - 1, start + horizon))
        ]
        if len(chain) >= 2:
            chains.append(chain)
    return chains


def sample_rollout_chains(adata, time_chain, batch_size, cfg, cell_mask=None):
    """Sample ``batch_size`` index chains aligned across all times in ``time_chain``."""
    if not time_chain:
        return []
    t0 = float(time_chain[0][0])
    obs = adata.obs
    tvals = obs[cfg.time_key].astype(float).values
    ct_vals = obs[cfg.cell_type_key].values
    candidates = []
    for ct in np.unique(ct_vals):
        mask_t0 = (ct_vals == ct) & np.isclose(tvals, t0)
        if cell_mask is not None:
            mask_t0 = mask_t0 & cell_mask
        candidates.extend(int(i) for i in np.where(mask_t0)[0])
    if not candidates:
        return []
    random.shuffle(candidates)
    chains = []
    for i0 in candidates:
        if len(chains) >= batch_size:
            break
        ct = ct_vals[i0]
        chain = [i0]
        ok = True
        for _t_curr, t_next in time_chain:
            next_mask = (ct_vals == ct) & np.isclose(tvals, float(t_next))
            if cell_mask is not None:
                next_mask = next_mask & cell_mask
            pool = np.where(next_mask)[0]
            if len(pool) == 0:
                ok = False
                break
            chain.append(int(np.random.choice(pool)))
        if ok and len(chain) == len(time_chain) + 1:
            chains.append(chain)
    return chains


def random_pair(adata, current_time, select_time, batch_size, config=None):
    """Backward-compatible wrapper."""
    cfg = config or globals()["config"]
    pairs = sample_transition_pairs(
        adata, current_time, select_time, batch_size, cfg, cell_mask=None
    )
    return zip(*pairs)


@torch.no_grad()
def _potential_neglogp_arrays_from_latent(
    z_np: np.ndarray,
    u_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute shifted -log KDE(z) aligned with potential samples."""
    z_np = np.asarray(z_np, dtype=float)
    u_np = np.asarray(u_np, dtype=float)
    nbrs = NearestNeighbors(n_neighbors=min(30, len(z_np))).fit(z_np)
    dists, _ = nbrs.kneighbors(z_np)
    bandwidth = float(np.median(dists[:, 1:])) if dists.shape[1] > 1 else 1.0
    kde = KernelDensity(bandwidth=max(bandwidth, 1e-6), kernel="gaussian")
    kde.fit(z_np)
    log_p = kde.score_samples(z_np)
    neg_log_p = -(log_p - np.max(log_p))
    return u_np, neg_log_p


def compute_potential_neglogp_consistency(
    adata,
    *,
    latent_key: str = "X_latent",
    potential_key: str = "potential",
    max_cells: int = 3000,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return Pearson/Spearman and per-cell potential vs -log KDE(z) arrays."""
    if latent_key not in adata.obsm or potential_key not in adata.obs.columns:
        empty = np.array([], dtype=float)
        return 0.0, 0.0, empty, empty

    z_all = np.asarray(adata.obsm[latent_key], dtype=float)
    u_all = np.asarray(adata.obs[potential_key].values, dtype=float)
    if len(z_all) > max_cells:
        idx = np.random.choice(len(z_all), max_cells, replace=False)
        z_all = z_all[idx]
        u_all = u_all[idx]

    u_np, neg_log_p = _potential_neglogp_arrays_from_latent(z_all, u_all)
    if np.std(u_np) < 1e-12 or np.std(neg_log_p) < 1e-12:
        return 0.0, 0.0, u_np, neg_log_p
    pearson = float(np.corrcoef(u_np, neg_log_p)[0, 1])
    spearman = float(spearmanr(u_np, neg_log_p).correlation)
    return pearson, spearman, u_np, neg_log_p


def _stage_batch_tensor(adata, indices, cfg, device):
    """Optional stage conditioner codes for a batch of obs indices."""
    stage_key = getattr(cfg, "stage_cond_key", None)
    if not stage_key or stage_key not in adata.obs.columns:
        return None
    return torch.tensor(
        np.asarray(adata.obs[stage_key].values[indices], dtype=np.int64)
    ).long().to(device)


@torch.no_grad()
def validate_potential_logp_consistency(model, adata, cfg, max_cells=3000):
    """
    Check whether U(z,t) correlates with -log p_hat(z) from latent KDE.
    Low correlation => potential is not a quasi physical landscape.
    """
    model.eval()
    x_all = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    t_all = adata.obs[cfg.time_key].values.astype(float)
    ct_all = adata.obs[cfg.cell_type_key].values.astype(int)
    stage_key = getattr(cfg, "stage_cond_key", None)
    stage_all = (
        adata.obs[stage_key].values.astype(int)
        if stage_key and stage_key in adata.obs.columns
        else None
    )

    if len(x_all) > max_cells:
        idx = np.random.choice(len(x_all), max_cells, replace=False)
        x_all, t_all, ct_all = x_all[idx], t_all[idx], ct_all[idx]
        if stage_all is not None:
            stage_all = stage_all[idx]

    z_list, u_list = [], []
    batch_size = cfg.batch_size
    for start in range(0, len(x_all), batch_size):
        end = min(start + batch_size, len(x_all))
        x = torch.tensor(x_all[start:end]).float().to(cfg.device)
        t = torch.tensor(t_all[start:end]).float().to(cfg.device)
        ct = torch.tensor(ct_all[start:end]).long().to(cfg.device)
        stage = None
        if stage_all is not None:
            stage = torch.tensor(stage_all[start:end]).long().to(cfg.device)
        z = model.encode(x, ct, stage=stage)
        u = model.potential(z, t).cpu().numpy().squeeze(-1)
        z_list.append(z.cpu().numpy())
        u_list.append(u)

    z_np = np.vstack(z_list)
    u_np = np.concatenate(u_list)
    u_np, neg_log_p = _potential_neglogp_arrays_from_latent(z_np, u_np)

    if np.std(u_np) < 1e-12 or np.std(neg_log_p) < 1e-12:
        return 0.0, 0.0
    pearson = float(np.corrcoef(u_np, neg_log_p)[0, 1])
    spearman = float(spearmanr(u_np, neg_log_p).correlation)
    return pearson, spearman


def _load_state_dict_compat(model, state_dict):
    """Load weights tolerating architecture-version drift (e.g. added momentum_net).

    Tries a strict load first; on a key mismatch it retries with strict=False and warns,
    so legacy checkpoints trained before a given submodule was added remain analyzable
    (the missing submodule then falls back to its default / EMA behaviour).
    """
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        warnings.warn(
            "Checkpoint architecture differs from the current model; loaded with "
            f"strict=False. missing_keys={missing[:6]}{'...' if len(missing) > 6 else ''}, "
            f"unexpected_keys={unexpected[:6]}{'...' if len(unexpected) > 6 else ''}. "
            f"(original error: {exc})",
            UserWarning,
            stacklevel=2,
        )


def predict(model, adata, config, save_dir):
    """Per-cell inference; velocity layer = expression delta, NOT RNA velocity."""
    ensure_cell_type_codes(adata, config.cell_type_key)
    _load_state_dict_compat(
        model, torch.load(f"{save_dir}/best_model.pth", map_location=config.device)
    )
    model = model.to(config.device)
    model.eval()

    n = adata.n_obs
    expr_pred = np.zeros((n, adata.n_vars), dtype=float)
    potentials = np.zeros(n, dtype=float)
    potentials_stationary = np.zeros(n, dtype=float)
    pseudotimes = np.zeros(n, dtype=float)
    hjb_vals = np.zeros(n, dtype=float)
    diffusion_eff = np.zeros(n, dtype=float)

    x_all = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    t_all = adata.obs[config.time_key].values.astype(float)
    ct_all = adata.obs[config.cell_type_key].values.astype(int)
    stage_key = getattr(config, "stage_cond_key", None)
    stage_all = (
        adata.obs[stage_key].values.astype(int)
        if stage_key and stage_key in adata.obs.columns
        else None
    )

    batch_size = config.batch_size
    z_all = np.zeros((n, model.latent_dim), dtype=float)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.tensor(x_all[start:end]).float().to(config.device)
            t = torch.tensor(t_all[start:end]).float().to(config.device)
            ct = torch.tensor(ct_all[start:end]).long().to(config.device)
            stage = None
            if stage_all is not None:
                stage = torch.tensor(stage_all[start:end]).long().to(config.device)
            z = model.encode(x, ct, stage=stage)
            z_all[start:end] = z.cpu().numpy()
            expr_pred[start:end] = model.predict_expression(z, t).cpu().numpy()
            potentials[start:end] = model.potential(z, t).cpu().numpy().squeeze(-1)
            potentials_stationary[start:end] = (
                model.stationary_potential(z).cpu().numpy().squeeze(-1)
            )
            pseudotimes[start:end] = (
                model.pseudotime_head(torch.cat([z, t.unsqueeze(1)], dim=1))
                .cpu()
                .numpy()
                .squeeze(-1)
            )
            sigma_rows = []
            for i in range(z.shape[0]):
                sigma_rows.append(model.sde_func.g(float(t[i].item()), z[i : i + 1]))
            sigma = torch.cat(sigma_rows, dim=0)
            diffusion_eff[start:end] = 0.5 * (sigma ** 2).mean(dim=1).cpu().numpy()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.tensor(x_all[start:end]).float().to(config.device)
        t = torch.tensor(t_all[start:end]).float().to(config.device)
        ct = torch.tensor(ct_all[start:end]).long().to(config.device)
        stage = None
        if stage_all is not None:
            stage = torch.tensor(stage_all[start:end]).long().to(config.device)
        z = model.encode(x, ct, stage=stage)
        with torch.enable_grad():
            hjb_vals[start:end] = model.hjb_residual(z, t).detach().cpu().numpy().squeeze(-1)

    delta = expr_pred - x_all
    adata.layers["predicted_expression"] = expr_pred
    adata.layers["predicted_delta"] = delta
    if config.keep_velocity_alias:
        adata.layers["velocity"] = delta
        adata.uns["velocity_description"] = VELOCITY_LAYER_TITLE

    residual_ratio_vals = np.full(n, np.nan, dtype=float)
    grad_norm_vals = np.full(n, np.nan, dtype=float)
    residual_norm_vals = np.full(n, np.nan, dtype=float)
    total_norm_vals = np.full(n, np.nan, dtype=float)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.tensor(x_all[start:end]).float().to(config.device)
        t = torch.tensor(t_all[start:end]).float().to(config.device)
        ct = torch.tensor(ct_all[start:end]).long().to(config.device)
        stage = None
        if stage_all is not None:
            stage = torch.tensor(stage_all[start:end]).long().to(config.device)
        z = model.encode(x, ct, stage=stage)
        decomp = model.drift_decomposition(z, t)
        residual_ratio_vals[start:end] = decomp["residual_ratio"].detach().cpu().numpy().reshape(-1)
        grad_norm_vals[start:end] = decomp["grad_norm"].detach().cpu().numpy().reshape(-1)
        residual_norm_vals[start:end] = decomp["residual_norm"].detach().cpu().numpy().reshape(-1)
        total_norm_vals[start:end] = decomp["total_norm"].detach().cpu().numpy().reshape(-1)

    adata.obs["potential"] = potentials
    adata.obs["potential_stationary"] = potentials_stationary
    adata.obs["pseudotime"] = pseudotimes
    adata.obs["hjb_residual"] = hjb_vals
    adata.obs["hj_regularizer"] = hjb_vals
    adata.obs["diffusion_eff"] = diffusion_eff
    adata.obs["residual_ratio"] = residual_ratio_vals
    adata.obs["grad_norm"] = grad_norm_vals
    adata.obs["residual_norm"] = residual_norm_vals
    adata.obs["total_norm"] = total_norm_vals

    if getattr(config, "compute_plasticity_scores", True):
        from potential_interpretation import attach_interpretation_scores

        attach_interpretation_scores(
            adata,
            time_key=config.time_key,
            cell_type_key=config.cell_type_key,
            homeostasis_ref_time=getattr(config, "homeostasis_ref_time", None),
        )

    from latent_embeddings import save_latent_embeddings_checkpoint
    from sklearn.decomposition import PCA

    adata.obsm["X_latent"] = z_all
    n_components = min(20, z_all.shape[1], z_all.shape[0] - 1)
    if n_components >= 1:
        pca = PCA(n_components=n_components, random_state=0)
        adata.obsm["X_latent_pca"] = pca.fit_transform(z_all)
        adata.uns["X_latent_pca_pca_components"] = pca.components_
        adata.uns["X_latent_pca_pca_mean"] = pca.mean_
        adata.uns["X_latent_pca_explained_variance_ratio"] = pca.explained_variance_ratio_
    save_latent_embeddings_checkpoint(adata, save_dir)
    return adata


def _get_prediction_layer(adata):
    if "predicted_expression" in adata.layers:
        return adata.layers["predicted_expression"]
    raise KeyError("未找到 predicted_expression layer")


def _get_delta_layer(adata):
    if "predicted_delta" in adata.layers:
        return adata.layers["predicted_delta"]
    if "velocity" in adata.layers:
        return adata.layers["velocity"]
    raise KeyError("未找到 predicted_delta / velocity layer")


def plot_metrics(trainer, adata, save_dir):
    import seaborn as sns
    from matplotlib.lines import Line2D

    plot_adata = getattr(trainer, "adata", adata)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("white")
    axs[0, 0].plot(trainer.metrics["train_loss"], label="train", color="#778ccc", lw=2)
    axs[0, 0].plot(trainer.metrics["val_loss"], label="val", color="#f17172", lw=2)
    axs[0, 0].set_title("Loss (train vs held-out validation)", fontweight="bold")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend(frameon=False)
    axs[0, 0].grid(True, linestyle="--", alpha=0.35)

    ax1 = axs[0, 1]
    ax1.plot(trainer.metrics["train_pcc"], label="train PCC", color="#778ccc", lw=2)
    ax1.plot(trainer.metrics["val_pcc"], label="val PCC", color="#f17172", lw=2, linestyle="--")
    ax1.set_ylabel("PCC")
    ax2 = ax1.twinx()
    ax2.plot(trainer.metrics["train_mse"], label="train MSE", color="#c2d7f3", lw=2)
    ax2.plot(trainer.metrics["val_mse"], label="val MSE", color="#ffbdb9", lw=2, linestyle="--")
    ax2.set_ylabel("MSE")
    ax1.set_title("Held-out validation metrics", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, linestyle="--", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False, fontsize=9)

    has_potential = "potential" in plot_adata.obs.columns
    if has_potential:
        sns.kdeplot(
            plot_adata.obs["potential"],
            ax=axs[1, 0],
            fill=True,
            alpha=0.5,
            color="#F5E6C8",
            linewidth=2,
        )
        axs[1, 0].set_title("Potential U(z,t) distribution", fontweight="bold")
        axs[1, 0].set_xlabel("Potential")
        axs[1, 0].grid(True, linestyle="--", alpha=0.25)
    else:
        axs[1, 0].axis("off")
        axs[1, 0].text(
            0.5,
            0.5,
            "Potential not computed\n(run full training for landscape panels)",
            ha="center",
            va="center",
            fontsize=11,
        )

    pearson = trainer.metrics.get("potential_logp_pearson")
    spearman = trainer.metrics.get("potential_logp_spearman")
    ax_sc = axs[1, 1]
    if has_potential:
        _, _, u_np, neg_log_p = compute_potential_neglogp_consistency(plot_adata)
        if u_np.size:
            ax_sc.scatter(
                neg_log_p,
                u_np,
                s=8,
                alpha=0.35,
                color="#778ccc",
                edgecolors="none",
                rasterized=True,
            )
        ax_sc.set_title("Potential vs -log KDE(z) consistency", fontweight="bold")
        ax_sc.set_xlabel("-log KDE(z)")
        ax_sc.set_ylabel("Potential U(z,t)")
        ax_sc.grid(True, linestyle="--", alpha=0.25)
    else:
        ax_sc.axis("off")
        ax_sc.text(
            0.5,
            0.5,
            "Skipped in smoke test",
            ha="center",
            va="center",
            fontsize=11,
        )
    if has_potential:
        pearson_txt = f"{pearson:.4f}" if pearson is not None and np.isfinite(pearson) else "N/A"
        spearman_txt = f"{spearman:.4f}" if spearman is not None and np.isfinite(spearman) else "N/A"
        ax_sc.legend(
            handles=[
                Line2D([], [], linestyle="None", marker=" ", label=f"Pearson r = {pearson_txt}"),
                Line2D([], [], linestyle="None", marker=" ", label=f"Spearman r = {spearman_txt}"),
            ],
            loc="upper right",
            frameon=False,
            fontsize=9,
        )

    fig.suptitle("Training metrics", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_figure(fig, save_dir, "training_loss_curve.png", subdir="", close=True)


def detect_expression_peaks(adata, n_genes=200):
    from scipy.stats import gaussian_kde

    expr = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    pt = adata.obs["pseudotime"].values
    peak_counts = np.zeros_like(pt, dtype=int)
    for gene_idx in np.random.choice(expr.shape[1], min(n_genes, expr.shape[1]), replace=False):
        kde = gaussian_kde(pt)
        local_density = kde(pt)
        window_size = np.clip((1 / (local_density + 1e-8)) * 50, 10, 100).astype(int)
        smoothed = np.zeros_like(pt)
        for i in range(len(pt)):
            win = window_size[i]
            start = max(0, i - win // 2)
            end = min(len(pt), i + win // 2 + 1)
            smoothed[i] = expr[start:end, gene_idx].mean()
        peaks, _ = find_peaks(
            smoothed, prominence=np.std(smoothed) * 0.8, width=np.median(window_size)
        )
        peak_counts[peaks] += 1
    denom = peak_counts.max() - peak_counts.min()
    return (peak_counts - peak_counts.min()) / denom if denom > 0 else peak_counts


def gene_specific_mse(adata, marker_genes):
    pred_layer = _get_prediction_layer(adata)
    results = {}
    for gene in marker_genes:
        idx = adata.var_names.get_loc(gene)
        pred = pred_layer[:, idx]
        true = adata.X[:, idx].toarray().flatten() if sp.issparse(adata.X) else adata.X[:, idx]
        results[gene] = float(np.mean((pred - true) ** 2))
    return results


class KineticAnalyzer:
    def __init__(self, model, adata, config, save_dir, time_key=None, celltype=None):
        self.config = config
        self.model = model.to(config.device)
        self.save_dir = save_dir
        self.show_figures = getattr(config, "show_figures", False)
        self.figure_subdir = getattr(config, "figure_subdir", "figures")
        ensure_figure_dir(save_dir, self.figure_subdir)
        setup_scanpy_figdir(save_dir, self.figure_subdir)
        self.time_key = time_key or config.time_key
        self.celltype = celltype or config.cell_type_key
        self.adata = predict(model=model, adata=adata, config=config, save_dir=save_dir)
        self.model.load_state_dict(
            torch.load(f"{self.save_dir}/best_model.pth", map_location=config.device)
        )
        self.model.eval()

    def _save(self, filename, fig=None, dpi=None):
        return finish_figure(
            self.save_dir,
            filename,
            fig=fig,
            show_figures=self.show_figures,
            subdir=self.figure_subdir,
            dpi=dpi or _adaptive_dpi(self.adata.n_obs),
            rasterize=True,
        )

    def _prepare_velocity_data(self):
        if "spliced" not in self.adata.layers:
            x = self.adata.X.toarray() if sp.issparse(self.adata.X) else self.adata.X
            self.adata.layers["spliced"] = np.expm1(x)
        if "X_umap" not in self.adata.obsm:
            from dataset_pipeline import compute_training_umap

            compute_training_umap(self.adata, config=self.config)
        if "velocity" not in self.adata.layers:
            if "predicted_delta" in self.adata.layers and self.config.keep_velocity_alias:
                self.adata.layers["velocity"] = np.asarray(
                    self.adata.layers["predicted_delta"]
                )
                self.adata.uns["velocity_description"] = VELOCITY_LAYER_TITLE

    def infer_kinetics(self):
        self._prepare_velocity_data()
        if scv is None:
            warnings.warn(
                "scvelo is not installed; skip velocity graph post-processing.",
                UserWarning,
                stacklevel=2,
            )
            return
        if "velocity" not in self.adata.layers:
            warnings.warn(
                "No velocity layer available; skip velocity graph.",
                UserWarning,
                stacklevel=2,
            )
            return
        print(f"------------- scVelo | {VELOCITY_LAYER_TITLE} -------------")
        scv.pp.moments(self.adata)
        scv.tl.velocity_graph(self.adata, vkey="velocity", n_jobs=1)

    def plot_kinetics(self):
        style = getattr(self.config, "plot_style", None)
        label_key = style.celltype_key if style else resolve_label_key(self.adata, self.config)
        if label_key is None:
            label_key = self.celltype
        stage_key = style.stage_key if style else self.time_key

        if "velocity" in self.adata.layers and scv is not None:
            scv.tl.velocity_embedding(self.adata, basis="umap")

        plot_adata = self.adata  # full cohort; legend counts = true abundances

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.patch.set_facecolor("white")
        pt_size = 8 if plot_adata.n_obs > 10000 else 12
        pseudotime_cmap = style.pseudotime_cmap if style else "magma"

        plot_embedding_panel(
            plot_adata,
            axes[0, 0],
            label_key,
            title="Cell type",
            size=pt_size,
            discrete=True,
            config=self.config,
        )
        plot_embedding_panel(
            plot_adata,
            axes[0, 1],
            "pseudotime",
            title="Pseudotime",
            cmap=pseudotime_cmap,
            size=pt_size,
            discrete=False,
            config=self.config,
        )
        if stage_key in plot_adata.obs:
            plot_embedding_panel(
                plot_adata,
                axes[1, 0],
                stage_key,
                title="Stage",
                size=pt_size,
                discrete=True,
                config=self.config,
            )
        else:
            axes[1, 0].axis("off")

        plot_embedding_panel(
            plot_adata,
            axes[1, 1],
            label_key,
            title="Cell type + expression flow",
            size=pt_size,
            discrete=True,
            config=self.config,
        )
        if "velocity_umap" in plot_adata.obsm:
            xy = np.asarray(plot_adata.obsm["X_umap"])
            v = np.asarray(plot_adata.obsm["velocity_umap"])
            step = max(1, plot_adata.n_obs // 2500)
            idx = np.arange(0, plot_adata.n_obs, step)
            axes[1, 1].quiver(
                xy[idx, 0],
                xy[idx, 1],
                v[idx, 0],
                v[idx, 1],
                angles="xy",
                scale_units="xy",
                scale=25,
                width=0.0025,
                alpha=0.55,
                color="#2f2f2f",
                zorder=3,
            )

        if plot_adata.n_obs < self.adata.n_obs:
            fig.suptitle(
                f"{VELOCITY_LAYER_TITLE}\n"
                f"({plot_adata.n_obs:,} / {self.adata.n_obs:,} cells shown)",
                fontsize=12,
                fontweight="bold",
                y=0.995,
            )
        else:
            fig.suptitle(VELOCITY_LAYER_TITLE, fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        self._save("kinetics_overview.png", fig=fig)

    def plot_gene_trends(self, genes=None, n_genes=3, use_pseudotime=True):
        if genes is None:
            weights = self.model.gene_encoder[0].weight.detach().cpu().numpy()
            gene_scores = np.linalg.norm(weights, axis=0)
            genes = self.adata.var_names[
                np.argsort(gene_scores)[::-1][:n_genes]
            ].tolist()
        from statsmodels.nonparametric.smoothers_lowess import lowess

        group_key = resolve_violin_groupby_key(self.adata, self.config)
        fig, axes = plt.subplots(len(genes), 2, figsize=(16, 5 * len(genes)))
        if len(genes) == 1:
            axes = np.array([axes])
        for i, gene in enumerate(genes):
            expr = (
                self.adata[:, gene].X.toarray().flatten()
                if sp.issparse(self.adata.X)
                else self.adata[:, gene].X.flatten()
            )
            sc.pl.violin(
                self.adata,
                gene,
                groupby=group_key,
                ax=axes[i, 0],
                show=False,
                rotation=45,
            )
            axes[i, 0].set_title(f"{gene} by {group_key}", fontweight="bold")
            x = (
                self.adata.obs["pseudotime"]
                if use_pseudotime
                else self.adata.obs[self.time_key].astype(float)
            )
            plot_embedding_panel(
                self.adata,
                axes[i, 1],
                gene,
                title=f"{gene} on UMAP",
                cmap="viridis",
                size=6 if self.adata.n_obs > 30000 else 8,
            )
            inset = axes[i, 1].inset_axes([0.58, 0.58, 0.38, 0.38])
            inset.scatter(x, expr, alpha=0.25, s=4, c="#457b9d", linewidths=0)
            smooth = lowess(expr, x, frac=0.2)
            inset.plot(smooth[:, 0], smooth[:, 1], c="#e07a2f", lw=2)
            inset.set_title("Trend", fontsize=8)
            inset.grid(True, linestyle="--", alpha=0.25)
        plt.tight_layout()
        self._save("gene_trends.png", fig=fig)

    def plot_potential_landscape(self):
        style = getattr(self.config, "plot_style", None)
        label_key = style.celltype_key if style else resolve_label_key(self.adata, self.config)
        plot_adata = self.adata  # full cohort; legend counts = true abundances
        embed = np.asarray(plot_adata.obsm["X_umap"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.patch.set_facecolor("white")

        sc1 = axes[0].scatter(
            embed[:, 0],
            embed[:, 1],
            c=plot_adata.obs["potential"],
            cmap=style.potential_cmap if style else "RdYlBu_r",
            s=8,
            alpha=0.75,
            linewidths=0,
            rasterized=True,
        )
        cb1 = fig.colorbar(sc1, ax=axes[0], pad=0.02)
        cb1.set_label("U(z,t)")
        axes[0].set_xlabel("UMAP1")
        axes[0].set_ylabel("UMAP2")
        axes[0].set_title("Learned potential on UMAP", fontweight="bold")

        if label_key:
            plot_embedding_panel(
                plot_adata,
                axes[1],
                label_key,
                title="Cell type labels",
                size=8 if plot_adata.n_obs > 10000 else 12,
                discrete=True,
                config=self.config,
            )
        else:
            axes[1].axis("off")

        plt.tight_layout()
        self._save("potential_landscape_umap.png", fig=fig)

    def plot_genes(self, gene_list, ncols=3, smooth_method="lowess", span=0.3, figsize=(10, 5)):
        from statsmodels.nonparametric.smoothers_lowess import lowess

        pseudotime = self.adata.obs["pseudotime"].values
        pred = _get_prediction_layer(self.adata)
        n_genes = len(gene_list)
        nrows = int(np.ceil(n_genes / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(figsize[0], 4 * ncols), max(figsize[1], 4 * nrows)))
        axes = np.atleast_1d(axes).flatten()
        for i, gene in enumerate(gene_list):
            idx = self.adata.var_names.get_loc(gene)
            y = pred[:, idx]
            order = np.argsort(pseudotime)
            x = pseudotime[order]
            y = y[order]
            ax = axes[i]
            ax.scatter(x, y, alpha=0.25, s=12, c="#457b9d", linewidths=0)
            if smooth_method == "lowess":
                sm = lowess(y, x, frac=span, it=3)
                ax.plot(sm[:, 0], sm[:, 1], color="#e07a2f", lw=2.5)
            ax.set_title(f"{gene} (predicted)", fontweight="bold")
            ax.set_xlabel("Pseudotime")
            ax.set_ylabel("Expression")
            ax.grid(True, linestyle="--", alpha=0.25)
        for j in range(n_genes, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        self._save("predicted_genes_vs_pseudotime.png", fig=fig)

    def plot_gene_phase(
        self,
        gene_x,
        gene_y,
        color_by="pseudotime",
        figsize=(8, 6),
        point_size=15,
        alpha=0.6,
        add_trend=False,
        trend_style="lowess",
        **kwargs,
    ):
        import seaborn as sns
        from scipy import stats
        from statsmodels.nonparametric.smoothers_lowess import lowess

        mat = _get_prediction_layer(self.adata)
        ix = self.adata.var_names.get_loc(gene_x)
        iy = self.adata.var_names.get_loc(gene_y)
        x, y = mat[:, ix], mat[:, iy]
        c = self.adata.obs[color_by]
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        if hasattr(c, "cat"):
            sns.scatterplot(x=x, y=y, hue=c, ax=ax, s=point_size, alpha=alpha, linewidth=0)
            ax.legend(title=color_by, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        else:
            scat = ax.scatter(x, y, c=c, cmap="viridis", s=point_size, alpha=alpha, linewidths=0)
            fig.colorbar(scat, ax=ax, label=color_by, pad=0.02)
        if add_trend:
            if trend_style == "lowess":
                sm = lowess(y, x, frac=0.3, it=3)
                ax.plot(sm[:, 0], sm[:, 1], color="#e07a2f", linewidth=2.5, label="LOESS")
            elif trend_style == "poly":
                coeffs = np.polyfit(x, y, deg=3)
                poly = np.poly1d(coeffs)
                x_cont = np.linspace(x.min(), x.max(), 200)
                ax.plot(x_cont, poly(x_cont), color="#9d0208", linewidth=2.5, label="Cubic fit")
            ax.legend(fontsize=9, frameon=False)

        r = stats.spearmanr(x, y)
        ax.text(
            0.03,
            0.97,
            f"Spearman r = {r.correlation:.2f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        )
        ax.set_xlabel(gene_x)
        ax.set_ylabel(gene_y)
        ax.set_title("Predicted expression phase plot", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.25)
        plt.tight_layout()
        self._save(f"gene_phase_{gene_x}_vs_{gene_y}.png", fig=fig)


AutoGenerator = LatentSDEFunc
TemporalSDETransformer = TemporalSDENetwork  # 兼容旧 LAP / 脚本命名
