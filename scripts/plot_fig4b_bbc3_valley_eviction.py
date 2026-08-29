#!/usr/bin/env python
"""Fig4B-style deep-valley U0 histogram for BBC3 hybrid KO (PDVS top gene).

Mirrors the published SOD2 panel protocol (hybrid reencode + unit latent shift on
protocol deep-valley EOC barcodes) on CPU, then writes:

  methods_enhancement/in_silico_KO_BBC3_hybrid_shift1_valley_eviction.{csv,png}
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from panel_style import apply_panel_title_rc, set_panel_title, PANEL_TITLE_SIZE
from dataset_pipeline import HGSOC, apply_train_config, resolve_data_path  # noqa: E402
from methods_model_utils import _load_state_dict_compat  # noqa: E402
from plot_utils import PALETTE, configure_headless, style_axis  # noqa: E402
from train_model import TemporalSDENetwork  # noqa: E402

configure_headless()
apply_panel_title_rc()

CK = ROOT / (
    "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
OUT = CK / "methods_enhancement"
PANELS = OUT
TABLES = OUT
GENE = "BBC3"
PUB_CUT = -0.0220824908465147


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * 0.0 if (not np.isfinite(n) or n < 1e-12) else v / n


def _encode(model, X, ct, device, bs=256, stage=None):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            x = torch.tensor(X[s : s + bs], dtype=torch.float32, device=device)
            c = torch.tensor(ct[s : s + bs], dtype=torch.long, device=device)
            st = None
            if stage is not None:
                st = torch.tensor(stage[s : s + bs], dtype=torch.long, device=device)
            outs.append(model.encode(x, c, stage=st).cpu().numpy())
    return np.vstack(outs)


def _U(model, z, device, bs=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(z), bs):
            t = torch.tensor(z[s : s + bs], dtype=torch.float32, device=device)
            outs.append(model.stationary_potential(t).squeeze(-1).cpu().numpy())
    return np.concatenate(outs)


def _boot_gt0(x, n=500, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    c = sum(1 for _ in range(n) if float(np.mean(rng.choice(x, x.size, True))) <= 0)
    return float((c + 1) / (n + 1))


def load_valley_matrix_and_model():
    print("[1/4] obs + panel ...", flush=True)
    obs = pd.read_csv(CK / "obs.csv", low_memory=False)
    if "cell" in obs.columns:
        obs = obs.set_index(obs["cell"].astype(str), drop=False)
    else:
        obs.index = obs.index.astype(str)
    gene_list = json.loads((CK / "training_var_names.json").read_text(encoding="utf-8"))
    valley_bc = (
        pd.read_csv(CK / "analysis_protocol_HGSOC" / "eoc_attractor_deep_valley_barcodes.csv")["barcode"]
        .astype(str)
        .tolist()
    )
    eoc = obs[obs["annotation"].astype(str) == "EOC"].copy()
    eoc_barcodes = eoc.index.astype(str).tolist()
    print(f"  EOC={len(eoc_barcodes)} valley_barcodes={len(valley_bc)}", flush=True)

    h5ad = resolve_data_path(HGSOC)
    print(f"[2/4] read EOC cells from {h5ad} ...", flush=True)
    raw = ad.read_h5ad(h5ad, backed="r")
    name_to_i = {b: i for i, b in enumerate(raw.obs_names.astype(str))}
    idx = [name_to_i[b] for b in eoc_barcodes if b in name_to_i]
    keep = [b for b in eoc_barcodes if b in name_to_i]
    # Avoid simultaneous fancy indexing on rows+cols (h5py restriction).
    sub = raw[idx].to_memory()
    raw.file.close()
    present = [g for g in gene_list if g in set(map(str, sub.var_names))]
    sub = sub[:, present].copy()
    missing = [g for g in gene_list if g not in sub.var_names]
    if missing:
        zeros = sp.csr_matrix((sub.n_obs, len(missing)), dtype=np.float32)
        filler = ad.AnnData(X=zeros, obs=sub.obs.copy(), var=pd.DataFrame(index=pd.Index(missing)))
        sub = ad.concat([sub, filler], axis=1, join="outer", merge="same")
    sub = sub[:, list(gene_list)].copy()
    sub.obs_names = pd.Index(keep)
    sub.obs["cell_type"] = eoc.loc[keep, "cell_type"].astype(int).values
    sub.obs["stage_code"] = eoc.loc[keep, "stage_code"].astype(int).values
    sub.obs["deep_valley"] = sub.obs_names.astype(str).isin(set(valley_bc))
    if "log1p" not in sub.uns:
        sc.pp.normalize_total(sub, target_sum=1e4, inplace=True)
        sc.pp.log1p(sub)

    print("[3/4] model on CPU ...", flush=True)
    config = apply_train_config(HGSOC)
    config.n_top_genes = len(gene_list)
    config.use_hvg = True
    config.device = "cpu"
    config.show_figures = False
    config.cell_type_key = "cell_type"
    config.use_stage_embedding = True
    config.n_stages = 3
    config.stage_cond_key = "stage_code"
    # Full-cohort n_types so type_embed matches checkpoint (EOC-only max can be < 3).
    n_types = int(obs["cell_type"].astype(int).max()) + 1
    stub = ad.AnnData(X=np.zeros((max(n_types, 3), len(gene_list)), dtype=np.float32))
    stub.var_names = pd.Index(gene_list)
    stub.obs["cell_type"] = np.arange(stub.n_obs, dtype=int)
    stub.obs["stage_code"] = 0
    model = TemporalSDENetwork(config, stub)
    state = torch.load(CK / "best_model.pth", map_location="cpu")
    _load_state_dict_compat(model, state)
    model = model.to("cpu").eval()

    X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X, dtype=float)
    ct = sub.obs["cell_type"].astype(int).values
    stage = sub.obs["stage_code"].astype(int).values
    valley = sub.obs["deep_valley"].to_numpy(bool)
    bs = int(getattr(config, "batch_size", 256))
    print("[3b/4] encode all EOC → U0 + q15 cutoff ...", flush=True)
    z_all = _encode(model, X, ct, "cpu", bs=bs, stage=stage)
    pot = _U(model, z_all, "cpu", bs=bs)
    u_cut = float(np.nanquantile(pot, 0.15))
    print(
        f"  matrix={X.shape} valley={int(valley.sum())} q15={u_cut:.6f} "
        f"valley_mean_U={float(np.mean(pot[valley])):.6f}",
        flush=True,
    )
    return model, config, X, ct, stage, valley, pot, u_cut, gene_list


def main() -> None:
    model, config, X, ct, stage, valley, pot, u_cut, gene_list = load_valley_matrix_and_model()
    if GENE not in gene_list:
        raise SystemExit(f"{GENE} not in HVG training panel")
    gcol = list(gene_list).index(GENE)
    bs = int(getattr(config, "batch_size", 256))

    print(f"[4/4] hybrid KO {GENE} on valley cells ...", flush=True)
    X_kd = X.copy()
    X_kd[:, gcol] = 0.0
    z_wt = _encode(model, X[valley], ct[valley], "cpu", bs=bs, stage=stage[valley])
    z_kd = _encode(model, X_kd[valley], ct[valley], "cpu", bs=bs, stage=stage[valley])
    z_kd = z_wt + _unit((z_kd - z_wt).mean(0))[None, :]

    u_wt = pot[valley]
    u_kd = _U(model, z_kd, "cpu", bs=bs)
    du = u_kd - u_wt
    mean_du = float(np.nanmean(du))
    frac = float(np.mean(u_kd > u_cut))
    frac_pub = float(np.mean(u_kd > PUB_CUT))
    try:
        _, wil_p = stats.wilcoxon(u_kd, u_wt, alternative="greater", zero_method="wilcox")
        wil_p = float(wil_p)
    except Exception:
        wil_p = float("nan")
    boot_p = _boot_gt0(du, seed=abs(hash(GENE + "hybrid")) % (2**31))

    summary = {
        "gene": GENE,
        "perturbation": "KO",
        "expr_factor": 0.0,
        "ko_mode": "hybrid",
        "direction_tag": "encoder_delta",
        "latent_shift_scale": 1.0,
        "n_deep_valley_cells": int(valley.sum()),
        "valley_U_cutoff": float(u_cut),
        "mean_U_WT_valley": float(np.nanmean(u_wt)),
        "mean_U_pert_valley": float(np.nanmean(u_kd)),
        "mean_delta_U": mean_du,
        "frac_escape_valley": frac,
        "frac_escape_published_cutoff": frac_pub,
        "raises_potential": bool(mean_du > 0),
        "wilcoxon_U_increase_p": wil_p,
        "bootstrap_mean_dU_gt0_p": boot_p,
        "eviction_significant_at_0.05": bool(
            (np.isfinite(wil_p) and wil_p < 0.05) or (boot_p < 0.05)
        ),
        "in_original_HVG_panel": True,
        "note": "PDVS top gene; Fig4B companion to SOD2",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    csv_name = f"in_silico_KO_{GENE}_hybrid_shift1_valley_eviction.csv"
    pd.DataFrame([summary]).to_csv(OUT / csv_name, index=False)
    shutil.copy2(OUT / csv_name, TABLES / f"Fig4B_{GENE}_valley_eviction.csv")

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(u_wt, bins=30, alpha=0.55, color=PALETTE[0], label="WT valley U0", density=True)
    ax.hist(u_kd, bins=30, alpha=0.55, color=PALETTE[5], label=f"KO {GENE} U0", density=True)
    ax.axvline(u_cut, color="k", ls="--", lw=1.2, label=f"valley cutoff={u_cut:.3g}")
    ax.set_xlabel("Stationary potential U0")
    ax.set_ylabel("Density")
    ax.set_title(f"HGSOC deep-valley EOC: KO {GENE}", loc="center", fontweight="bold", fontsize=PANEL_TITLE_SIZE)
    ax.legend(fontsize=8)
    style_axis(ax, grid_axis="y")
    fig.tight_layout()

    png_me = OUT / "figures" / f"in_silico_KO_{GENE}_hybrid_shift1_valley_eviction.png"
    png_panel = PANELS / f"Fig4B_{GENE}.png"
    fig.savefig(png_me, dpi=300, bbox_inches="tight")
    fig.savefig(png_panel, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"BBC3 hybrid: escape={frac:.3f} (pub_cut={frac_pub:.3f}) "
        f"mean_dU={mean_du:+.6f} boot_p={boot_p:.4g}",
        flush=True,
    )
    print(f"wrote {png_panel}", flush=True)
    print(f"wrote {png_me}", flush=True)
    print(f"wrote {OUT / csv_name}", flush=True)


if __name__ == "__main__":
    main()
