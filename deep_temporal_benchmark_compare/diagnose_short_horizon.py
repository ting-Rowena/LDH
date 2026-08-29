#!/usr/bin/env python
"""Short-horizon (adjacent-timepoint) diagnostic for the latent-SDE model.

Question this answers: does the model fail because its *local dynamics* are poorly
calibrated, or only because *long-horizon extrapolation* compounds error?

For every adjacent timepoint pair (t_i -> t_{i+1}) across the FULL series it:
  - predicts the t_{i+1} population from the t_i population with the model (one step
    over that interval) and with persistence (x_next = x_curr),
  - scores both against the observed t_{i+1} population on the same distribution
    metrics used in the baseline table (Sinkhorn OT / MMD / energy / mean per-gene W1 /
    mean-shift L2),
  - additionally reports the cosine between the model's mean displacement and the
    observed mean displacement (direction correctness; persistence has zero displacement
    so no direction).

Note on splits: with the publication checkpoint (val_mode=time_extrapolate, holdout t=28)
the model was *trained* on transitions among {0,3,7,10,14,21}; only the last adjacent
pair (21->28) is a truly held-out one-step transition. In-sample pairs still tell us
whether the model even fits the local flow (over-dispersion shows up here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from train_model import Config, TemporalDataProcessor, TemporalSDENetwork, _load_state_dict_compat, _sinkhorn_plan
from baseline_evaluation import (
    _model_predict_population,
    _subsample,
    _sync_config_to_checkpoint,
    population_distribution_metrics,
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _encode_np(model, x, ct, device, batch_size=512):
    """Encode cells to latent space (no dynamics)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            e = min(s + batch_size, len(x))
            xb = torch.tensor(x[s:e], dtype=torch.float32, device=device)
            cb = torch.tensor(ct[s:e], dtype=torch.long, device=device)
            outs.append(model.encode(xb, cb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def _latent_predict(model, x, ct, t_curr, t_next, device, batch_size=512):
    """Return (z_src, z_pred): encoded source and its latent state after the dynamics.

    Fully decoder-free: tests whether the flow predicts the next *latent* population
    better than doing nothing. integrate_latent needs autograd for -grad U, so no
    torch.no_grad() here; outputs are detached.
    """
    model.eval()
    z_src_list, z_pred_list = [], []
    for s in range(0, len(x), batch_size):
        e = min(s + batch_size, len(x))
        xb = torch.tensor(x[s:e], dtype=torch.float32, device=device)
        cb = torch.tensor(ct[s:e], dtype=torch.long, device=device)
        z = model.encode(xb, cb)
        ts = torch.tensor([float(t_curr), float(t_next)], dtype=torch.float32, device=device)
        z_pred = model.integrate_latent(z, ts)[-1]
        z_src_list.append(z.detach().cpu().numpy().astype(np.float32))
        z_pred_list.append(z_pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(z_src_list, axis=0), np.concatenate(z_pred_list, axis=0)


def _ot_barycentric_targets(z_src, z_tgt, ct_src, ct_tgt, blur=0.05):
    """Per-source-cell entropic-OT barycentric image in z_tgt (per cell type)."""
    tgt = np.zeros_like(z_src)
    for c in np.unique(ct_src):
        ms = ct_src == c
        mt = ct_tgt == c
        if int(ms.sum()) < 1 or int(mt.sum()) < 1:
            continue
        zs = torch.tensor(z_src[ms], dtype=torch.float32)
        zt = torch.tensor(z_tgt[mt], dtype=torch.float32)
        with torch.no_grad():
            plan = _sinkhorn_plan(zs, zt, blur=blur)
            row = plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
            bary = (plan @ zt) / row
        tgt[ms] = bary.numpy().astype(np.float32)
    return tgt


def _per_cell_disp_ratios(z_src, z_pred, ot_tgt, eps=1e-9):
    pred_norm = np.linalg.norm(z_pred - z_src, axis=1)
    true_norm = np.linalg.norm(ot_tgt - z_src, axis=1)
    return pred_norm / (true_norm + eps)


def _ae_reconstruct(model, x, ct, t_val, device, batch_size=512):
    """Encode source cells and decode at time ``t_val`` WITHOUT running the dynamics.

    Isolates autoencoder reconstruction distortion (the cost persistence avoids by
    staying in raw expression space).
    """
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            e = min(s + batch_size, len(x))
            xb = torch.tensor(x[s:e], dtype=torch.float32, device=device)
            cb = torch.tensor(ct[s:e], dtype=torch.long, device=device)
            z = model.encode(xb, cb)
            t_col = torch.full((z.shape[0], 1), float(t_val), device=device)
            expr = model.predict_expression(z, t_col)
            outs.append(expr.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def main(argv=None):
    from run_training import DATASETS
    from dataset_pipeline import apply_train_config, build_training_checkpoint_dir, resolve_data_path

    ap = argparse.ArgumentParser(description="Short-horizon adjacent-transition diagnostic")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--processed-cache", default=None)
    ap.add_argument("--max-source-cells", type=int, default=512)
    ap.add_argument("--max-cells", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    spec, prepare_fn = DATASETS[args.dataset]
    config = apply_train_config(spec)
    config.val_mode = "time_extrapolate"
    config.data_path = resolve_data_path(spec)

    cache = args.processed_cache
    if cache and Path(cache).is_file():
        print(f"Loading cached processed AnnData from {cache}", flush=True)
        adata = sc.read(cache)
    else:
        adata = sc.read(config.data_path)
        adata = prepare_fn(adata, config)
        adata = TemporalDataProcessor(adata).process()
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            adata.write(cache)

    checkpoint_dir = args.checkpoint_dir or build_training_checkpoint_dir(spec, config)
    print(f"Checkpoint dir: {checkpoint_dir}", flush=True)

    device = config.device
    ckpt = Path(checkpoint_dir) / "best_model.pth"
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    _sync_config_to_checkpoint(config, checkpoint_dir)
    model = TemporalSDENetwork(config, adata).to(device)
    _load_state_dict_compat(model, torch.load(ckpt, map_location=device))

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)
    t_all = adata.obs[config.time_key].astype(float).values
    ct_all = adata.obs[config.cell_type_key].values.astype(int)
    times = sorted(set(float(t) for t in t_all))
    print(f"Timepoints: {times}", flush=True)

    rng = np.random.RandomState(args.seed)
    rows: List[dict] = []
    for a, b in zip(times[:-1], times[1:]):
        src_mask = np.isclose(t_all, a)
        tgt_mask = np.isclose(t_all, b)
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            continue
        x_src_full = X[src_mask]
        ct_src_full = ct_all[src_mask]
        x_tgt = X[tgt_mask]
        ct_tgt = ct_all[tgt_mask]

        idx = np.arange(x_src_full.shape[0])
        if args.max_source_cells and idx.shape[0] > args.max_source_cells:
            idx = rng.choice(idx, size=args.max_source_cells, replace=False)
        x_src = x_src_full[idx]
        ct_src = ct_src_full[idx]

        model_pred = _model_predict_population(model, x_src, ct_src, a, b, device)
        pers_pred = np.asarray(x_src, dtype=np.float32)
        # Autoencoder control: encode source and decode WITHOUT moving z. This isolates
        # the reconstruction error the model pays (but persistence, in raw space, does not).
        ae_self = _ae_reconstruct(model, x_src, ct_src, a, device)  # decode @ t_curr
        ae_next = _ae_reconstruct(model, x_src, ct_src, b, device)  # decode @ t_next, no dynamics

        mm = population_distribution_metrics(model_pred, x_tgt, max_cells=args.max_cells, seed=args.seed)
        mp = population_distribution_metrics(pers_pred, x_tgt, max_cells=args.max_cells, seed=args.seed)
        # AE floor: how far the pure round-trip distorts the source distribution.
        ae_self_energy = population_distribution_metrics(
            ae_self, x_src, max_cells=args.max_cells, seed=args.seed
        )["energy_distance"]
        # AE-persistence: decode-but-don't-move, scored against the true next population
        # (the fair analogue of persistence *through* the autoencoder).
        ae_persist = population_distribution_metrics(
            ae_next, x_tgt, max_cells=args.max_cells, seed=args.seed
        )
        ae_persist_energy = ae_persist["energy_distance"]
        ae_persist_w1 = ae_persist["mean_marginal_w1"]

        obs_disp = x_tgt.mean(0) - x_src.mean(0)
        model_disp = model_pred.mean(0) - x_src.mean(0)
        cos = _cos(model_disp, obs_disp)
        disp_ratio = float(np.linalg.norm(model_disp) / (np.linalg.norm(obs_disp) + 1e-9))

        # ---- Decoder-free latent-space fair test ----
        z_src_full = _encode_np(model, x_src_full, ct_src_full, device)
        z_tgt_full = _encode_np(model, x_tgt, ct_tgt, device)
        z_src, z_pred = _latent_predict(model, x_src, ct_src, a, b, device)
        tgt_idx = np.arange(x_tgt.shape[0])
        if args.max_source_cells and tgt_idx.shape[0] > args.max_source_cells:
            tgt_idx = rng.choice(tgt_idx, size=args.max_source_cells, replace=False)
        z_tgt = z_tgt_full[tgt_idx]
        lat_m = population_distribution_metrics(z_pred, z_tgt, max_cells=args.max_cells, seed=args.seed)
        lat_p = population_distribution_metrics(z_src, z_tgt, max_cells=args.max_cells, seed=args.seed)
        lat_obs_disp = z_tgt.mean(0) - z_src.mean(0)
        lat_model_disp = z_pred.mean(0) - z_src.mean(0)
        lat_cos = _cos(lat_model_disp, lat_obs_disp)
        ot_blur = float(getattr(config, "latent_disp_ot_blur", 0.05) or 0.05)
        ot_tgt_src = _ot_barycentric_targets(
            z_src, z_tgt_full, ct_src, ct_tgt, blur=ot_blur
        )
        per_cell_ratios = _per_cell_disp_ratios(z_src, z_pred, ot_tgt_src)
        lat_pc_ratio_med = float(np.median(per_cell_ratios))
        lat_pc_ratio_p90 = float(np.percentile(per_cell_ratios, 90))

        row = {
            "t_curr": a,
            "t_next": b,
            "dt": b - a,
            "n_src": int(src_mask.sum()),
            "n_tgt": int(tgt_mask.sum()),
            "model_energy": mm["energy_distance"],
            "pers_energy": mp["energy_distance"],
            "ae_persist_energy": ae_persist_energy,
            "ae_self_energy": ae_self_energy,
            "model_w1": mm["mean_marginal_w1"],
            "pers_w1": mp["mean_marginal_w1"],
            "ae_persist_w1": ae_persist_w1,
            "model_ot": mm["ot_sinkhorn"],
            "pers_ot": mp["ot_sinkhorn"],
            "model_mmd": mm["mmd"],
            "pers_mmd": mp["mmd"],
            "dir_cosine": cos,
            "disp_norm_ratio": disp_ratio,
            "lat_model_energy": lat_m["energy_distance"],
            "lat_pers_energy": lat_p["energy_distance"],
            "lat_model_w1": lat_m["mean_marginal_w1"],
            "lat_pers_w1": lat_p["mean_marginal_w1"],
            "lat_model_ot": lat_m["ot_sinkhorn"],
            "lat_pers_ot": lat_p["ot_sinkhorn"],
            "lat_dir_cosine": lat_cos,
            "lat_per_cell_disp_ratio_median": lat_pc_ratio_med,
            "lat_per_cell_disp_ratio_p90": lat_pc_ratio_p90,
        }
        rows.append(row)
        print(
            f"[{a:>5}->{b:<5}] dt={b-a:<5g} "
            f"RAW energy model={mm['energy_distance']:.3f} pers={mp['energy_distance']:.3f} "
            f"| LATENT energy model={lat_m['energy_distance']:.3f} pers={lat_p['energy_distance']:.3f} "
            f"w1 model={lat_m['mean_marginal_w1']:.3f} pers={lat_p['mean_marginal_w1']:.3f} "
            f"lat_cos={lat_cos:.3f} pc_ratio_med={lat_pc_ratio_med:.3f} pc_ratio_p90={lat_pc_ratio_p90:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "short_horizon_diagnostic.csv", index=False)

    # Summary: on how many adjacent steps does the model beat persistence, and by how much?
    if not df.empty:
        df["model_beats_pers_energy"] = df["model_energy"] < df["pers_energy"]
        df["model_beats_pers_w1"] = df["model_w1"] < df["pers_w1"]
        # Does the dynamics help beyond just decode-in-place (the autoencoder floor)?
        df["model_beats_aepersist_energy"] = df["model_energy"] < df["ae_persist_energy"]
        # Decoder-free latent-space fair test: does the flow beat latent-persistence?
        df["lat_model_beats_pers_energy"] = df["lat_model_energy"] < df["lat_pers_energy"]
        df["lat_model_beats_pers_w1"] = df["lat_model_w1"] < df["lat_pers_w1"]
        n = len(df)
        try:
            table = df.round(4).to_markdown(index=False)
        except ImportError:
            table = df.round(4).to_string(index=False)
        lines = [
            "# Short-horizon (adjacent-timepoint) diagnostic",
            "",
            "Lower is better for energy/w1/ot/mmd. `dir_cosine` in [-1,1]: +1 = model's "
            "mean displacement points the same way as the observed one (direction correct); "
            "`disp_norm_ratio` ~1 = right magnitude, >1 = over-shoots, <1 = under-shoots. "
            "`lat_per_cell_disp_ratio_*` use full-population OT barycentric targets per cell "
            "(the metric aligned with OT-ratio training).",
            "",
            table,
            "",
            "## Summary",
            "",
            f"- adjacent steps evaluated: {n}",
            f"- model beats persistence (raw space) on energy: {int(df['model_beats_pers_energy'].sum())}/{n}",
            f"- model beats persistence (raw space) on W1: {int(df['model_beats_pers_w1'].sum())}/{n}",
            f"- model beats AE-persistence (decode-in-place) on energy: "
            f"{int(df['model_beats_aepersist_energy'].sum())}/{n}  <- isolates the dynamics",
            f"- mean autoencoder floor (ae_self energy, source round-trip): {df['ae_self_energy'].mean():.3f}",
            f"- mean AE-persistence energy (decode, no motion): {df['ae_persist_energy'].mean():.3f}",
            f"- mean model energy: {df['model_energy'].mean():.3f}",
            f"- mean persistence (raw) energy: {df['pers_energy'].mean():.3f}",
            f"- mean direction cosine (raw): {df['dir_cosine'].mean():.3f}",
            f"- mean displacement-norm ratio (model/obs): {df['disp_norm_ratio'].mean():.2f}",
            "",
            "### Decoder-free latent-space fair test (the clinching comparison)",
            f"- LATENT: model beats latent-persistence on energy: "
            f"{int(df['lat_model_beats_pers_energy'].sum())}/{n}",
            f"- LATENT: model beats latent-persistence on W1: "
            f"{int(df['lat_model_beats_pers_w1'].sum())}/{n}",
            f"- LATENT mean model energy: {df['lat_model_energy'].mean():.3f}  "
            f"vs latent-persistence: {df['lat_pers_energy'].mean():.3f}",
            f"- LATENT mean model W1: {df['lat_model_w1'].mean():.4f}  "
            f"vs latent-persistence: {df['lat_pers_w1'].mean():.4f}",
            f"- LATENT mean direction cosine: {df['lat_dir_cosine'].mean():.3f}",
            f"- LATENT per-cell disp ratio (OT target) median: "
            f"{df['lat_per_cell_disp_ratio_median'].mean():.3f}  "
            f"(population-mean `disp_norm_ratio` analogue: {df['disp_norm_ratio'].mean():.2f})",
            f"- LATENT per-cell disp ratio (OT target) p90 mean: "
            f"{df['lat_per_cell_disp_ratio_p90'].mean():.3f}",
        ]
        (out / "short_horizon_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n" + "\n".join(lines[-8:]), flush=True)
    print(f"\nWritten to {out}/short_horizon_diagnostic.md", flush=True)


if __name__ == "__main__":
    sys.exit(main())
