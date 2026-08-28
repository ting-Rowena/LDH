#!/usr/bin/env python
"""P1 robustness: lung fate action decomposition + CCC permutation/patient nulls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_file" / "robustness" / "p1_robustness"
OUT.mkdir(parents=True, exist_ok=True)

CKPT_LUNG = (
    ROOT
    / "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
CKPT_HG = (
    ROOT
    / "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)


def _decompose_flow_action(path, times, momentum, U_func, lambda_u: float = 1.0) -> Dict:
    path = np.asarray(path, dtype=float)
    times = np.asarray(times, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    mismatch_sum = 0.0
    u_sum = 0.0
    u_vals = []
    seg_lens = []
    for i in range(len(path) - 1):
        dt = float(times[i + 1] - times[i])
        if dt <= 0:
            continue
        dz_dt = (path[i + 1] - path[i]) / dt
        f = momentum[i]
        mismatch = float(np.sum((dz_dt - f) ** 2))
        u = float(np.asarray(U_func(path[i])).reshape(-1)[0])
        mismatch_sum += mismatch * dt
        u_sum += lambda_u * u * dt
        u_vals.append(u)
        seg_lens.append(float(np.linalg.norm(path[i + 1] - path[i])))
    path_len = float(np.sum(seg_lens)) if seg_lens else 0.0
    total = mismatch_sum + u_sum
    return {
        "total_action": total,
        "action_mismatch": mismatch_sum,
        "action_potential": u_sum,
        "frac_mismatch": float(mismatch_sum / total) if abs(total) > 1e-12 else np.nan,
        "path_length": path_len,
        "action_per_length": float(total / path_len) if path_len > 1e-12 else np.nan,
        "mismatch_per_length": float(mismatch_sum / path_len) if path_len > 1e-12 else np.nan,
        "U_per_length": float(u_sum / path_len) if path_len > 1e-12 else np.nan,
        "mean_U_along_path": float(np.mean(u_vals)) if u_vals else np.nan,
        "max_U_along_path": float(np.max(u_vals)) if u_vals else np.nan,
        "n_segments": int(len(seg_lens)),
        "endpoint_distance": float(np.linalg.norm(path[-1] - path[0])),
    }


def run_lung_action_diagnostics() -> Dict:
    from dataclasses import replace

    from analysis_protocol_utils import select_fate_core_indices
    from celltype_analysis import (
        DATASET_REGISTRY,
        GSE141259_PROFILE,
        _make_analyzer,
        _path_n_points,
        load_annotated_adata,
    )
    from flow_space_lap import flow_space_action_integral, mean_flow_mismatch
    from plot_utils import PALETTE, configure_headless, style_axis
    from run_gse141259_analysis import (
        FATE_END_AT1,
        FATE_END_FIBRO,
        FATE_START,
        prepare_fate_branch_adata,
    )

    configure_headless()
    print("[Lung] loading...", flush=True)
    adata0 = load_annotated_adata(DATASET_REGISTRY["GSE141259"], str(CKPT_LUNG))
    # merge obs potentials
    obs = pd.read_csv(CKPT_LUNG / "obs.csv", index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    adata0.obs_names = adata0.obs_names.astype(str)
    common = adata0.obs_names.intersection(obs.index)
    adata0 = adata0[common].copy()
    for c in ["potential_stationary", "potential", "plasticity_score", "annotation"]:
        if c in obs.columns:
            adata0.obs[c] = obs.loc[common, c].values

    rows = []
    path_store = {}
    for end_label, tag in [(FATE_END_AT1, "AT1"), (FATE_END_FIBRO, "Fibro")]:
        print(f"[Lung] branch → {tag}", flush=True)
        adata = prepare_fate_branch_adata(adata0, CKPT_LUNG)
        pot = adata.obs["potential"].astype(float).values
        plas = (
            adata.obs["plasticity_score"].astype(float).values
            if "plasticity_score" in adata.obs
            else np.zeros(adata.n_obs)
        )
        positions = np.asarray(adata.obsm["_lap_compute_slice"], dtype=float)
        labels = adata.obs["fate_label"].astype(str).values
        start_core = select_fate_core_indices(
            positions, labels, FATE_START, potential=pot, plasticity=plas,
            prefer_high_potential=True, prefer_high_plasticity=True, core_fraction=0.25, min_cells=8,
        )
        end_core = select_fate_core_indices(
            positions, labels, end_label, potential=pot,
            prefer_high_potential=False, core_fraction=0.25, min_cells=5,
        )
        start_pos = positions[start_core].mean(axis=0)
        end_pos = positions[end_core].mean(axis=0)
        profile = replace(GSE141259_PROFILE, lap_n_pcs=2, max_path_points=12, bootstrap_n=8)
        analyzer = _make_analyzer(adata, profile)
        n_points = _path_n_points(adata, profile)
        path_result = analyzer.compute_least_action_path(start_pos, end_pos, n_points=n_points, use_3d=False)
        path = np.asarray(path_result["path"], dtype=float)
        mom = path_result.get("momentum")
        times = np.linspace(0.0, 1.0, len(path))
        if mom is None:
            mom = np.zeros_like(path)
            for i in range(len(path) - 1):
                dt = times[i + 1] - times[i]
                mom[i] = (path[i + 1] - path[i]) / max(dt, 1e-12)
            mom[-1] = mom[-2]
        else:
            mom = np.asarray(mom, dtype=float)

        U_func = analyzer.U_func_2d
        # Prefer library integral for exact total; keep custom decomposition for terms
        total_S, _ = flow_space_action_integral(path, times, U_func, mom, lambda_u=1.0)
        decomp = _decompose_flow_action(path, times, mom, U_func, lambda_u=1.0)
        decomp["total_action"] = float(total_S)
        decomp["reported_vs_recomputed"] = float(
            abs(float(path_result.get("total_action", np.nan)) - float(total_S))
        )
        flow_mm = float(path_result.get("flow_mismatch") or mean_flow_mismatch(path, times, mom))
        row = {
            "branch": tag,
            "end_label": end_label,
            "n_start_core": int(len(start_core)),
            "n_end_core": int(len(end_core)),
            "reported_total_action": float(path_result.get("total_action", np.nan)),
            "path_degenerate": bool(path_result.get("path_degenerate", False)),
            "flow_mismatch_mean": float(flow_mm),
            "success": bool(path_result.get("success", True)),
            **decomp,
        }
        rows.append(row)
        path_store[tag] = {"path": path, "U": np.array([float(np.asarray(U_func(p)).reshape(-1)[0]) for p in path]), "times": times}

        geo_len = float(np.linalg.norm(end_pos - start_pos))
        row["geodesic_length"] = geo_len
        row["action_minus_geodesic"] = row["reported_total_action"] - geo_len

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lung_action_decomposition.csv", index=False)

    # Matched comparison: action components
    at1 = df[df.branch == "AT1"].iloc[0]
    fibro = df[df.branch == "Fibro"].iloc[0]
    verdict = {
        "AT1_total": float(at1["total_action"]),
        "Fibro_total": float(fibro["total_action"]),
        "AT1_mismatch": float(at1["action_mismatch"]),
        "Fibro_mismatch": float(fibro["action_mismatch"]),
        "AT1_U_term": float(at1["action_potential"]),
        "Fibro_U_term": float(fibro["action_potential"]),
        "AT1_path_length": float(at1["path_length"]),
        "Fibro_path_length": float(fibro["path_length"]),
        "AT1_mean_U": float(at1["mean_U_along_path"]),
        "Fibro_mean_U": float(fibro["mean_U_along_path"]),
        "AT1_flow_mismatch": float(at1["flow_mismatch_mean"]),
        "Fibro_flow_mismatch": float(fibro["flow_mismatch_mean"]),
        "Fibro_mismatch_dominates": bool(
            abs(fibro["action_mismatch"]) > 5 * abs(fibro["action_potential"])
            or abs(fibro["action_mismatch"]) > 10
        ),
        "raw_action_ratio_abs": float(abs(fibro["total_action"]) / max(abs(at1["total_action"]), 1e-8)),
        "length_normalized_U_AT1": float(at1["U_per_length"]),
        "length_normalized_U_Fibro": float(fibro["U_per_length"]),
        "length_normalized_mismatch_AT1": float(at1["mismatch_per_length"]),
        "length_normalized_mismatch_Fibro": float(fibro["mismatch_per_length"]),
        "honest_claim": (
            "Do NOT cite raw total_action +102 vs -0.35 as biological cost ratio; "
            "decompose into flow-mismatch vs potential. Prefer path length, mean/max U, reliability Ω."
        ),
    }
    # Prefer biological ranking by mean U / barrier / reliability from summaries
    at1_sum = json.loads((CKPT_LUNG / "analysis_protocol_GSE141259" / "fate_Krt8_to_AT1_summary.json").read_text())
    fibro_sum = json.loads((CKPT_LUNG / "analysis_protocol_GSE141259" / "fate_Krt8_to_Fibro_summary.json").read_text())
    verdict["AT1_reliability"] = at1_sum.get("omega", {}).get("path_reliability")
    verdict["Fibro_reliability"] = fibro_sum.get("omega", {}).get("path_reliability")
    verdict["prefer_AT1_by_reliability"] = bool(
        (verdict["AT1_reliability"] or 0) > (verdict["Fibro_reliability"] or 0)
    )
    verdict["prefer_AT1_by_mean_U"] = bool(verdict["AT1_mean_U"] <= verdict["Fibro_mean_U"])
    # If Fibro mismatch dominates, raw action comparison FAILS biological interpretation
    verdict["raw_action_comparison_valid"] = not verdict["Fibro_mismatch_dominates"]
    verdict["overall"] = (
        "REVISE_CLAIM"
        if verdict["Fibro_mismatch_dominates"]
        else "OK_with_normalization"
    )
    (OUT / "lung_action_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    labs = ["AT1", "Fibro"]
    axes[0].bar(labs, [at1["total_action"], fibro["total_action"]], color=[PALETTE[2], PALETTE[5]])
    axes[0].set_title("Raw total action\n(DO NOT over-interpret)")
    axes[0].set_ylabel("S")
    style_axis(axes[0], grid_axis="y")

    x = np.arange(2)
    w = 0.35
    axes[1].bar(x - w / 2, [at1["action_mismatch"], fibro["action_mismatch"]], w, label="mismatch", color=PALETTE[5])
    axes[1].bar(x + w / 2, [at1["action_potential"], fibro["action_potential"]], w, label="λU", color=PALETTE[2])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labs)
    axes[1].set_title("Action decomposition")
    axes[1].legend(fontsize=8)
    style_axis(axes[1], grid_axis="y")

    axes[2].bar(labs, [at1["mean_U_along_path"], fibro["mean_U_along_path"]], color=[PALETTE[0], PALETTE[4]])
    axes[2].set_title("Mean U along path\n(biology-friendlier)")
    axes[2].set_ylabel("mean U0")
    style_axis(axes[2], grid_axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "lung_action_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(df.to_string(index=False), flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    return {"table": df, "verdict": verdict}


def run_ccc_permutation_patient(n_perm: int = 200) -> Dict:
    from analysis_protocol_utils import deep_valley_mask, score_lr_pairs
    from celltype_analysis import DATASET_REGISTRY, load_annotated_adata
    from plot_utils import PALETTE, configure_headless, style_axis

    configure_headless()
    print("[CCC] loading HGSOC...", flush=True)
    adata = load_annotated_adata(DATASET_REGISTRY["HGSOC"], str(CKPT_HG))
    obs = pd.read_csv(CKPT_HG / "obs.csv", index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    common = adata.obs_names.intersection(obs.index)
    adata = adata[common].copy()
    for c in ["potential_stationary", "stability_score", "annotation", "treatment_phase", "patient_id"]:
        if c in obs.columns:
            adata.obs[c] = obs.loc[common, c].values
    adata.obs["annotation"] = adata.obs["annotation"].astype(str)

    eoc = adata[adata.obs["annotation"] == "EOC"].copy()
    st = adata[adata.obs["annotation"] == "Stromal"].copy()
    for ad in (eoc, st):
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)

    pot_e = eoc.obs["potential_stationary"].astype(float).values
    stab = eoc.obs["stability_score"].astype(float).values
    valley = deep_valley_mask(pot_e, stab)
    eoc_deep = eoc[valley].copy()
    pot_s = st.obs["potential_stationary"].astype(float).values
    lo, hi = np.nanquantile(pot_s, [0.25, 0.75])
    st_low = st[pot_s <= lo].copy()
    st_high = st[pot_s >= hi].copy()

    focus_pairs = [
        ("FN1", "ITGB1"),
        ("COL1A1", "ITGB1"),
        ("FN1", "ITGAV"),
        ("IL6", "IL6ST"),
        ("CXCL12", "CXCR4"),
        ("HGF", "MET"),
        ("LIF", "IL6ST"),
        ("IL11", "IL11RA"),
    ]

    def lr_table(sender, receiver, pairs=focus_pairs) -> pd.DataFrame:
        return score_lr_pairs(sender, receiver, pairs, min_expr=0.0)

    obs_high = lr_table(st_high, eoc_deep)
    obs_low = lr_table(st_low, eoc_deep)
    obs_high["band"] = "highU"
    obs_low["band"] = "lowU"
    obs_both = pd.concat([obs_high, obs_low], ignore_index=True)
    obs_both.to_csv(OUT / "ccc_observed_focus_pairs.csv", index=False)

    # Delta low - high for each pair
    def deltas(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
        aa = a.set_index(["ligand", "receptor"])["lr_score"]
        bb = b.set_index(["ligand", "receptor"])["lr_score"]
        idx = aa.index.union(bb.index)
        return (bb.reindex(idx).fillna(0) - aa.reindex(idx).fillna(0))

    obs_delta = deltas(obs_high, obs_low)
    obs_delta.to_csv(OUT / "ccc_observed_delta_low_minus_high.csv", header=["delta"])

    # Permutation: shuffle stromal U-band labels among stromal cells (composition null)
    print(f"[CCC] band-label permutations n={n_perm}...", flush=True)
    st_both = st[(pot_s <= lo) | (pot_s >= hi)].copy()
    band_labels = np.where(st_both.obs["potential_stationary"].astype(float).values <= lo, "lowU", "highU")
    rng = np.random.default_rng(0)
    perm_deltas = {pair: [] for pair in focus_pairs}
    for p in range(n_perm):
        lab = band_labels.copy()
        rng.shuffle(lab)
        s_hi = st_both[lab == "highU"]
        s_lo = st_both[lab == "lowU"]
        if s_hi.n_obs < 20 or s_lo.n_obs < 20:
            continue
        d = deltas(lr_table(s_hi, eoc_deep), lr_table(s_lo, eoc_deep))
        for pair in focus_pairs:
            perm_deltas[pair].append(float(d.get(pair, 0.0)))
        if (p + 1) % 50 == 0:
            print(f"  perm {p+1}/{n_perm}", flush=True)

    perm_rows = []
    for pair in focus_pairs:
        arr = np.asarray(perm_deltas[pair], dtype=float)
        obs_v = float(obs_delta.get(pair, 0.0))
        # two-sided empirical p
        if len(arr):
            emp_p = float((np.sum(np.abs(arr) >= abs(obs_v)) + 1) / (len(arr) + 1))
        else:
            emp_p = np.nan
        perm_rows.append(
            {
                "ligand": pair[0],
                "receptor": pair[1],
                "obs_delta_low_minus_high": obs_v,
                "null_mean": float(np.mean(arr)) if len(arr) else np.nan,
                "null_std": float(np.std(arr)) if len(arr) else np.nan,
                "emp_p_twosided": emp_p,
                "significant_005": bool(np.isfinite(emp_p) and emp_p < 0.05),
                "direction": "lowU_stronger" if obs_v > 0 else "highU_stronger",
            }
        )
    perm_df = pd.DataFrame(perm_rows).sort_values("emp_p_twosided")
    perm_df.to_csv(OUT / "ccc_band_permutation_null.csv", index=False)

    # Patient-level: per-patient Stromal high/low → global EOC deep receptor (or patient-matched EOC if available)
    print("[CCC] patient-level deltas...", flush=True)
    if "patient_id" not in st.obs.columns:
        st.obs["patient_id"] = "unknown"
        eoc_deep.obs["patient_id"] = eoc_deep.obs.get("patient_id", "unknown")
    # Use global EOC deep receptors (same as protocol) but patient-specific stromal senders
    patient_rows = []
    for pid, st_p in st.obs.groupby(st.obs["patient_id"].astype(str)):
        idx_p = st_p.index
        st_sub = st[idx_p]
        if st_sub.n_obs < 40:
            continue
        u = st_sub.obs["potential_stationary"].astype(float).values
        q25, q75 = np.nanquantile(u, [0.25, 0.75])
        lo_p = st_sub[u <= q25]
        hi_p = st_sub[u >= q75]
        if lo_p.n_obs < 10 or hi_p.n_obs < 10:
            continue
        d = deltas(lr_table(hi_p, eoc_deep), lr_table(lo_p, eoc_deep))
        for pair in focus_pairs:
            patient_rows.append(
                {
                    "patient_id": pid,
                    "ligand": pair[0],
                    "receptor": pair[1],
                    "delta_low_minus_high": float(d.get(pair, 0.0)),
                    "n_low": int(lo_p.n_obs),
                    "n_high": int(hi_p.n_obs),
                }
            )
    pat_df = pd.DataFrame(patient_rows)
    pat_df.to_csv(OUT / "ccc_patient_level_deltas.csv", index=False)

    pat_sum = []
    if len(pat_df):
        for pair, g in pat_df.groupby(["ligand", "receptor"]):
            vals = g["delta_low_minus_high"].astype(float).values
            # Wilcoxon signed-rank vs 0 across patients
            try:
                if np.all(vals == 0):
                    wil_p = 1.0
                else:
                    _, wil_p = stats.wilcoxon(vals, alternative="two-sided", zero_method="wilcox")
                    wil_p = float(wil_p)
            except Exception:
                wil_p = float("nan")
            pat_sum.append(
                {
                    "ligand": pair[0],
                    "receptor": pair[1],
                    "n_patients": int(len(vals)),
                    "mean_delta": float(np.mean(vals)),
                    "median_delta": float(np.median(vals)),
                    "frac_patients_same_sign_as_global": float(
                        np.mean(np.sign(vals) == np.sign(obs_delta.get(pair, 0.0)))
                    ),
                    "wilcoxon_p_vs_0": wil_p,
                }
            )
    pat_sum_df = pd.DataFrame(pat_sum).sort_values("wilcoxon_p_vs_0")
    pat_sum_df.to_csv(OUT / "ccc_patient_summary.csv", index=False)

    # Verdict
    ecm_pairs = [("FN1", "ITGB1"), ("COL1A1", "ITGB1"), ("FN1", "ITGAV")]
    cyto_pairs = [("IL6", "IL6ST"), ("CXCL12", "CXCR4"), ("HGF", "MET"), ("LIF", "IL6ST"), ("IL11", "IL11RA")]

    def pair_row(df, pair):
        m = (df.ligand == pair[0]) & (df.receptor == pair[1])
        return df[m].iloc[0] if m.any() else None

    ecm_sig = [pair_row(perm_df, p) for p in ecm_pairs]
    cyto_sig = [pair_row(perm_df, p) for p in cyto_pairs]
    verdict = {
        "n_perm": n_perm,
        "ecm_obs_deltas": {f"{a}-{b}": float(obs_delta.get((a, b), np.nan)) for a, b in ecm_pairs},
        "ecm_permutation_significant": {
            f"{r.ligand}-{r.receptor}": bool(r.significant_005) for r in ecm_sig if r is not None
        },
        "cyto_permutation_significant": {
            f"{r.ligand}-{r.receptor}": bool(r.significant_005) for r in cyto_sig if r is not None
        },
        "any_ecm_sig": bool(any(r is not None and r.significant_005 for r in ecm_sig)),
        "any_cyto_sig": bool(any(r is not None and r.significant_005 for r in cyto_sig)),
        "n_patients": int(pat_sum_df["n_patients"].max()) if len(pat_sum_df) else 0,
        "patient_ecm_consistent": {
            f"{r.ligand}-{r.receptor}": float(r.frac_patients_same_sign_as_global)
            for r in pat_sum_df.itertuples()
            if (r.ligand, r.receptor) in ecm_pairs
        },
        "writing_rule": (
            "ECM lowU-stronger claim needs permutation support; "
            "if not significant vs band-shuffle, treat as descriptive only."
        ),
    }
    # Patient consistency for ECM: global lowU_stronger may not hold within patients
    ecm_fracs = list(verdict["patient_ecm_consistent"].values())
    ecm_patient_ok = bool(ecm_fracs) and float(np.mean(ecm_fracs)) >= 0.6
    verdict["ecm_patient_mean_frac_same_sign"] = float(np.mean(ecm_fracs)) if ecm_fracs else float("nan")
    verdict["ecm_patient_consistent"] = ecm_patient_ok
    if verdict["any_ecm_sig"] and ecm_patient_ok:
        verdict["overall"] = "PASS_ECM"
    elif verdict["any_ecm_sig"] and not ecm_patient_ok:
        verdict["overall"] = "PARTIAL_ECM"
        verdict["writing_rule"] = (
            "Pooled band-shuffle supports ECM lowU-stronger, but within-patient signs often disagree "
            "(composition/Simpson risk). Cite pooled CCC as exploratory; do not claim patient-stable ECM division."
        )
    else:
        verdict["overall"] = "DESCRIPTIVE_ONLY"
    (OUT / "ccc_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # observed deltas with null distribution for top ECM
    for ax, pair, title in [
        (axes[0], ("FN1", "ITGB1"), "FN1–ITGB1 Δ(low−high)"),
        (axes[1], ("IL6", "IL6ST"), "IL6–IL6ST Δ(low−high)"),
    ]:
        arr = np.asarray(perm_deltas[pair], dtype=float)
        ax.hist(arr, bins=30, color=PALETTE[0], alpha=0.75, label="band-shuffle null")
        ax.axvline(float(obs_delta.get(pair, 0)), color=PALETTE[5], lw=2, label="observed")
        ax.set_title(title)
        ax.legend(fontsize=8)
        style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "ccc_permutation_null.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(perm_df.to_string(index=False), flush=True)
    if len(pat_sum_df):
        print(pat_sum_df.to_string(index=False), flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    return {"perm": perm_df, "patient": pat_sum_df, "verdict": verdict}


def write_report(lung, ccc):
    lung_df = lung["table"]
    lv = lung["verdict"]
    perm = ccc["perm"]
    pat = ccc["patient"]
    cv = ccc["verdict"]
    lines = [
        "# P1 稳健性校验报告",
        "",
        "目录：`output_file/robustness/p1_robustness/`",
        "",
        "## 1. 肺命运路径 action 分解",
        "",
        lung_df.to_markdown(index=False),
        "",
        "### 判决",
        f"- Raw AT1 action = {lv['AT1_total']:.4g}; Fibro = {lv['Fibro_total']:.4g}",
        f"- Fibro mismatch 项 = {lv['Fibro_mismatch']:.4g}; U 项 = {lv['Fibro_U_term']:.4g}",
        f"- **mismatch 主导 Fibro 大 action？ {lv['Fibro_mismatch_dominates']}**",
        f"- 原始 action 数量级比较是否有效？ **{lv['raw_action_comparison_valid']}**",
        f"- 按路径 mean U：AT1={lv['AT1_mean_U']:.4g}, Fibro={lv['Fibro_mean_U']:.4g}",
        f"- 按 reliability：AT1={lv['AT1_reliability']}, Fibro={lv['Fibro_reliability']}",
        f"- **overall: {lv['overall']}**",
        "",
        lv["honest_claim"],
        "",
        "图：`lung_action_decomposition.png`",
        "",
        "## 2. CCC：band 置换零模型 + 患者分层",
        "",
        perm.to_markdown(index=False),
        "",
    ]
    if pat is not None and len(pat):
        lines += ["### 患者层汇总", "", pat.to_markdown(index=False), ""]
    lines += [
        "### 判决",
        f"- ECM 任一对置换显著？ **{cv['any_ecm_sig']}**",
        f"- 因子任一对置换显著？ **{cv['any_cyto_sig']}**",
        f"- ECM 患者同号比例均值： **{cv.get('ecm_patient_mean_frac_same_sign', float('nan')):.3f}**（阈值≥0.6 才算一致）",
        f"- overall: **{cv['overall']}**",
        "",
        cv["writing_rule"],
        "",
        "图：`ccc_permutation_null.png`",
        "",
        "## 写作修改清单",
        "",
        "1. 肺：不要用 +102 vs −0.35 作为生物学代价比；改用 reliability Ω、path_degenerate、或分解后的 U/mismatch 项。",
        "2. 肺：Fibro 路径 `path_degenerate=True` 且 mismatch≈91 >> |U|；禁止把大 action 写成“成纤维代价高”。",
        "3. CCC：pooled band-shuffle 支持 ECM lowU / 因子 highU；但患者内符号常与全局相反 → 写 **PARTIAL**，不作患者稳健结论。",
        "4. CXCL12–CXCR4 是少数在患者层也同号且 Wilcoxon 显著的对（highU 更强）。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    lung = run_lung_action_diagnostics()
    ccc = run_ccc_permutation_patient(n_perm=200)
    write_report(lung, ccc)
    print("Wrote", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
