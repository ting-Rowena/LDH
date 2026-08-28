#!/usr/bin/env python
"""PDVS Top7 analysis: DEG context + deep-valley KO eviction for panel genes.

ZC3H12A and IRF1 are outside the HGSOC HVG training panel → encoder KO is not
identifiable (would require injecting into a foreign weight slot). Those two are
reported from DEG/biology/PDVS only; eviction is run for the 5 in-panel genes.

Outputs:
  methods_enhancement/PDVS7_valley_eviction_summary.csv
  methods_enhancement/PDVS7_gene_card.csv
  methods_enhancement/figures/PDVS7_valley_eviction_compare.png
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from panel_style import apply_panel_title_rc, set_panel_title, PANEL_TITLE_SIZE
import json
import shutil
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy import stats

from analysis_protocol_utils import resolve_genes
from dataset_pipeline import PROJECT_ROOT, HGSOC, apply_train_config, resolve_data_path
from methods_model_utils import _load_state_dict_compat
from plot_utils import configure_headless
from train_model import TemporalSDENetwork, ensure_cell_type_codes

configure_headless()
apply_panel_title_rc()
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.titlelocation": "center",
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)

CK = PROJECT_ROOT / (
    "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
OUT = CK / "methods_enhancement"
PANELS = OUT / "figures"
TABLES = OUT

PDVS = ["BBC3", "SOD2", "ZC3H12A", "WFDC2", "FTL", "CEBPD", "IRF1"]

BIO = {
    "BBC3": ("PUMA", "p53-inducible pro-apoptotic BH3-only", "apoptosis / stress"),
    "SOD2": ("MnSOD", "Mitochondrial superoxide dismutase", "oxidative stress / chemo tolerance"),
    "ZC3H12A": ("Regnase-1", "RNase that destabilizes inflammatory mRNAs", "inflammation / NF-κB control"),
    "WFDC2": ("HE4", "WAP-domain secretory marker (OV clinical)", "epithelial tumour marker"),
    "FTL": ("Ferritin L", "Iron storage; redox metabolism", "iron / redox stress"),
    "CEBPD": ("C/EBPδ", "Stress / acute-phase transcription factor", "TNF / inflammatory TF"),
    "IRF1": ("IRF1", "Interferon regulatory factor 1", "IFN / innate immunity"),
}

INK, UP, DOWN, MUTED, GREY = "#111111", "#9c3d2e", "#2f5f8a", "#555555", "#9a9a9a"


def _unit(v):
    n = np.linalg.norm(v)
    return v * 0.0 if (not np.isfinite(n) or n < 1e-12) else v / n


def _encode(model, X, ct, device, bs=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            x = torch.tensor(X[s : s + bs], dtype=torch.float32, device=device)
            c = torch.tensor(ct[s : s + bs], dtype=torch.long, device=device)
            outs.append(model.encode(x, c).cpu().numpy())
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


def load_eoc_and_model():
    print("[1/4] obs + panel ...", flush=True)
    obs = pd.read_csv(CK / "obs.csv", low_memory=False)
    if "cell" in obs.columns:
        obs = obs.set_index(obs["cell"].astype(str), drop=False)
    else:
        obs.index = obs.index.astype(str)
    gene_list = json.loads((CK / "training_var_names.json").read_text(encoding="utf-8"))
    in_panel = {g: g in gene_list for g in PDVS}

    eoc = obs[obs["annotation"].astype(str) == "EOC"].copy()
    valley_bc = set(
        pd.read_csv(CK / "analysis_protocol_HGSOC" / "eoc_attractor_deep_valley_barcodes.csv")["barcode"].astype(str)
    )
    eoc_barcodes = eoc.index.astype(str).tolist()
    print(f"  EOC={len(eoc_barcodes)} valley_barcodes={len(valley_bc)}", flush=True)

    h5ad = resolve_data_path(HGSOC)
    print(f"[2/4] backed read {h5ad} ...", flush=True)
    raw = ad.read_h5ad(h5ad, backed="r")
    name_to_i = {b: i for i, b in enumerate(raw.obs_names.astype(str))}
    idx = [name_to_i[b] for b in eoc_barcodes if b in name_to_i]
    keep = [b for b in eoc_barcodes if b in name_to_i]
    present = [g for g in gene_list if g in set(map(str, raw.var_names))]
    # h5py backed AnnData allows only one fancy index at a time.
    sub = raw[idx].to_memory()
    raw.file.close()
    present = [g for g in present if g in set(map(str, sub.var_names))]
    sub = sub[:, present].copy()

    missing = [g for g in gene_list if g not in sub.var_names]
    if missing:
        zeros = sp.csr_matrix((sub.n_obs, len(missing)), dtype=np.float32)
        filler = ad.AnnData(X=zeros, obs=sub.obs.copy(), var=pd.DataFrame(index=pd.Index(missing)))
        sub = ad.concat([sub, filler], axis=1, join="outer", merge="same")
    sub = sub[:, list(gene_list)].copy()
    sub.obs_names = pd.Index(keep)
    sub.obs["annotation"] = "EOC"
    sub.obs["cell_type"] = eoc.loc[keep, "cell_type"].astype(int).values
    pot_key = "potential_stationary" if "potential_stationary" in eoc.columns else "potential"
    sub.obs["U0_ckpt"] = eoc.loc[keep, pot_key].astype(float).values
    sub.obs["deep_valley"] = sub.obs_names.astype(str).isin(valley_bc)

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
    n_types = int(obs["cell_type"].max()) + 1
    stub = ad.AnnData(X=np.zeros((n_types, len(gene_list)), dtype=np.float32))
    stub.var_names = pd.Index(gene_list)
    stub.obs["cell_type"] = np.arange(n_types, dtype=int)
    model = TemporalSDENetwork(config, stub)
    state = torch.load(CK / "best_model.pth", map_location="cpu")
    _load_state_dict_compat(model, state)
    model = model.to("cpu").eval()

    X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X, dtype=float)
    ct = sub.obs["cell_type"].astype(int).values
    valley = sub.obs["deep_valley"].to_numpy(bool)
    print("[3b/4] encode all EOC → model U0 + q15 cutoff ...", flush=True)
    z_all = _encode(model, X, ct, "cpu", bs=int(getattr(config, "batch_size", 256)))
    pot = _U(model, z_all, "cpu", bs=int(getattr(config, "batch_size", 256)))
    u_cut = float(np.nanquantile(pot, 0.15))
    # protocol reference cutoff from published SOD2 run (for reporting)
    pub_cut = -0.0220824908465147
    print(
        f"  matrix={X.shape} valley={int(valley.sum())} model_q15={u_cut:.6f} "
        f"published_q15={pub_cut:.6f} valley_mean_U={float(np.mean(pot[valley])):.6f}",
        flush=True,
    )
    return model, config, X, ct, valley, pot, u_cut, gene_list, in_panel, pub_cut


def eval_gene(model, X, ct, valley, pot, u_cut, gene_col, gene, mode, bs):
    X_kd = X.copy()
    X_kd[:, gene_col] = 0.0
    z_wt = _encode(model, X[valley], ct[valley], "cpu", bs=bs)
    z_kd = _encode(model, X_kd[valley], ct[valley], "cpu", bs=bs)
    if mode == "hybrid":
        z_kd = z_wt + _unit((z_kd - z_wt).mean(0))[None, :]
    u_wt = pot[valley]  # model WT U0 (consistent scale)
    u_kd = _U(model, z_kd, "cpu", bs=bs)
    du = u_kd - u_wt
    mean_du = float(np.nanmean(du))
    frac = float(np.mean(u_kd > u_cut))
    # also report escape vs published cutoff for comparability with Fig4B SOD2
    pub_cut = -0.0220824908465147
    frac_pub = float(np.mean(u_kd > pub_cut))
    try:
        _, wil_p = stats.wilcoxon(u_kd, u_wt, alternative="greater", zero_method="wilcox")
        wil_p = float(wil_p)
    except Exception:
        wil_p = float("nan")
    boot_p = _boot_gt0(du, seed=abs(hash(gene + mode)) % (2**31))
    return {
        "gene": gene,
        "mode": mode,
        "n_valley": int(valley.sum()),
        "valley_U_cutoff_q15": u_cut,
        "frac_escape_q15": frac,
        "frac_escape_published_cutoff": frac_pub,
        "mean_U_WT": float(np.nanmean(u_wt)),
        "mean_U_pert": float(np.nanmean(u_kd)),
        "mean_delta_U": mean_du,
        "raises_potential": bool(mean_du > 0),
        "wilcoxon_U_increase_p": wil_p,
        "bootstrap_mean_dU_gt0_p": boot_p,
        "eviction_sig_0.05": bool((np.isfinite(wil_p) and wil_p < 0.05) or (boot_p < 0.05)),
    }


def draw(res, cards, out):
    sub = res[res["mode"].isin(["reencode_only", "hybrid"])].copy()
    genes_ev = [g for g in PDVS if g in set(sub["gene"])]
    hy = sub[sub["mode"] == "hybrid"].set_index("gene").reindex(genes_ev)
    re = sub[sub["mode"] == "reencode_only"].set_index("gene").reindex(genes_ev)
    # plot claim escape (published cutoff) when available
    y_re = re["frac_escape_published_cutoff"] if "frac_escape_published_cutoff" in re else re["frac_escape_q15"]
    y_hy = hy["frac_escape_published_cutoff"] if "frac_escape_published_cutoff" in hy else hy["frac_escape_q15"]
    x = np.arange(len(genes_ev))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax = axes[0]
    ax.bar(x - w / 2, y_re, w, color="#6a8fa0", label="reencode only", edgecolor="none")
    ax.bar(x + w / 2, y_hy, w, color=UP, label="hybrid", edgecolor="none")
    ax.axhline(0.788, color=MUTED, ls=":", lw=0.9, label="SOD2 protocol 78.8%")
    null_p = PROJECT_ROOT / "output_file/robustness/p0_robustness/SOD2_random_gene_nulls.csv"
    if null_p.is_file():
        med = float(pd.read_csv(null_p)["frac_escape_q15"].median())
        ax.axhline(med, color=GREY, ls="--", lw=0.8, label=f"random-null median={med:.2f}")
    ax.set_xticks(x, genes_ev, rotation=25, ha="right")
    ax.set_ylabel("Escape fraction (pub. q15 cut)")
    ax.set_ylim(0, 1.05)
    set_panel_title(ax, "In-panel PDVS genes: valley escape")
    ax.legend(fontsize=6.5, frameon=False)
    ax.yaxis.grid(True, color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    colors = [UP if (g in hy.index and hy.loc[g, "mean_delta_U"] > 0) else (DOWN if g in hy.index else GREY) for g in PDVS]
    vals = []
    for g in PDVS:
        if g in hy.index:
            vals.append(float(hy.loc[g, "mean_delta_U"]))
        else:
            vals.append(0.0)
    ax.bar(np.arange(7), vals, color=colors, edgecolor="none", width=0.72)
    ax.axhline(0, color=INK, lw=0.6)
    for i, g in enumerate(PDVS):
        if g not in hy.index:
            ax.text(i, 0.0, "out of\npanel", ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks(np.arange(7), PDVS, rotation=25, ha="right")
    ax.set_ylabel(r"Mean $\Delta U_0$ (hybrid)")
    set_panel_title(ax, "Potential shift (all PDVS; grey = no KO)")
    ax.yaxis.grid(True, color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)
    fig.suptitle("PDVS Top7: deep-valley programme & KO eviction", fontsize=10, fontweight="bold", y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("Saved", out, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    deg = pd.read_csv(CK / "analysis_protocol_HGSOC" / "eoc_attractor_deep_valley_DEG.csv")
    model, config, X, ct, valley, pot, u_cut, gene_list, in_panel, pub_cut = load_eoc_and_model()
    bs = int(getattr(config, "batch_size", 256))

    # gene cards for all 7
    cards = []
    for rank, g in enumerate(PDVS, 1):
        alias, role, axis = BIO[g]
        m = deg.loc[deg.gene == g]
        card = {
            "gene": g,
            "pdvs_rank": rank,
            "alias": alias,
            "role": role,
            "programme_axis": axis,
            "in_HVG_training_panel": bool(in_panel[g]),
            "ko_eviction_feasible": bool(in_panel[g]),
        }
        if len(m):
            r = m.iloc[0]
            card.update(
                {
                    "deg_rank_in_file": int(m.index[0]) + 1,
                    "deg_score": float(r["score"]),
                    "deg_logFC": float(r["logFC"]) if "logFC" in r else float(r["logfoldchange"]),
                    "deg_padj": float(r["pval_adj"]),
                }
            )
        cards.append(card)
    cards_df = pd.DataFrame(cards)
    cards_df.to_csv(TABLES / "PDVS7_gene_card.csv", index=False)
    cards_df.to_csv(OUT / "PDVS7_gene_card.csv", index=False)

    print("[4/4] eviction for in-panel genes ...", flush=True)
    rows = []
    null = None
    null_p = PROJECT_ROOT / "output_file/robustness/p0_robustness/SOD2_random_gene_nulls.csv"
    if null_p.is_file():
        null = pd.read_csv(null_p)["frac_escape_q15"].astype(float).values

    for g in PDVS:
        if not in_panel[g]:
            print(f"  skip KO {g} (outside HVG panel)", flush=True)
            continue
        rr = resolve_genes(gene_list, [g])
        col = gene_list.index(rr[0])
        for mode in ("reencode_only", "hybrid"):
            print(f"  {g} {mode}", flush=True)
            r = eval_gene(model, X, ct, valley, pot, u_cut, col, g, mode, bs)
            r["in_original_HVG_panel"] = True
            # Prefer published-scale cutoff for claims (matches Fig4B SOD2 ~78.8%)
            esc = r["frac_escape_published_cutoff"]
            r["frac_escape_claim"] = esc
            if null is not None and mode == "hybrid":
                emp = float(np.mean(null >= esc))
                r["emp_p_vs_random_null_q15"] = emp
                r["beats_random_null_p0.1"] = bool(emp <= 0.1)
            if esc >= 0.5 and r["mean_delta_U"] > 0:
                pat = "ESCAPE_HIGH"
            elif esc >= 0.2 and r["mean_delta_U"] > 0:
                pat = "ESCAPE_MODERATE"
            elif r["mean_delta_U"] <= 0 or esc < 0.05:
                pat = "NO_ESCAPE"
            else:
                pat = "WEAK"
            r["pattern"] = pat
            if pat.startswith("ESCAPE") and not r.get("beats_random_null_p0.1", False):
                r["claim"] = "PARTIAL"
            elif pat.startswith("ESCAPE"):
                r["claim"] = "ESCAPE_BEATS_NULL"
            else:
                r["claim"] = "NEGATIVE"
            rows.append(r)
            print(
                f"    esc_q15={esc:.3f} esc_pubCut={r['frac_escape_published_cutoff']:.3f} "
                f"dU={r['mean_delta_U']:.5f} {r['claim']}",
                flush=True,
            )

    # attach published SOD2/IFI27 hybrid rows for reference
    for g, note in [("SOD2", "published_methods_enhancement"), ("IFI27", "published_negative_control")]:
        p = OUT / f"in_silico_KO_{g}_hybrid_shift1_valley_eviction.csv"
        if p.is_file():
            pub = pd.read_csv(p).iloc[0].to_dict()
            rows.append(
                {
                    "gene": g,
                    "mode": "hybrid_published",
                    "n_valley": pub.get("n_deep_valley_cells"),
                    "valley_U_cutoff_q15": pub.get("valley_U_cutoff"),
                    "mean_delta_U": pub.get("mean_delta_U"),
                    "frac_escape_q15": pub.get("frac_escape_valley"),
                    "raises_potential": pub.get("raises_potential"),
                    "bootstrap_mean_dU_gt0_p": pub.get("bootstrap_mean_dU_gt0_p"),
                    "pattern": "PUBLISHED",
                    "claim": "REFERENCE",
                    "note": note,
                }
            )

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "PDVS7_valley_eviction_summary.csv", index=False)
    draw(res[res["mode"].isin(["reencode_only", "hybrid"])], cards_df, PANELS / "PDVS7_valley_eviction_compare.png")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
