#!/usr/bin/env python3
r"""Supplementary Table 4: landscape metrics for injury-state neurons (GSE155622).

Manuscript counterpart of former main-text Table 3
(“Landscape metrics across injury-induced sensory neuron states (GSE155622)”).

Neuron-only cells (n≈6466). Relative potential ``potential_relative_type``
(U_rel). Expression: ``normalize_total(1e4) + log1p``.

For each SNIIC module / Nav gene:
  - Spearman ρ vs U_rel
  - Deep-valley score = mean score in lowest U_rel quartile
  - Slope score = OLS slope of score ~ U_rel
  - p-value = Spearman p-value

Default mode **assembles**
``output_file/_cache/SuppTable3_GSE155622_SNIIC_Nav_vs_Urel_protocol_metrics.csv``.

Optional ``--rebuild`` reloads the adopted GSE155622 AnnData and recomputes
(also refreshes the protocol metrics CSV).

Default output:
  output_file/Supplementary_table4.csv

Usage:
  python output_file/Supplementary_table4.py
  python output_file/Supplementary_table4.py --rebuild
  python output_file/Supplementary_table4.py /path/to/out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import CK_PAIN, cache_file  # noqa: E402

PROTOCOL_METRICS = cache_file("SuppTable3_GSE155622_SNIIC_Nav_vs_Urel_protocol_metrics.csv")
DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_table4.csv"

FEATURES = [
    {
        "feature": "SNIIC1 Module",
        "genes": ("Atf3", "Gfra3", "Gal"),
        "role": "Atf3 / Gfra3 / Gal (Chronic Injury)",
        "match_prefix": "SNIIC1",
    },
    {
        "feature": "SNIIC2 Module",
        "genes": ("Atf3", "Mrgprd"),
        "role": "Atf3 / Mrgprd (Acute Injury)",
        "match_prefix": "SNIIC2",
    },
    {
        "feature": "SNIIC3 Module",
        "genes": ("Atf3", "S100b", "Gal"),
        "role": "Atf3 / S100b / Gal (Late Injury)",
        "match_prefix": "SNIIC3",
    },
    {
        "feature": "Scn9a (Na_v 1.7)",
        "genes": ("Scn9a",),
        "role": "Nociceptor excitability and pain-signal initiation",
        "match_prefix": "Scn9a",
    },
    {
        "feature": "Scn10a (Na_v 1.8)",
        "genes": ("Scn10a",),
        "role": "Nociceptor action-potential sodium channel",
        "match_prefix": "Scn10a",
    },
    {
        "feature": "Scn11a (Na_v 1.9)",
        "genes": ("Scn11a",),
        "role": "Persistent-current channel linked to inflammatory and persistent pain",
        "match_prefix": "Scn11a",
    },
]


def _fmt_signed(x: float, digits: int = 3) -> str:
    v = float(x)
    if not np.isfinite(v):
        return ""
    s = f"{abs(v):.{digits}f}"
    return f"+{s}" if v > 0 else (f"-{s}" if v < 0 else f"{s}")


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p == 0.0 or p < 1e-300:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3g}"


def _match_row(df: pd.DataFrame, prefix: str) -> pd.Series:
    feat = df["feature"].astype(str)
    hit = df[feat.str.startswith(prefix)]
    if hit.empty:
        raise KeyError(f"No protocol row starting with {prefix!r}")
    return hit.iloc[0]


def assemble_from_protocol(protocol: Path | None = None) -> pd.DataFrame:
    protocol = Path(protocol or PROTOCOL_METRICS)
    if not protocol.is_file():
        raise FileNotFoundError(
            f"Missing {protocol}. Run with --rebuild to recompute from the checkpoint."
        )
    raw = pd.read_csv(protocol)
    rows = []
    for spec in FEATURES:
        r = _match_row(raw, spec["match_prefix"])
        rows.append(
            {
                "Marker Panel / Gene": spec["feature"],
                "Biological Role": spec["role"],
                "Spearman ρ vs U_rel": _fmt_signed(float(r["spearman_rho_vs_U_rel"])),
                "Deep-Valley Score": f"{float(r['deep_valley_score']):.3f}",
                "Slope Score": _fmt_signed(float(r["slope_score"])),
                "p-value": _fmt_p(float(r.get("spearman_pvalue", np.nan))),
            }
        )
    return pd.DataFrame(rows)


def rebuild_protocol_metrics(*, checkpoint: Path | None = None) -> pd.DataFrame:
    """Recompute protocol metrics on Neuron cells; write PROTOCOL_METRICS."""
    import scanpy as sc
    from scipy.stats import linregress, spearmanr

    from analysis_protocol_utils import gene_expression, module_score, resolve_genes
    from celltype_analysis import DATASET_REGISTRY, load_annotated_adata

    ck = Path(checkpoint or CK_PAIN)
    profile = DATASET_REGISTRY["GSE155622"]
    print(f"[rebuild] loading {ck.name} ...", flush=True)
    adata = load_annotated_adata(profile, str(ck))
    col = "celltype" if "celltype" in adata.obs else "annotation"
    neu = adata[adata.obs[col].astype(str) == "Neuron"].copy()
    sc.pp.normalize_total(neu, target_sum=1e4, inplace=True)
    sc.pp.log1p(neu)

    if "potential_relative_type" not in neu.obs:
        raise KeyError("potential_relative_type missing from neuron obs")
    rel = neu.obs["potential_relative_type"].astype(float).to_numpy()
    hi = rel >= np.nanquantile(rel, 0.75)
    lo = rel <= np.nanquantile(rel, 0.25)

    rows = []
    for spec in FEATURES:
        genes = resolve_genes(neu.var_names, spec["genes"])
        if len(spec["genes"]) > 1:
            score = module_score(neu, genes)
            gene_str = ",".join(genes)
        else:
            if not genes:
                raise KeyError(f"Gene not found: {spec['genes']}")
            score = gene_expression(neu, genes[0])
            gene_str = genes[0]

        m = np.isfinite(score) & np.isfinite(rel)
        rho, p_sp = spearmanr(score[m], rel[m])
        lr = linregress(rel[m], score[m])
        rows.append(
            {
                "feature": spec["feature"],
                "genes": gene_str,
                "spearman_rho_vs_U_rel": float(rho),
                "spearman_pvalue": float(p_sp),
                "deep_valley_score": float(np.nanmean(score[lo])),
                "high_U_rel_score": float(np.nanmean(score[hi])),
                "deep_minus_high": float(np.nanmean(score[lo]) - np.nanmean(score[hi])),
                "slope_score": float(lr.slope),
                "slope_se": float(lr.stderr) if lr.stderr is not None else np.nan,
                "slope_pvalue": float(lr.pvalue) if lr.pvalue is not None else np.nan,
                "n_cells": int(neu.n_obs),
                "n_deep_quartile": int(lo.sum()),
                "n_high_quartile": int(hi.sum()),
                "preprocessing": "normalize_total_1e4+log1p",
            }
        )

    out = pd.DataFrame(rows)
    PROTOCOL_METRICS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROTOCOL_METRICS, index=False)
    print(f"wrote {PROTOCOL_METRICS}", flush=True)
    return out


def compose(out: Path | None = None, *, rebuild: bool = True) -> Path:
    out = Path(out or DEFAULT_OUT)
    if rebuild or not PROTOCOL_METRICS.is_file():
        rebuild_protocol_metrics()
    paper = assemble_from_protocol()
    out.parent.mkdir(parents=True, exist_ok=True)
    paper.to_csv(out, index=False)
    print(paper.to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", nargs="?", default=str(DEFAULT_OUT))
    p.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Use cached protocol metrics in output_file/_cache if present",
    )
    args = p.parse_args(argv)
    compose(out=Path(args.out), rebuild=not bool(args.no_rebuild))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
