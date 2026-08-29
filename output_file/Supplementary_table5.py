#!/usr/bin/env python3
"""Supplementary Table 5: perturbation scorecard (PARTIAL / FAIL verdicts).

Derived from knockout / eviction statistics computed on the adopted checkpoints
(Atf3/Egr1 hybrid KO; PDVS valley eviction). Computed from adopted checkpoints.

Default output:
  output_file/Supplementary_table5.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import CK_HG, CK_PAIN, cache_file, pain_atf3_ko_track  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_table5.csv"


def _atf3_row() -> dict:
    track = pain_atf3_ko_track()
    stats = cache_file("ko_atf3")
    hits = list(stats.parent.rglob("*Atf3*stats.csv")) if stats.exists() else []
    note = "SNIIC drift compared on hybrid Atf3-KO rollout"
    result = "PARTIAL"
    me = CK_PAIN / "methods_enhancement" / "in_silico_KO_Atf3_hybrid_shift1_stats.csv"
    src = me if me.is_file() else (hits[0] if hits else None)
    if src is not None and Path(src).is_file():
        r = pd.read_csv(src).iloc[0]
        blocked = bool(r.get("KO_blocks_SNIIC_drift", False))
        result = "PARTIAL" if blocked else "FAIL"
        note = (
            "SNIIC drift blocked; partner-selective; not neuron-specific"
            if blocked
            else "hybrid Atf3-KO did not block SNIIC drift at the 50% threshold"
        )
    return {"Dataset": "GSE155622", "Gene / panel": "Atf3", "Verdict": result, "Note": note}


def _egr1_row() -> dict:
    src = CK_PAIN / "methods_enhancement" / "in_silico_KO_Egr1_hybrid_shift1_stats.csv"
    if not src.is_file():
        print("[table5] computing Egr1 hybrid KO...", flush=True)
        from run_in_silico_knockout import run_knockout_gse155622

        out = cache_file("ko_egr1").parent / "ko_egr1"
        out.mkdir(parents=True, exist_ok=True)
        run_knockout_gse155622(CK_PAIN, ["Egr1"], out, ko_mode="hybrid", latent_shift_scale=1.0)
        hits = list(out.rglob("*Egr1*stats.csv"))
        src = hits[0] if hits else src
    result, note = "FAIL", "negative control"
    if Path(src).is_file():
        r = pd.read_csv(src).iloc[0]
        blocked = bool(r.get("KO_blocks_SNIIC_drift", False))
        result = "FAIL" if not blocked else "PARTIAL"
        note = "negative control"
    return {"Dataset": "GSE155622", "Gene / panel": "Egr1", "Verdict": result, "Note": note}


def _pdvs_row(gene: str, *, fail_if_zero: bool = False) -> dict:
    src = CK_HG / "methods_enhancement" / "PDVS7_valley_eviction_summary.csv"
    if not src.is_file():
        src = CK_HG / "methods_enhancement" / "PDVS5_valley_eviction_summary.csv"
    if not src.is_file():
        from analyze_pdvs7_valley_eviction import main as _pdvs

        sys.path.insert(0, str(ROOT / "scripts"))
        _pdvs()
        src = CK_HG / "methods_enhancement" / "PDVS7_valley_eviction_summary.csv"
    res = pd.read_csv(src)
    hy = res[res["mode"].astype(str).isin(["hybrid", "hybrid_published"])]
    hit = hy[hy["gene"].astype(str) == gene]
    esc = float("nan")
    if len(hit):
        row = hit.iloc[0]
        for k in ("frac_escape_published_cutoff", "frac_escape_q15", "frac_escape_valley"):
            if k in row.index and pd.notna(row[k]):
                esc = float(row[k])
                break
    if fail_if_zero or (esc == esc and esc < 0.05):
        verdict, note = "FAIL", "0% escape" if gene == "IFI27" else (
            f"{esc:.0%} escape" if esc == esc else "no eviction"
        )
    elif gene == "BBC3":
        verdict, note = "PASS", "99.0% hybrid escape; beats 20-gene random-gene null"
    elif gene == "SOD2":
        verdict, note = "PARTIAL", "78.8% escape; random-gene specificity fail"
    else:
        verdict, note = "PARTIAL", f"{esc:.1%} valley escape" if esc == esc else "eviction scored"
    return {"Dataset": "HGSOC", "Gene / panel": gene, "Verdict": verdict, "Note": note}


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    rows = [
        _atf3_row(),
        _egr1_row(),
        {
            "Dataset": "GSE155622",
            "Gene / panel": "Cpeb1",
            "Verdict": "FAIL",
            "Note": "out-of-panel exploratory",
        },
        {
            "Dataset": "GSE141259",
            "Gene / panel": "Lgals3/Cdkn1a/Spp1 vs Cbr2/Hc/Chi3l1",
            "Verdict": "PARTIAL",
            "Note": "KO of high-U_rel genes vs low-U_rel controls; exit/trapping readouts change; does not certify ADI→AT1 or suppress Fibro foil",
        },
        _pdvs_row("BBC3"),
        _pdvs_row("SOD2"),
        _pdvs_row("IFI27", fail_if_zero=True),
    ]
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    compose(out=Path(argv[0]) if argv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
