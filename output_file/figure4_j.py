#!/usr/bin/env python3
"""Figure 4: exploratory Club 3D paths + contextual AT2→ADI overlay.

Consolidated entrypoint from:
  - scripts/export_mac_alv_3d_landscape_interactive.py  (HTML)
  - scripts/export_3d_landscape_html_to_2d_png.py       (top-view PNG)

Rebuild uses ``build_club_bundle(extra_edges=[("AT2 cells", "Krt8 ADI")])``
so HTML and PNG stay aligned.

Default outputs:
  output_file/figure4_j.html
  output_file/figure4_j_topview.png

Usage:
  python output_file/figure4_j.py                 # recompute field + LAP + export
  python output_file/figure4_j.py --auto-edges
  python output_file/figure4_j.py /path/to/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT_DIR = Path(__file__).resolve().parent

DEFAULT_HTML = OUT_DIR / "figure4_j.html"
DEFAULT_PNG = OUT_DIR / "figure4_j_topview.png"

EXTRA_EDGES = [("AT2 cells", "Krt8 ADI")]

for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _rebuild(
    *,
    html_out: Path,
    png_out: Path,
    auto_edges: bool = False,
) -> tuple[Path, Path]:
    import matplotlib

    if __import__("os").environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    import export_3d_landscape_html_to_2d_png as P2  # noqa: E402
    import export_mac_alv_3d_landscape_interactive as E  # noqa: E402

    mode = "auto-edges" if auto_edges else "curated"
    print(f"===== Building Club + AT2→ADI Landscape ({mode}) =====", flush=True)
    field, paths, wells, saddle = E.build_club_bundle(
        auto_edges=auto_edges,
        extra_edges=EXTRA_EDGES,
    )

    html_out.parent.mkdir(parents=True, exist_ok=True)
    title = (
        "Exploratory Club paths · AT2→ADI shown only as alveolar context"
        + (", auto edges" if auto_edges else ", curated candidates")
    )
    E.export_single(
        field,
        paths,
        wells,
        saddle,
        title=title,
        out=html_out,
        eye=dict(x=1.40, y=-1.40, z=0.88),
    )

    print(f"===== Club+AT2→ADI 2D top-view (auto_edges={auto_edges}) =====", flush=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.9), facecolor="white")
    pcm, _lo, _hi = P2._draw_2d_projection(
        ax,
        field,
        paths,
        wells,
        title="Exploratory Club paths + contextual AT2→ADI overlay",
        show_saddles=False,
    )
    P2._add_cbar(fig, ax, pcm)
    fig.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.12)
    fig.text(
        0.01,
        0.008,
        "AT2→ADI is an alveolar context overlay, not evidence for Club multipotency · red=candidate LAP",
        fontsize=6.2,
        style="italic",
        color=P2.MUTED,
        ha="left",
    )
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {html_out}", flush=True)
    print(f"wrote {png_out}", flush=True)
    return html_out, png_out


def compose(
    *,
    html_out: Path | None = None,
    png_out: Path | None = None,
    auto_edges: bool = False,
) -> tuple[Path, Path]:
    html_out = Path(html_out or DEFAULT_HTML)
    png_out = Path(png_out or DEFAULT_PNG)
    return _rebuild(html_out=html_out, png_out=png_out, auto_edges=auto_edges)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "out_dir",
        nargs="?",
        default=None,
        help="Optional directory for figure4_j.html / figure4_j_topview.png",
    )
    ap.add_argument(
        "--auto-edges",
        action="store_true",
        help="Infer LAP edges from pseudotime/kNN (default: curated)",
    )
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        html_out = out_dir / "figure4_j.html"
        png_out = out_dir / "figure4_j_topview.png"
    else:
        html_out, png_out = DEFAULT_HTML, DEFAULT_PNG

    compose(
        html_out=html_out,
        png_out=png_out,
        auto_edges=args.auto_edges,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
