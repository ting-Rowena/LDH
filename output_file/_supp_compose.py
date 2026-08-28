#!/usr/bin/env python3
"""Shared PNG compose helpers for output_file supplementary figures."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import numpy as np

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
from PIL import Image


def trim_whitespace(arr: np.ndarray, pad: int = 8) -> np.ndarray:
    ink = (arr < 250).any(axis=2)
    cols = np.where(ink.any(axis=0))[0]
    rows = np.where(ink.any(axis=1))[0]
    if cols.size == 0 or rows.size == 0:
        return arr
    r0 = max(0, int(rows[0]) - pad)
    r1 = min(arr.shape[0] - 1, int(rows[-1]) + pad)
    c0 = max(0, int(cols[0]) - pad)
    c1 = min(arr.shape[1] - 1, int(cols[-1]) + pad)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def load_rgb(path: Path, *, trim: bool = True) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"))
    return trim_whitespace(arr) if trim else arr


def show_panel(ax, img: np.ndarray, letter: str | None = None) -> None:
    ax.imshow(img)
    ax.set_axis_off()
    if letter:
        ax.text(
            0.0,
            1.02,
            letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
            clip_on=False,
        )


def save_fig(fig, out: Path, *, dpi: int = 300) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def compose_side_by_side(
    left: Path,
    right: Path,
    out: Path,
    *,
    target_size: tuple[int, int],
    gap: int = 40,
    left_width: int | None = None,
) -> Path:
    """Scale two RGB panels to a published canvas (height-matched, LANCZOS)."""
    target_w, target_h = int(target_size[0]), int(target_size[1])
    img_l = Image.open(left).convert("RGB")
    img_r = Image.open(right).convert("RGB")
    if left_width is None:
        ar_l = img_l.size[0] / max(1, img_l.size[1])
        ar_r = img_r.size[0] / max(1, img_r.size[1])
        usable = max(1, target_w - gap)
        left_w = int(round(usable * ar_l / (ar_l + ar_r)))
    else:
        left_w = int(left_width)
    left_w = max(1, min(left_w, target_w - gap - 1))
    right_w = max(1, target_w - gap - left_w)
    left_r = img_l.resize((left_w, target_h), Image.Resampling.LANCZOS)
    right_r = img_r.resize((right_w, target_h), Image.Resampling.LANCZOS)
    combo = Image.new("RGB", (target_w, target_h), "white")
    combo.paste(left_r, (0, 0))
    combo.paste(right_r, (left_w + gap, 0))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combo.save(out)
    print(f"wrote {out}", flush=True)
    return out
