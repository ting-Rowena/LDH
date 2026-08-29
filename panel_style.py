"""Shared manuscript panel title style.

All subplot titles in output_file figures should use:
  loc=center, fontweight=bold, fontsize=PANEL_TITLE_SIZE
"""

from __future__ import annotations

import matplotlib.pyplot as plt

PANEL_TITLE_SIZE = 10
PANEL_TITLE_WEIGHT = "bold"
PANEL_TITLE_LOC = "center"
PANEL_TITLE_PAD = 4

# Shared axis / tick / legend sizes for Fig1-style bar panels
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 7.5
ANNOT_SIZE = 7
YGRID_KW = dict(linestyle=":", linewidth=0.6, color="0.85", zorder=0)


def apply_panel_title_rc() -> None:
    """Set matplotlib rcDefaults for axes titles."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Prefer faces with a reliable bold cut for panel titles.
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": PANEL_TITLE_WEIGHT,
            "axes.titlelocation": PANEL_TITLE_LOC,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )


def set_panel_title(ax, title: str, *, pad: float | None = None, color=None, **kwargs) -> None:
    """Apply the unified panel subplot title style."""
    kw = {
        "loc": PANEL_TITLE_LOC,
        "fontweight": PANEL_TITLE_WEIGHT,
        "fontsize": PANEL_TITLE_SIZE,
        "pad": PANEL_TITLE_PAD if pad is None else pad,
    }
    if color is not None:
        kw["color"] = color
    kw.update(kwargs)
    ax.set_title(title, **kw)


def apply_ygrid(ax) -> None:
    """Dotted horizontal grid matching Fig1C/D."""
    ax.yaxis.grid(True, **YGRID_KW)
    ax.set_axisbelow(True)
