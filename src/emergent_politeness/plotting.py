"""Shared chart style for the sanity-check figures (needs the dev dependency matplotlib).

Colours follow the dataviz reference palette (light mode): categorical slots in a
fixed order, one blue ramp for magnitudes, recessive grid and axes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LW = 1.6  # ≈ 2 px line

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9.5, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def headroom(ax, factor: float, fixed_top: float | None = None) -> None:
    """Leave space above the data for the legend."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, (fixed_top if fixed_top is not None else hi) * factor)


def end_label(ax, x, y, text, color=INK2, dy=0.0) -> None:
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", fontsize=8.5, color=color)
