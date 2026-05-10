# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/submission_figure_style.py
# Renamed package path: code/analysis_support/submission_figure_style.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

"""Shared style helpers for the PNAS Nexus submission figure rebuilds."""

from __future__ import annotations

import matplotlib.pyplot as plt


EXPORT_DPI = 600
FONT = {
    "base": 8.6,
    "axis": 8.8,
    "tick": 7.6,
    "legend": 7.4,
    "title": 9.6,
    "panel": 10.8,
    "annotation": 7.2,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": EXPORT_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONT["base"],
            "axes.titlesize": FONT["title"],
            "axes.labelsize": FONT["axis"],
            "legend.fontsize": FONT["legend"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(ax, label: str, *, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        str(label).strip("()").upper(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT["panel"],
        fontweight="bold",
        color="0.05",
        clip_on=False,
    )
