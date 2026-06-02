#!/usr/bin/env python3
"""
Shared publication figure style for the manuscript figures (addresses the visual-audit
"one figure style guide" recommendation). Imported by analyze_rrtbc.py and
make_heatmaps_rrtbc.py so every figure uses the same fonts, colorblind-safe palette,
line styles, markers, panel labels, grid, and vector (PDF) export.

Palette: Wong (2011) colorblind-safe set, paired with redundant marker shapes and line
styles so figures remain interpretable in grayscale and under color-vision deficiency.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Wong colorblind-safe palette ----
ORANGE      = "#E69F00"   # Map2-naive
BLUE        = "#0072B2"   # Map2-adapted
BLACK       = "#222222"   # online RRT / no-sharing baseline
VERMILLION  = "#D55E00"   # static-obstacle component / invalid routes
SKYBLUE     = "#56B4E9"
GREEN       = "#009E73"   # valid routes (paired with solid style)
PURPLE      = "#CC79A7"
GRAY        = "#888888"

# ---- semantic method styles (color, marker, linestyle) ----
# Skill-transfer / planning-cost / composition methods:
M_NAIVE   = dict(color=ORANGE, marker="o", ls="-",  label="Hybrid · Map2-naive library")
M_ADAPT   = dict(color=BLUE,   marker="s", ls="-",  label="Hybrid · Map2-adapted library")
M_ONLINE  = dict(color=BLACK,  marker="^", ls="--", label="Online RRT")

# Collision-history-sharing levels (Figs 8-10):
S_NONE = dict(color=BLACK,  marker="o", ls="-",  label="none ($q_{\\mathrm{share}}=0$)")
S_SEL  = dict(color=ORANGE, marker="s", ls="--", label="selective ($q_{\\mathrm{share}}=0.6$)")
S_FULL = dict(color=BLUE,   marker="D", ls="-.", label="full ($q_{\\mathrm{share}}=1.0$)")

# Collision-composition components (stacked bars):
C_STATIC = dict(color=VERMILLION, hatch="//", label="static-obstacle")
C_ROBOT  = dict(color=BLUE,       hatch="",   label="robot--robot")

FLEET_TICKS = [2, 4, 6, 8, 10]   # measured fleet sizes only


def apply():
    """Apply the shared rcParams. Call once at import time in each figure script."""
    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType (editable/clean) rather than Type 3
        "ps.fonttype": 42,
    })


def panel(ax, letter, x=-0.02, y=1.06):
    """Bold top-left panel letter, consistent across all multi-panel figures."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right")


def fleet_xaxis(ax):
    """Discrete fleet-size x-axis: ticks only at measured N (no unsampled 3,5,7,9)."""
    ax.set_xticks(FLEET_TICKS)
    ax.set_xlim(min(FLEET_TICKS) - 0.5, max(FLEET_TICKS) + 0.5)


def save(fig, path_noext):
    """Save as vector PDF (the manuscript references the .pdf)."""
    fig.savefig(path_noext + ".pdf")
    plt.close(fig)
