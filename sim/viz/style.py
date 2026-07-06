"""Shared plot styling: colors, labels, rcParams, save helper."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "C1": "#1b9e77",
    "C2": "#d95f02",
    "C3": "#7570b3",
    "C4": "#e7298a",
}

LABELS = {
    "C1": "SFM only",
    "C2": "SFM + TTC",
    "C3": "SFM + ORCA",
    "C4": "Full hybrid",
}


def set_style():
    """Set publication-quality matplotlib rcParams."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig, name: str, output_dir: str = "figures/"):
    """Save figure as PDF.

    Args:
        fig: Matplotlib figure.
        name: Filename (without extension).
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path
