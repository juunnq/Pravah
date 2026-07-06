"""Density and risk heatmap visualization."""

import numpy as np
import matplotlib.pyplot as plt

from sim.viz.style import set_style, save_figure


def plot_density_heatmap(
    positions: np.ndarray,
    xlim: tuple[float, float] = (0, 15),
    ylim: tuple[float, float] = (0, 10),
    resolution: float = 0.25,
    output_dir: str = "figures/",
    name: str = "density_heatmap",
) -> str:
    """Plot 2D density heatmap using histogram binning.

    Args:
        positions: Agent positions, shape (N, 2).
        xlim: x-axis limits.
        ylim: y-axis limits.
        resolution: Grid cell size (m).
        output_dir: Output directory.
        name: Figure filename.

    Returns:
        Path to saved figure.
    """
    set_style()
    nx = int((xlim[1] - xlim[0]) / resolution)
    ny = int((ylim[1] - ylim[0]) / resolution)

    H, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=[nx, ny], range=[list(xlim), list(ylim)],
    )
    # Convert count to density (count / cell_area)
    cell_area = resolution * resolution
    density = H.T / cell_area

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        density, origin="lower", cmap="YlOrRd",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        aspect="auto",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="Density (ped/m²)")
    fig.tight_layout()
    return save_figure(fig, name, output_dir)


def plot_risk_heatmap(
    positions: np.ndarray,
    risk_values: np.ndarray,
    xlim: tuple[float, float] = (0, 15),
    ylim: tuple[float, float] = (0, 10),
    resolution: float = 0.5,
    output_dir: str = "figures/",
    name: str = "risk_heatmap",
) -> str:
    """Plot risk heatmap (scattered data gridded).

    Args:
        positions: Agent positions (N, 2).
        risk_values: Per-agent risk (N,).
        xlim, ylim: Axis limits.
        resolution: Grid cell size.
        output_dir: Output directory.
        name: Figure filename.

    Returns:
        Path to saved figure.
    """
    set_style()
    from scipy.interpolate import griddata

    nx = int((xlim[1] - xlim[0]) / resolution)
    ny = int((ylim[1] - ylim[0]) / resolution)
    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata(positions, risk_values, (Xi, Yi), method="linear", fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        Zi, origin="lower", cmap="YlOrRd",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        aspect="auto", vmin=0,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="Composite Risk")
    fig.tight_layout()
    return save_figure(fig, name, output_dir)
