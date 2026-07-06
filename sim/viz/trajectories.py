"""Top-down trajectory visualization colored by time."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sim.viz.style import set_style, save_figure


def plot_trajectories(
    positions_log: list[np.ndarray],
    walls: list | None = None,
    output_dir: str = "figures/",
    name: str = "trajectories",
) -> str:
    """Plot agent trajectories colored by timestep.

    Args:
        positions_log: List of (N, 2) position arrays per timestep.
        walls: Optional list of Wall objects for context.
        output_dir: Output directory.
        name: Figure filename.

    Returns:
        Path to saved figure.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    n_steps = len(positions_log)
    n_agents = positions_log[0].shape[0]

    for i in range(n_agents):
        xs = [positions_log[t][i, 0] for t in range(n_steps)]
        ys = [positions_log[t][i, 1] for t in range(n_steps)]
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments, cmap="viridis", linewidth=0.5, alpha=0.7)
        lc.set_array(colors)
        ax.add_collection(lc)

    if walls:
        for w in walls:
            ax.plot([w.start[0], w.end[0]], [w.start[1], w.end[1]], "k-", lw=1.5)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(ax.collections[0] if ax.collections else None, ax=ax, label="Time (normalized)")
    fig.tight_layout()
    return save_figure(fig, name, output_dir)
