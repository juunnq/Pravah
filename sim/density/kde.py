"""Kernel Density Estimation for per-agent density.

Uses scipy.stats.gaussian_kde evaluated at agent positions.
"""

import numpy as np
from scipy.stats import gaussian_kde

from sim.density.base import DensityEstimator


class KDEDensityEstimator(DensityEstimator):
    """Gaussian KDE density estimator.

    Args:
        bandwidth: KDE bandwidth parameter. Default 1.0.
    """

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def estimate(self, positions: np.ndarray, **kwargs) -> np.ndarray:
        """Estimate density using Gaussian KDE.

        Args:
            positions: Agent positions, shape (N, 2).

        Returns:
            Per-agent density, shape (N,).
        """
        n = len(positions)
        if n < 2:
            return np.ones(max(n, 0))

        kde = gaussian_kde(positions.T, bw_method=self.bandwidth)
        # gaussian_kde returns probability density (integral≈1); multiply by N
        # to get pedestrian density in ped/m², consistent with grid and Voronoi.
        densities = kde.evaluate(positions.T) * n
        return densities
