"""Grid-based (counting) density estimator.

rho(i) = count(neighbors within R, excluding self) / (pi * R^2).
"""

import numpy as np

from sim.density.base import DensityEstimator


class GridDensityEstimator(DensityEstimator):
    """Counting-based local density using KDTree neighbor lists.

    Args:
        radius: Counting radius R (m). Default 2.0.
    """

    def __init__(self, radius: float = 2.0):
        self.radius = radius

    def estimate(
        self,
        positions: np.ndarray,
        neighbor_lists: list[list[int]] | None = None,
    ) -> np.ndarray:
        """Estimate density from neighbor counts.

        Note: self-exclusion assumes self IS in the neighbor list (subtracts 1).
        If passing pre-filtered lists (self already removed), density will be
        underestimated by 1/(pi*R^2). Use None to let the estimator build
        its own KDTree (recommended).

        Args:
            positions: Agent positions, shape (N, 2).
            neighbor_lists: Per-agent neighbor indices including self
                (from KDTree query_ball_point with r=self.radius).
                If None, builds KDTree internally (recommended).

        Returns:
            Per-agent density, shape (N,).
        """
        n = len(positions)
        if n == 0:
            return np.array([])

        if neighbor_lists is None:
            from scipy.spatial import KDTree
            tree = KDTree(positions)
            neighbor_lists = tree.query_ball_point(positions, r=self.radius)

        area = np.pi * self.radius * self.radius
        # Exclude self from count
        densities = np.array(
            [max(0, len(nbs) - 1) / area for nbs in neighbor_lists],
            dtype=float,
        )
        return densities
