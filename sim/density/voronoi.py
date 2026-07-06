"""Voronoi-based density estimator: rho = 1 / cell_area.

Uses scipy.spatial.Voronoi with mirror points for boundary handling
and shapely for clipping cells to the domain polygon.
"""

import numpy as np
from scipy.spatial import Voronoi

from sim.density.base import DensityEstimator


class VoronoiDensityEstimator(DensityEstimator):
    """Voronoi tessellation density: rho_i = 1 / area(cell_i).

    Args:
        domain: Domain polygon as (N_verts, 2) array or None for auto-detect.
        max_density: Cap for zero/tiny-area cells.
    """

    def __init__(
        self,
        domain: np.ndarray | None = None,
        max_density: float = 100.0,
    ):
        self.domain = domain
        self.max_density = max_density

    def estimate(self, positions: np.ndarray, **kwargs) -> np.ndarray:
        """Estimate Voronoi density at each agent position.

        Args:
            positions: Agent positions, shape (N, 2).

        Returns:
            Per-agent density, shape (N,).
        """
        n = len(positions)
        if n == 0:
            return np.array([])
        if n < 4:
            # Fallback to uniform density for very few agents
            from sim.density.grid import GridDensityEstimator
            return GridDensityEstimator().estimate(positions)

        try:
            from shapely.geometry import Polygon
        except ImportError:
            import warnings
            warnings.warn("shapely not installed — Voronoi falling back to grid density")
            from sim.density.grid import GridDensityEstimator
            return GridDensityEstimator().estimate(positions)

        # Build domain polygon
        if self.domain is not None:
            domain_poly = Polygon(self.domain)
        else:
            # Auto-detect bounding box with margin
            margin = 1.0
            x_min, y_min = positions.min(axis=0) - margin
            x_max, y_max = positions.max(axis=0) + margin
            domain_poly = Polygon([
                (x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max),
            ])

        # Add mirror points far outside to handle boundary cells
        cx, cy = domain_poly.centroid.x, domain_poly.centroid.y
        x_range = float(positions[:, 0].max() - positions[:, 0].min())
        y_range = float(positions[:, 1].max() - positions[:, 1].min())
        spread = max(x_range, y_range, 10.0) * 3.0
        mirror = np.array([
            [cx - spread, cy],
            [cx + spread, cy],
            [cx, cy - spread],
            [cx, cy + spread],
        ])
        all_pts = np.vstack([positions, mirror])

        try:
            vor = Voronoi(all_pts)
        except Exception:
            from sim.density.grid import GridDensityEstimator
            return GridDensityEstimator().estimate(positions)

        densities = np.full(n, self.max_density)

        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]

            if -1 in region or len(region) == 0:
                continue  # infinite region — keep max_density

            verts = vor.vertices[region]
            try:
                cell = Polygon(verts)
                if not cell.is_valid:
                    cell = cell.buffer(0)
                clipped = cell.intersection(domain_poly)
                area = clipped.area
            except Exception:
                continue

            if area > 1e-8:
                densities[i] = min(1.0 / area, self.max_density)

        return densities
