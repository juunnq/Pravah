"""Tests for density estimators: grid, Voronoi, KDE."""

import numpy as np
import pytest

from sim.density.grid import GridDensityEstimator
from sim.density.voronoi import VoronoiDensityEstimator
from sim.density.kde import KDEDensityEstimator


def test_voronoi_peaks_higher_than_grid_on_cluster():
    """Voronoi density reveals local peaks that grid density washes out.

    20 agents clustered within 1m radius should give:
    - Grid density (R=2m disk): count / (pi*4) <= 20/12.57 ~ 1.6 ped/m^2
    - Voronoi density: tight cells in cluster, peak well above 5 ped/m^2.
    """
    rng = np.random.Generator(np.random.PCG64(7))
    # 20 agents clustered tightly in 1m radius
    angles = rng.uniform(0, 2 * np.pi, 20)
    radii = rng.uniform(0.0, 1.0, 20)
    positions = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles),
    ])

    grid = GridDensityEstimator(radius=2.0)
    voronoi = VoronoiDensityEstimator()

    grid_densities = grid.estimate(positions)
    vor_densities = voronoi.estimate(positions)

    assert grid_densities.max() < 3.0  # grid smooths density
    assert vor_densities.max() > 5.0   # Voronoi reveals local peaks
    assert vor_densities.max() > grid_densities.max()


def test_grid_uniform():
    """25 agents on a 5x5 grid (spacing 1m), R=2 -> each has ~12 neighbors."""
    xs, ys = np.meshgrid(np.arange(5), np.arange(5))
    positions = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)

    grid = GridDensityEstimator(radius=2.0)
    densities = grid.estimate(positions)

    assert densities.shape == (25,)
    assert not np.any(np.isnan(densities))
    # Center agent at (2,2): neighbors within R=2 are all agents with |dx|<=2 and |dy|<=2
    # That's a 5x5 grid minus corners beyond radius -> about 12 neighbors
    # density = 12 / (pi*4) ~ 0.95
    center = 12  # index of (2,2)
    assert 0.5 < densities[center] < 2.0


def test_grid_empty_neighbors():
    """Single isolated agent -> density = 0."""
    positions = np.array([[50.0, 50.0]])
    grid = GridDensityEstimator(radius=2.0)
    densities = grid.estimate(positions)

    assert densities[0] == 0.0  # only self in neighbor list


def test_voronoi_corners():
    """4 agents at corners of 2x2 box -> each cell ~1m^2 -> rho ~1.0."""
    positions = np.array([
        [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0],
    ])
    domain = np.array([[-.5, -.5], [2.5, -.5], [2.5, 2.5], [-.5, 2.5]])
    vor = VoronoiDensityEstimator(domain=domain)
    densities = vor.estimate(positions)

    assert densities.shape == (4,)
    # Each Voronoi cell in a regular grid is ~(3/2)^2 = 2.25 m^2
    # but clipped to domain: each corner cell ~1.5x1.5 = 2.25
    # So density ~0.44
    for d in densities:
        assert 0.1 < d < 5.0


def test_voronoi_no_nan():
    """50 random agents -> no NaN or inf."""
    rng = np.random.Generator(np.random.PCG64(42))
    positions = rng.uniform(0, 10, (50, 2))
    vor = VoronoiDensityEstimator()
    densities = vor.estimate(positions)

    assert densities.shape == (50,)
    assert not np.any(np.isnan(densities))
    assert not np.any(np.isinf(densities))


def test_kde_cluster_center():
    """Tight cluster -> center has highest density."""
    rng = np.random.Generator(np.random.PCG64(42))
    # Cluster of 20 agents near (5, 5) + 5 outliers
    cluster = rng.normal(5.0, 0.3, (20, 2))
    outliers = rng.uniform(0, 10, (5, 2))
    positions = np.vstack([cluster, outliers])

    kde = KDEDensityEstimator(bandwidth=1.0)
    densities = kde.estimate(positions)

    assert densities.shape == (25,)
    # Cluster agents should have higher density than outliers
    cluster_mean = np.mean(densities[:20])
    outlier_mean = np.mean(densities[20:])
    assert cluster_mean > outlier_mean


def test_all_same_magnitude():
    """Grid, Voronoi, KDE on same layout -> same order of magnitude."""
    rng = np.random.Generator(np.random.PCG64(42))
    positions = rng.uniform(1, 9, (30, 2))

    d_grid = GridDensityEstimator(radius=2.0).estimate(positions)
    d_vor = VoronoiDensityEstimator().estimate(positions)
    d_kde = KDEDensityEstimator(bandwidth=1.0).estimate(positions)

    # All should be positive and within 2 orders of magnitude of each other
    for d in [d_grid, d_vor, d_kde]:
        assert np.all(d >= 0)
    # Mean values in the same ballpark (within factor of 5)
    means = [np.mean(d_grid), np.mean(d_vor), np.mean(d_kde)]
    assert max(means) / max(min(means), 1e-8) < 5
