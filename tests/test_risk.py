"""Tests for composite risk metric."""

import numpy as np
import pytest

from sim.density.risk import CompositeRiskMetric


def test_high_density_high_variance():
    """High density + high speed variance -> R > 2."""
    risk = CompositeRiskMetric()
    n = 10
    positions = np.random.default_rng(42).uniform(0, 1, (n, 2))
    # High speed variance: some fast, some slow
    velocities = np.zeros((n, 2))
    velocities[:5, 0] = 3.0
    velocities[5:, 0] = 0.1
    rho_V = np.full(n, 8.0)   # high density
    rho_KDE = np.full(n, 7.0)
    neighbors = [list(range(n))] * n

    R = risk.compute(positions, velocities, rho_V, rho_KDE, neighbors)

    assert R.shape == (n,)
    assert np.mean(R) > 2.0


def test_low_density():
    """Low density -> R < 1."""
    risk = CompositeRiskMetric()
    n = 5
    positions = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], dtype=float)
    velocities = np.full((n, 2), 1.34)
    rho_V = np.full(n, 0.5)
    rho_KDE = np.full(n, 0.3)
    neighbors = [[i] for i in range(n)]  # only self

    R = risk.compute(positions, velocities, rho_V, rho_KDE, neighbors)

    assert np.all(R < 1.0)


def test_zero_variance():
    """All agents same speed -> sigma_v=0 -> P=0."""
    risk = CompositeRiskMetric()
    n = 10
    positions = np.random.default_rng(42).uniform(0, 5, (n, 2))
    velocities = np.full((n, 2), 1.34)  # all same speed
    rho_V = np.full(n, 3.0)
    rho_KDE = np.full(n, 3.0)
    neighbors = [list(range(n))] * n

    R = risk.compute(positions, velocities, rho_V, rho_KDE, neighbors)

    # With P=0 and grad small: R ~ rho_hat/rho_ref * (1 + small)
    # = 3/6 * ~1 = ~0.5
    assert np.all(R < 1.5)
    assert np.all(R > 0.0)
