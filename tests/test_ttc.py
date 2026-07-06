"""Tests for Time-to-Collision (TTC) force model."""

import numpy as np
import pytest

from sim.core.agent import AgentState
from sim.steering.ttc import TTCForceModel


def _make_state(positions, velocities, radii=0.25):
    """Helper to create a minimal AgentState for testing."""
    n = len(positions)
    positions = np.array(positions, dtype=float)
    velocities = np.array(velocities, dtype=float)
    if np.isscalar(radii):
        radii = np.full(n, radii)
    else:
        radii = np.array(radii, dtype=float)
    return AgentState(
        positions=positions,
        velocities=velocities,
        goals=np.zeros((n, 2)),
        radii=radii,
        desired_speeds=np.full(n, 1.34),
        masses=np.full(n, 80.0),
        taus=np.full(n, 0.5),
        active=np.ones(n, dtype=bool),
    )


def test_analytical_tau():
    """Agents at (-1,0)->(1,0) and (1,0)->(-1,0), r=0.25 each. Expected tau=0.75s.

    Derivation from the model spec:
        dx = (1,0) - (-1,0) = (2,0)
        dv = (1,0) - (-1,0) = (2,0)
        r = 0.25 + 0.25 = 0.5
        a = dot(dv,dv) = 4
        b = -dot(dx,dv) = -4
        c = dot(dx,dx) - r^2 = 4 - 0.25 = 3.75
        disc = 16 - 4*3.75 = 16 - 15 = 1.0
        tau = (-(-4) - 1.0) / 4 = 3/4 = 0.75

    Physical check: 2m gap, 2m/s closing, collide when gap=0.5m -> (2-0.5)/2 = 0.75s
    """
    ttc = TTCForceModel()
    state = _make_state(
        [[-1.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
    )
    neighbors = [[0, 1], [0, 1]]

    # Manually verify tau computation
    dx = state.positions[1] - state.positions[0]  # (2, 0)
    dv = state.velocities[0] - state.velocities[1]  # (2, 0)
    r = 0.5
    a = np.dot(dv, dv)
    assert abs(a - 4.0) < 1e-10
    b = -np.dot(dx, dv)
    assert abs(b - (-4.0)) < 1e-10
    c = np.dot(dx, dx) - r * r
    assert abs(c - 3.75) < 1e-10
    disc = b * b - a * c
    assert abs(disc - 1.0) < 1e-10
    tau = (-b - np.sqrt(disc)) / a
    assert abs(tau - 0.75) < 1e-6

    # Verify force magnitude: F = k_ttc * exp(-tau/tau_0) / tau^2
    forces = ttc.compute_ttc_forces(state, neighbors)
    expected_mag = 1.5 * np.exp(-0.75 / 3.0) / (0.75 ** 2)
    np.testing.assert_allclose(np.linalg.norm(forces[0]), expected_mag, rtol=0.01)
    # Direction: agent 0 pushed left (-x)
    assert forces[0, 0] < 0
    np.testing.assert_allclose(forces[0, 1], 0.0, atol=1e-10)


def test_parallel_no_collision():
    """Two agents moving in the same direction at the same speed -> no force."""
    ttc = TTCForceModel()
    state = _make_state(
        [[0.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    neighbors = [[0, 1], [0, 1]]

    forces = ttc.compute_ttc_forces(state, neighbors)

    # No relative velocity -> a < 1e-8 -> skip
    np.testing.assert_allclose(forces, 0.0, atol=1e-12)


def test_moving_apart():
    """Diverging agents -> tau <= 0 -> no force."""
    ttc = TTCForceModel()
    state = _make_state(
        [[-1.0, 0.0], [1.0, 0.0]],
        [[-1.0, 0.0], [1.0, 0.0]],  # moving apart
    )
    neighbors = [[0, 1], [0, 1]]

    forces = ttc.compute_ttc_forces(state, neighbors)

    np.testing.assert_allclose(forces, 0.0, atol=1e-12)


def test_beyond_horizon():
    """Collision at tau >> tau_max -> no force."""
    ttc = TTCForceModel(tau_max=8.0)
    # Very slow approach: agents 20m apart, moving toward each other at 0.1 m/s
    state = _make_state(
        [[-10.0, 0.0], [10.0, 0.0]],
        [[0.1, 0.0], [-0.1, 0.0]],
    )
    neighbors = [[0, 1], [0, 1]]

    # tau ~= (20 - 0.5) / 0.2 = 97.5s >> 8.0
    forces = ttc.compute_ttc_forces(state, neighbors)

    np.testing.assert_allclose(forces, 0.0, atol=1e-12)


def test_force_direction():
    """Head-on collision -> forces push agents apart (agent 0 left, agent 1 right)."""
    ttc = TTCForceModel()
    state = _make_state(
        [[-1.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
    )
    neighbors = [[0, 1], [0, 1]]

    forces = ttc.compute_ttc_forces(state, neighbors)

    # Agent 0 should be pushed left (negative x)
    assert forces[0, 0] < 0
    # Agent 1 should be pushed right (positive x)
    assert forces[1, 0] > 0
    # y components should be ~0
    assert abs(forces[0, 1]) < 1e-10
    assert abs(forces[1, 1]) < 1e-10


def test_inverse_square():
    """Force at tau=0.5 should be ~4x force at tau=1.0 (inverse square in tau).

    F ~ exp(-tau/tau_0) / tau^2
    F(0.5) / F(1.0) = [exp(-0.5/3) / 0.25] / [exp(-1/3) / 1.0]
                     = 4 * exp(0.5/3) ~= 4 * 1.181 ~= 4.72
    """
    ttc = TTCForceModel()

    # Setup for tau ~= 0.5: agents 1.5m apart, 2 m/s closing, r=0.5
    # tau = (1.5 - 0.5) / 2 = 0.5
    state_close = _make_state(
        [[-0.75, 0.0], [0.75, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
    )

    # Setup for tau ~= 1.0: agents 2.5m apart, 2 m/s closing, r=0.5
    # tau = (2.5 - 0.5) / 2 = 1.0
    state_far = _make_state(
        [[-1.25, 0.0], [1.25, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
    )

    neighbors = [[0, 1], [0, 1]]
    f_close = ttc.compute_ttc_forces(state_close, neighbors)
    f_far = ttc.compute_ttc_forces(state_far, neighbors)

    mag_close = np.linalg.norm(f_close[0])
    mag_far = np.linalg.norm(f_far[0])

    ratio = mag_close / mag_far
    expected_ratio = 4.0 * np.exp(0.5 / 3.0)  # ~4.72
    np.testing.assert_allclose(ratio, expected_ratio, rtol=0.05)


def test_multi_agent_accumulation():
    """Center agent hit from both sides: symmetric forces cancel to zero."""
    ttc = TTCForceModel()
    state = _make_state(
        [[0.0, 0.0], [-2.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
    )
    neighbors = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    forces = ttc.compute_ttc_forces(state, neighbors)

    # Agent 0 gets equal and opposite forces from agents 1 and 2
    np.testing.assert_allclose(forces[0], [0.0, 0.0], atol=1e-10)
