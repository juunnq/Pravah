"""Tests for ORCA velocity optimization."""

import numpy as np
import pytest

from sim.core.agent import AgentState
from sim.core.helpers import clamp_speed
from sim.steering.orca import ORCAModel, solve_2d_lp


def _make_state(positions, velocities=None, radii=0.25, goals=None):
    """Helper to create a minimal AgentState for testing."""
    n = len(positions)
    positions = np.array(positions, dtype=float)
    if velocities is None:
        velocities = np.zeros((n, 2))
    else:
        velocities = np.array(velocities, dtype=float)
    if goals is None:
        goals = np.zeros((n, 2))
    else:
        goals = np.array(goals, dtype=float)
    if np.isscalar(radii):
        radii_arr = np.full(n, radii)
    else:
        radii_arr = np.array(radii, dtype=float)
    return AgentState(
        positions=positions,
        velocities=velocities,
        goals=goals,
        radii=radii_arr,
        desired_speeds=np.full(n, 1.34),
        masses=np.full(n, 80.0),
        taus=np.full(n, 0.5),
        active=np.ones(n, dtype=bool),
    )


def test_no_neighbors():
    """No neighbors -> v_orca = v_pref -> force drives toward desired velocity."""
    orca = ORCAModel()
    state = _make_state(
        [[0.0, 0.0]],
        velocities=[[0.0, 0.0]],
        goals=[[10.0, 0.0]],
    )
    neighbors = [[]]

    forces = orca.compute_orca_forces(state, neighbors)

    # F = m*(v_pref - v)/tau_orca = 80*(1.34*[1,0] - [0,0])/0.5 = [214.4, 0]
    expected_fx = 80.0 * 1.34 / 0.5
    np.testing.assert_allclose(forces[0, 0], expected_fx, atol=1.0)
    np.testing.assert_allclose(forces[0, 1], 0.0, atol=1e-6)


def test_head_on_deflection():
    """Two agents head-on -> ORCA deflects them laterally (nonzero y)."""
    orca = ORCAModel()
    state = _make_state(
        [[-3.0, 0.0], [3.0, 0.0]],
        velocities=[[1.0, 0.0], [-1.0, 0.0]],
        goals=[[10.0, 0.0], [-10.0, 0.0]],
    )
    neighbors = [[0, 1], [0, 1]]

    forces = orca.compute_orca_forces(state, neighbors)

    # Both agents should get a lateral (y) deflection
    assert abs(forces[0, 1]) > 0.1, "Agent 0 should deflect laterally"
    assert abs(forces[1, 1]) > 0.1, "Agent 1 should deflect laterally"
    # Forces should be anti-symmetric in a symmetric setup
    np.testing.assert_allclose(forces[0], -forces[1], atol=5.0)


def test_single_halfplane():
    """One constraint -> result satisfies it (dot with normal >= 0)."""
    point = np.array([0.0, 0.0])
    normal = np.array([0.0, 1.0])
    v_pref = np.array([1.0, -1.0])  # violates constraint
    max_speed = 2.0

    result = solve_2d_lp([(point, normal)], v_pref, max_speed)

    # Result must satisfy the half-plane: dot(result - point, normal) >= 0
    assert np.dot(result - point, normal) >= -1e-8
    # y should be >= 0 since normal is [0,1] through origin
    assert result[1] >= -1e-8


def test_collision_case():
    """Two overlapping agents -> ORCA produces non-trivial forces.

    When overlap is severe, the collision half-plane may be infeasible with
    the speed circle. ORCA then keeps the best achievable velocity —
    body-contact separation is handled by SFM, not ORCA.
    """
    orca = ORCAModel()
    # Agents with mild overlap: distance 0.45 < r_sum 0.5
    state = _make_state(
        [[0.0, 0.0], [0.45, 0.0]],
        velocities=[[0.5, 0.0], [-0.5, 0.0]],
        goals=[[10.0, 0.0], [-10.0, 0.0]],
    )
    neighbors = [[0, 1], [0, 1]]

    forces = orca.compute_orca_forces(state, neighbors)

    # Forces should be non-trivial (ORCA adjusts velocity)
    assert np.linalg.norm(forces[0]) > 1.0
    assert np.linalg.norm(forces[1]) > 1.0
    # No NaN
    assert not np.any(np.isnan(forces))


def test_four_agent_crossing():
    """Four agents crossing at 90-degree angles.

    Agents at corners moving toward opposite corners.
    After 300 steps with ORCA+desired, min pairwise distance > 0.3 at all times.
    """
    orca = ORCAModel()
    from scipy.spatial import KDTree
    from sim.steering.desired import compute_desired_force

    positions = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
    ])
    goals = np.array([
        [10.0, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0],
    ])
    state = _make_state(positions, goals=goals)
    state.velocities = np.zeros((4, 2))

    dt = 0.01
    min_dist_ever = float("inf")

    for _ in range(300):
        tree = KDTree(state.positions)
        neighbor_lists = tree.query_ball_point(state.positions, r=5.0)

        # Combined desired + ORCA
        f_desired = compute_desired_force(
            state.positions, state.velocities, state.goals,
            state.desired_speeds, state.masses, state.taus,
        )
        f_orca = orca.compute_orca_forces(state, neighbor_lists)
        forces = f_desired + f_orca

        # Integrate (Euler)
        accel = forces / state.masses[:, None]
        state.velocities = state.velocities + accel * dt
        state.velocities = clamp_speed(state.velocities, 2.0 * state.desired_speeds)
        state.positions = state.positions + state.velocities * dt

        # Track minimum pairwise distance
        for a in range(4):
            for b in range(a + 1, 4):
                d = np.linalg.norm(state.positions[a] - state.positions[b])
                if d < min_dist_ever:
                    min_dist_ever = d

    assert min_dist_ever > 0.3, f"Min pairwise distance {min_dist_ever:.3f} < 0.3"
