"""Tests for Social Force Model agent-agent forces."""

import numpy as np
import pytest

from sim.core.agent import AgentState
from sim.steering.sfm import SocialForceModel


def _make_state(positions, velocities=None, radii=0.25):
    """Helper to create a minimal AgentState for testing."""
    n = len(positions)
    positions = np.array(positions, dtype=float)
    if velocities is None:
        velocities = np.zeros((n, 2))
    else:
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


def test_repulsion_direction():
    """Agents at (0,0) and (2,0), r=0.25 -> force on agent 0 has negative x."""
    sfm = SocialForceModel()
    state = _make_state([[0.0, 0.0], [2.0, 0.0]])
    neighbors = [[0, 1], [0, 1]]

    forces = sfm.compute_agent_forces(state, neighbors)

    # Agent 0 is pushed away from agent 1 (to the left)
    assert forces[0, 0] < 0
    # y component should be ~0 (on the same horizontal line)
    assert abs(forces[0, 1]) < 1e-10


def test_no_contact():
    """d=2 > r_sum=0.5 -> only social force, magnitude < 100N."""
    sfm = SocialForceModel()
    state = _make_state([[0.0, 0.0], [2.0, 0.0]])
    neighbors = [[0, 1], [0, 1]]

    forces = sfm.compute_agent_forces(state, neighbors)

    mag = np.linalg.norm(forces[0])
    assert mag > 0  # social repulsion present
    assert mag < 100  # but weak at this distance


def test_contact_activates():
    """Agents at (0,0) and (0.3,0) -> overlap=0.2, total force > 5000N."""
    sfm = SocialForceModel()
    state = _make_state([[0.0, 0.0], [0.3, 0.0]])
    neighbors = [[0, 1], [0, 1]]

    forces = sfm.compute_agent_forces(state, neighbors)

    mag = np.linalg.norm(forces[0])
    assert mag > 5000  # contact forces are very strong


def test_worked_example():
    """Verify against the documented worked example.

    Agents at (0,0) and (0.4,0), r=0.25 each.
    r_ij=0.5, d_ij=0.4, overlap=0.1
    social = A*exp((0.5-0.4)/0.08) = 2000*exp(1.25) ~= 6980N
    body = k*0.1 = 12000N
    Total on agent 0 ~= 18980N pushing left (negative x).
    """
    sfm = SocialForceModel()
    state = _make_state([[0.0, 0.0], [0.4, 0.0]])
    neighbors = [[0, 1], [0, 1]]

    forces = sfm.compute_agent_forces(state, neighbors)

    # Social: 2000*exp(0.1/0.08) = 2000*exp(1.25)
    social_expected = 2000.0 * np.exp(0.1 / 0.08)
    body_expected = 120000.0 * 0.1
    total_expected = social_expected + body_expected

    # Force pushes agent 0 to the left (negative x)
    assert forces[0, 0] < 0
    np.testing.assert_allclose(abs(forces[0, 0]), total_expected, rtol=0.01)


def test_symmetry():
    """f_01 should be approximately -f_10."""
    sfm = SocialForceModel()
    state = _make_state([[0.0, 0.0], [0.6, 0.3]])
    neighbors = [[0, 1], [0, 1]]

    forces = sfm.compute_agent_forces(state, neighbors)

    np.testing.assert_allclose(forces[0], -forces[1], atol=1e-6)


def test_decay():
    """Force decreases with distance: d=0.6 > d=1.0 > d=2.0."""
    sfm = SocialForceModel()

    mags = []
    for dist in [0.6, 1.0, 2.0]:
        state = _make_state([[0.0, 0.0], [dist, 0.0]])
        neighbors = [[0, 1], [0, 1]]
        forces = sfm.compute_agent_forces(state, neighbors)
        mags.append(np.linalg.norm(forces[0]))

    assert mags[0] > mags[1] > mags[2]


def test_friction_direction_and_magnitude():
    """Friction pulls agent 0 toward agent 1's tangential velocity."""
    sfm = SocialForceModel()
    # Overlapping agents: agent 1 moves upward
    state = _make_state(
        [[0.0, 0.0], [0.3, 0.0]],
        velocities=[[0.0, 0.0], [0.0, 1.0]],
    )
    neighbors = [[0, 1], [0, 1]]

    forces = sfm.compute_agent_forces(state, neighbors)

    # n_ij[0,1] = (-1, 0), t_ij[0,1] = (0, -1)
    # dv_ji = v1 - v0 = (0, 1), dot(dv_ji, t) = dot((0,1),(0,-1)) = -1
    # friction on 0 = kappa * overlap * (-1) * (0, -1) = 240000*0.2*1*(0,1) = (0, 48000)
    # Agent 1 moves up -> friction drags agent 0 upward (positive y)
    assert forces[0, 1] > 0, "Friction should pull agent 0 in +y (toward agent 1's motion)"

    # Verify friction magnitude: kappa * overlap * |dv_dot_t| = 240000 * 0.2 * 1 = 48000
    expected_friction_y = 48000.0
    # y-component of force is purely friction (social/body are purely in x)
    np.testing.assert_allclose(forces[0, 1], expected_friction_y, rtol=0.01)


def test_no_neighbors():
    """Agent with no neighbors -> zero force."""
    sfm = SocialForceModel()
    state = _make_state([[5.0, 5.0]])
    neighbors = [[]]

    forces = sfm.compute_agent_forces(state, neighbors)

    np.testing.assert_allclose(forces[0], [0.0, 0.0], atol=1e-12)
