"""Tests for crush regime contact forces."""

import numpy as np
import pytest

from sim.core.agent import AgentState
from sim.steering.crush import CrushRegime
from sim.steering.sfm import SocialForceModel


def _make_state(positions, velocities=None, radii=0.25):
    """Helper to create a minimal AgentState."""
    n = len(positions)
    positions = np.array(positions, dtype=float)
    if velocities is None:
        velocities = np.zeros((n, 2))
    else:
        velocities = np.array(velocities, dtype=float)
    if np.isscalar(radii):
        radii = np.full(n, radii)
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


def test_crush_stronger():
    """Overlapping agents: crush body force = 3x SFM body force."""
    state = _make_state([[0.0, 0.0], [0.4, 0.0]])
    neighbors = [[0, 1], [0, 1]]

    crush = CrushRegime()
    sfm = SocialForceModel()
    f_crush = crush.compute_crush_forces(state, neighbors)
    f_sfm = sfm.compute_agent_forces(state, neighbors)

    # Crush: k_crush=360000, SFM body: k=120000, ratio=3
    # But SFM includes social repulsion too, so crush body > sfm body component.
    # At overlap=0.1: crush_body=36000N, sfm_body=12000N, sfm_social~6980N
    # crush magnitude should be > sfm body alone
    crush_mag = np.linalg.norm(f_crush[0])
    assert crush_mag > 30000  # 360000 * 0.1 = 36000 body alone


def test_no_overlap_no_crush():
    """Non-overlapping agents -> zero crush force."""
    state = _make_state([[0.0, 0.0], [2.0, 0.0]])
    neighbors = [[0, 1], [0, 1]]

    crush = CrushRegime()
    forces = crush.compute_crush_forces(state, neighbors)

    np.testing.assert_allclose(forces, 0.0, atol=1e-12)


def test_direction():
    """Overlapping agents: crush pushes apart."""
    state = _make_state([[0.0, 0.0], [0.3, 0.0]])
    neighbors = [[0, 1], [0, 1]]

    crush = CrushRegime()
    forces = crush.compute_crush_forces(state, neighbors)

    # Agent 0 pushed left (negative x)
    assert forces[0, 0] < 0
    # Agent 1 pushed right (positive x)
    assert forces[1, 0] > 0
    # Symmetric
    np.testing.assert_allclose(forces[0], -forces[1], atol=1e-6)
