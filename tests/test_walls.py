"""Tests for wall forces."""

import numpy as np
import pytest

from sim.core.agent import AgentState
from sim.core.world import Wall
from sim.steering.walls import WallForces


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


# --- corridor walls for convenience ---
_bottom = Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0]))
_right = Wall(np.array([10.0, 0.0]), np.array([10.0, 3.6]))
_top = Wall(np.array([10.0, 3.6]), np.array([0.0, 3.6]))
_left = Wall(np.array([0.0, 3.6]), np.array([0.0, 0.0]))
_corridor_walls = [_bottom, _right, _top, _left]


def test_wall_pushes_away():
    """Agent at (5, 0.3) near bottom wall -> positive y force."""
    wf = WallForces()
    state = _make_state([[5.0, 0.3]])
    forces = wf.compute_wall_forces(state, [_bottom])

    # Should be pushed upward (positive y)
    assert forces[0, 1] > 0
    # x component should be ~0 (agent is along middle of wall)
    assert abs(forces[0, 0]) < 1e-6


def test_contact_wall():
    """Agent at (5, 0.1) with r=0.25 -> overlapping bottom wall -> large force."""
    wf = WallForces()
    state = _make_state([[5.0, 0.1]])
    forces = wf.compute_wall_forces(state, [_bottom])

    mag = np.linalg.norm(forces[0])
    # overlap = 0.25 - 0.1 = 0.15, body = 120000*0.15 = 18000N plus social
    assert mag > 10000


def test_far_from_wall():
    """Agent at (5, 1.8) -> far from bottom wall -> force magnitude < 1.0N."""
    wf = WallForces()
    state = _make_state([[5.0, 1.8]])
    forces = wf.compute_wall_forces(state, [_bottom])

    mag = np.linalg.norm(forces[0])
    assert mag < 1.0


def test_corner_push():
    """Agent at (0.3, 0.3) near two walls -> force pushes toward center."""
    wf = WallForces()
    state = _make_state([[0.3, 0.3]])
    forces = wf.compute_wall_forces(state, [_bottom, _left])

    # Both components positive (pushed away from corner)
    assert forces[0, 0] > 0  # pushed right from left wall
    assert forces[0, 1] > 0  # pushed up from bottom wall


def test_no_walls():
    """No walls -> zero force."""
    wf = WallForces()
    state = _make_state([[5.0, 1.8]])
    forces = wf.compute_wall_forces(state, [])

    np.testing.assert_allclose(forces[0], [0.0, 0.0], atol=1e-12)


def test_corridor_symmetric():
    """Agent centered in corridor should have ~zero net wall force."""
    wf = WallForces()
    state = _make_state([[5.0, 1.8]])  # center of 3.6m-wide corridor
    forces = wf.compute_wall_forces(state, _corridor_walls)

    # Top and bottom cancel, left and right cancel at midpoint
    np.testing.assert_allclose(forces[0], [0.0, 0.0], atol=1.0)


def test_multiple_agents():
    """Vectorized wall forces for multiple agents don't produce NaN."""
    wf = WallForces()
    rng = np.random.Generator(np.random.PCG64(42))
    positions = np.column_stack([rng.uniform(0.5, 9.5, 20), rng.uniform(0.5, 3.1, 20)])
    state = _make_state(positions)
    forces = wf.compute_wall_forces(state, _corridor_walls)

    assert forces.shape == (20, 2)
    assert not np.any(np.isnan(forces))


def test_friction_opposes_motion():
    """Agent sliding right along bottom wall -> friction opposes rightward motion."""
    wf = WallForces()
    state = _make_state([[5.0, 0.1]], velocities=[[2.0, 0.0]])
    forces = wf.compute_wall_forces(state, [_bottom])

    # Agent overlaps bottom wall (r=0.25, d=0.1, overlap=0.15)
    # Normal is upward (+y), tangent is (-1, 0) [= (-n_y, n_x)]
    # Friction should oppose rightward motion -> force has negative x component
    assert forces[0, 0] < 0, f"Friction should oppose rightward motion, got fx={forces[0, 0]}"
    # y component is positive (pushed away from wall)
    assert forces[0, 1] > 0


def test_friction_magnitude():
    """Verify wall friction magnitude against hand calculation."""
    wf = WallForces()
    state = _make_state([[5.0, 0.1]], velocities=[[1.0, 0.0]])
    forces = wf.compute_wall_forces(state, [_bottom])

    # overlap = 0.25 - 0.1 = 0.15
    # normal = (0, 1), tangent = (-1, 0)
    # dot(v, t) = dot((1,0), (-1,0)) = -1
    # friction = -kappa * overlap * dot(v,t) * t = -240000*0.15*(-1)*(-1,0) = -36000*(−1,0)
    # = (−36000, 0) ... wait: -kappa*overlap*vt*t = -240000*0.15*(-1)*(-1,0) = -240000*0.15*1*(1,0)
    # Hmm let me recalculate: -kappa*overlap*vt * t_iw
    # vt = dot(v_i, t_iw) = dot((1,0),(-1,0)) = -1
    # f_friction = -(240000 * 0.15 * (-1)) * (-1, 0) = -(−36000) * (−1, 0) = 36000*(−1,0) = (−36000, 0)
    # So friction component is (-36000, 0) -- opposing rightward motion. Good.
    # Total x force = social_x + body_x + friction_x.
    # social/body push upward (y), so x component is friction only.
    # But social has a tiny x component from the normal direction...
    # For agent at (5, 0.1) above bottom wall: normal = (0, 1), so social/body are purely +y.
    # Friction is purely -x.
    expected_friction_x = -240000 * 0.15 * 1.0  # -36000
    # x-component of force should be close to friction-only (social/body have zero x)
    np.testing.assert_allclose(forces[0, 0], expected_friction_x, rtol=0.01)
