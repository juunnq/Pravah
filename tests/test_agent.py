"""Tests for AgentState dataclass."""

import numpy as np
from sim.core.agent import AgentState


def test_create_shape():
    """50 agents: verify all array shapes."""
    state = AgentState.create(50, spawn_area=(0, 10, 0, 10), goals=np.array([5.0, 5.0]), seed=1)
    assert state.positions.shape == (50, 2)
    assert state.velocities.shape == (50, 2)
    assert state.goals.shape == (50, 2)
    assert state.radii.shape == (50,)
    assert state.desired_speeds.shape == (50,)
    assert state.masses.shape == (50,)
    assert state.taus.shape == (50,)
    assert state.active.shape == (50,)
    assert state.n == 50
    assert state.n_active == 50


def test_heterogeneous():
    """100 agents heterogeneous=True: speed std > 0.1."""
    state = AgentState.create(100, spawn_area=(0, 10, 0, 10), goals=np.array([5.0, 5.0]), seed=7, heterogeneous=True)
    assert np.std(state.desired_speeds) > 0.1
    assert np.all(state.desired_speeds >= 0.5)
    assert np.all(state.radii >= 0.15)


def test_homogeneous():
    """100 agents heterogeneous=False: all speeds identical."""
    state = AgentState.create(100, spawn_area=(0, 10, 0, 10), goals=np.array([5.0, 5.0]), seed=7, heterogeneous=False)
    assert np.all(state.desired_speeds == state.desired_speeds[0])
    assert np.all(state.radii == state.radii[0])
    assert np.all(state.taus == state.taus[0])


def test_deactivate():
    """Create 20, deactivate indices [0,5,10], verify n_active==17."""
    state = AgentState.create(20, spawn_area=(0, 5, 0, 5), goals=np.array([10.0, 2.5]), seed=3)
    state.deactivate(np.array([0, 5, 10]))
    assert state.n == 20
    assert state.n_active == 17
    assert not state.active[0]
    assert not state.active[5]
    assert not state.active[10]


def test_active_indices():
    """Deactivate some, verify active_indices matches."""
    state = AgentState.create(10, spawn_area=(0, 5, 0, 5), goals=np.array([10.0, 2.5]), seed=3)
    state.deactivate(np.array([2, 7]))
    indices = state.active_indices
    assert len(indices) == 8
    assert 2 not in indices
    assert 7 not in indices
    assert np.array_equal(indices, np.array([0, 1, 3, 4, 5, 6, 8, 9]))
