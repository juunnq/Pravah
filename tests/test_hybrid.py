"""Tests for hybrid steering model and sigmoid weights."""

import numpy as np
import pytest

from sim.core.agent import AgentState
from sim.core.world import World, Wall
from sim.steering.hybrid import HybridSteeringModel
from sim.experiments.configs import CONFIGS


def _corridor_world():
    """10x3.6 corridor."""
    walls = [
        Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0])),
        Wall(np.array([10.0, 0.0]), np.array([10.0, 3.6])),
        Wall(np.array([10.0, 3.6]), np.array([0.0, 3.6])),
        Wall(np.array([0.0, 3.6]), np.array([0.0, 0.0])),
    ]
    return World(walls)


def _make_state(n=10, seed=42):
    return AgentState.create(
        n,
        spawn_area=(1.0, 3.0, 0.5, 3.1),
        goals=np.array([9.0, 1.8]),
        seed=seed,
        heterogeneous=False,
    )


def _default_params():
    return {
        "A": 2000.0, "B": 0.08, "k": 120000.0, "kappa": 240000.0,
        "k_ttc": 1.5, "tau_0": 3.0, "tau_max": 8.0,
        "time_horizon": 5.0, "tau_orca": 0.5, "dt": 0.01,
        "k_crush": 360000.0, "kappa_crush": 480000.0,
        "rho_orca_fade": 4.0, "k_orca_fade": 2.0,
        "rho_crit": 5.5, "k_crit": 3.0,
    }


def test_c1_only_sfm():
    """C1 config: SFM + desired + walls only. No TTC, ORCA, crush."""
    hybrid = HybridSteeringModel(CONFIGS["C1"], _default_params())
    assert hybrid.ttc is None
    assert hybrid.orca is None
    assert hybrid.crush is None

    state = _make_state(5)
    world = _corridor_world()
    from scipy.spatial import KDTree
    tree = KDTree(state.positions)
    neighbors = tree.query_ball_point(state.positions, r=3.0)
    densities = np.ones(5) * 0.5  # low density

    forces = hybrid.compute_forces(state, neighbors, world.walls, densities)
    assert forces.shape == (5, 2)
    assert not np.any(np.isnan(forces))


def test_sigmoid_midpoint():
    """sigmoid(4.0, 4.0, 2.0) == 0.5 at midpoint."""
    result = HybridSteeringModel._sigmoid(np.array([4.0]), 4.0, 2.0)
    np.testing.assert_allclose(result[0], 0.5, atol=1e-10)


def test_low_density_weights():
    """rho=1 -> w_orca ~1.0, w_crush ~0.0."""
    rho = np.array([1.0])
    w_orca = 1.0 - HybridSteeringModel._sigmoid(rho, 4.0, 2.0)
    w_crush = HybridSteeringModel._sigmoid(rho, 5.5, 3.0)

    assert w_orca[0] > 0.99
    assert w_crush[0] < 0.001


def test_high_density_weights():
    """rho=7 -> w_orca ~0.0, w_crush ~1.0."""
    rho = np.array([7.0])
    w_orca = 1.0 - HybridSteeringModel._sigmoid(rho, 4.0, 2.0)
    w_crush = HybridSteeringModel._sigmoid(rho, 5.5, 3.0)

    assert w_orca[0] < 0.01
    assert w_crush[0] > 0.98


def test_all_configs():
    """Run C1-C4 each for 100 steps with 10 agents -> no crash, no NaN."""
    from sim.core.integrator import EulerIntegrator
    from sim.core.simulation import Simulation
    from sim.core.helpers import clamp_speed

    world = _corridor_world()
    params = _default_params()
    params.update({"neighbor_radius": 3.0, "goal_reached_dist": 0.5, "max_time": 300.0})

    for name, config in CONFIGS.items():
        state = _make_state(10, seed=42)
        hybrid = HybridSteeringModel(config, params)
        sim = Simulation(world, state, hybrid, EulerIntegrator(), params)

        for _ in range(100):
            sim.step()

        assert not np.any(np.isnan(state.positions)), f"{name}: NaN in positions"
        assert not np.any(np.isnan(state.velocities)), f"{name}: NaN in velocities"
