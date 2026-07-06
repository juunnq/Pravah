"""Smoke tests: end-to-end simulation with full steering configs."""

import numpy as np
import pytest

from sim.core.agent import AgentState
from sim.core.integrator import EulerIntegrator
from sim.core.simulation import Simulation
from sim.core.world import World, Wall
from sim.experiments.configs import CONFIGS
from sim.steering.hybrid import HybridSteeringModel


def _default_params():
    return {
        "A": 2000.0, "B": 0.08, "k": 120000.0, "kappa": 240000.0,
        "k_ttc": 1.5, "tau_0": 3.0, "tau_max": 8.0,
        "time_horizon": 5.0, "tau_orca": 0.5, "dt": 0.01,
        "k_crush": 360000.0, "kappa_crush": 480000.0,
        "rho_orca_fade": 4.0, "k_orca_fade": 2.0,
        "rho_crit": 5.5, "k_crit": 3.0,
        "neighbor_radius": 3.0, "goal_reached_dist": 0.5, "max_time": 300.0,
    }


def _bottleneck_world(exit_width=1.2):
    """10x10 room with an exit gap on the right wall."""
    half = exit_width / 2.0
    walls = [
        Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0])),   # bottom
        Wall(np.array([0.0, 10.0]), np.array([0.0, 0.0])),    # left
        Wall(np.array([10.0, 10.0]), np.array([0.0, 10.0])),  # top
        # right wall with gap centered at y=5
        Wall(np.array([10.0, 0.0]), np.array([10.0, 5.0 - half])),
        Wall(np.array([10.0, 5.0 + half]), np.array([10.0, 10.0])),
    ]
    return World(walls)


def _corridor_world():
    walls = [
        Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0])),
        Wall(np.array([10.0, 0.0]), np.array([10.0, 3.6])),
        Wall(np.array([10.0, 3.6]), np.array([0.0, 3.6])),
        Wall(np.array([0.0, 3.6]), np.array([0.0, 0.0])),
    ]
    return World(walls)


def test_evacuation_c4():
    """50 agents in 10x10 room (1.2m exit), C4, 2000 steps -> no NaN, some exit."""
    world = _bottleneck_world(exit_width=1.2)
    state = AgentState.create(
        50, spawn_area=(1.0, 8.0, 1.0, 9.0),
        goals=np.array([11.0, 5.0]), seed=42,
    )
    params = _default_params()
    hybrid = HybridSteeringModel(CONFIGS["C4"], params)
    sim = Simulation(world, state, hybrid, EulerIntegrator(), params)

    result = sim.run(max_steps=2000, max_time=20.0)

    assert not np.any(np.isnan(state.positions))
    assert not np.any(np.isnan(state.velocities))
    assert result["agents_exited"] > 0, "No agents exited the bottleneck"


def test_all_configs_run():
    """Each C1-C4 with 20 agents in corridor, 300 steps -> no crash."""
    world = _corridor_world()
    params = _default_params()

    for name, config in CONFIGS.items():
        state = AgentState.create(
            20, spawn_area=(0.5, 3.0, 0.5, 3.1),
            goals=np.array([9.5, 1.8]), seed=42,
        )
        hybrid = HybridSteeringModel(config, params)
        sim = Simulation(world, state, hybrid, EulerIntegrator(), params)

        for _ in range(300):
            metrics = sim.step()

        assert not np.any(np.isnan(state.positions)), f"{name} NaN positions"
        assert not np.any(np.isnan(state.velocities)), f"{name} NaN velocities"
        assert metrics["mean_speed"] >= 0, f"{name} negative speed"


def test_single_agent_reaches_goal():
    """One agent in corridor, C1, 1000 steps -> reaches goal."""
    world = _corridor_world()
    state = AgentState.create(
        1, spawn_area=(1.0, 1.5, 1.5, 2.0),
        goals=np.array([9.0, 1.8]), seed=42, heterogeneous=False,
    )
    params = _default_params()
    hybrid = HybridSteeringModel(CONFIGS["C1"], params)
    sim = Simulation(world, state, hybrid, EulerIntegrator(), params)

    # ~8m at 1.34m/s needs ~6s; allow 10s (1000 steps at dt=0.01)
    result = sim.run(max_steps=1000, max_time=10.0)

    assert result["agents_exited"] == 1, "Single agent did not reach goal"


def test_head_on_deflection():
    """Two agents walking at each other, C4 -> they pass without exploding."""
    walls = [
        Wall(np.array([0.0, 0.0]), np.array([20.0, 0.0])),
        Wall(np.array([20.0, 0.0]), np.array([20.0, 3.6])),
        Wall(np.array([20.0, 3.6]), np.array([0.0, 3.6])),
        Wall(np.array([0.0, 3.6]), np.array([0.0, 0.0])),
    ]
    world = World(walls)

    state = AgentState(
        positions=np.array([[2.0, 1.8], [18.0, 1.8]]),
        velocities=np.array([[1.34, 0.0], [-1.34, 0.0]]),
        goals=np.array([[19.0, 1.8], [1.0, 1.8]]),
        radii=np.array([0.25, 0.25]),
        desired_speeds=np.array([1.34, 1.34]),
        masses=np.array([80.0, 80.0]),
        taus=np.array([0.5, 0.5]),
        active=np.array([True, True]),
    )

    params = _default_params()
    hybrid = HybridSteeringModel(CONFIGS["C4"], params)
    sim = Simulation(world, state, hybrid, EulerIntegrator(), params)

    max_speed_seen = 0.0
    for _ in range(1500):
        sim.step()
        speeds = np.linalg.norm(state.velocities[state.active], axis=1)
        if len(speeds) > 0:
            max_speed_seen = max(max_speed_seen, float(np.max(speeds)))

    assert not np.any(np.isnan(state.positions)), "NaN in positions"
    assert not np.any(np.isnan(state.velocities)), "NaN in velocities"
    # Speed should stay reasonable (< 2x desired = 2.68)
    assert max_speed_seen < 3.0, f"Speed exploded: {max_speed_seen:.2f}"
