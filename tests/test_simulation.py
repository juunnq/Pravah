"""Tests for the simulation loop."""

import numpy as np
import pytest

from sim.core.agent import AgentState
from sim.core.integrator import EulerIntegrator
from sim.core.simulation import Simulation
from sim.core.world import World, Wall


@pytest.fixture
def simple_sim():
    """10 agents in a box, desired-force only."""
    walls = [
        Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0])),
        Wall(np.array([10.0, 0.0]), np.array([10.0, 10.0])),
        Wall(np.array([10.0, 10.0]), np.array([0.0, 10.0])),
        Wall(np.array([0.0, 10.0]), np.array([0.0, 0.0])),
    ]
    world = World(walls)
    state = AgentState.create(
        10,
        spawn_area=(1.0, 3.0, 1.0, 9.0),
        goals=np.array([9.0, 5.0]),
        seed=42,
        heterogeneous=False,
    )
    params = {"dt": 0.01, "neighbor_radius": 3.0, "max_time": 300.0, "goal_reached_dist": 0.5}
    return Simulation(world, state, steering_model=None, integrator=EulerIntegrator(), params=params)


def test_agents_move(simple_sim):
    """After 200 steps, mean x-position of active agents should have increased."""
    initial_active = simple_sim.state.active.copy()
    initial_mean_x = np.mean(simple_sim.state.positions[initial_active, 0])
    for _ in range(200):
        simple_sim.step()
    # Compare active agents only, using the same population basis
    still_active = simple_sim.state.active
    if np.any(still_active):
        final_mean_x = np.mean(simple_sim.state.positions[still_active, 0])
    else:
        # All agents exited (reached goal at x=9), so they definitely moved right
        final_mean_x = 9.0
    assert final_mean_x > initial_mean_x


def test_no_nan(simple_sim):
    """200 steps produce no NaN in positions or velocities."""
    for _ in range(200):
        simple_sim.step()
    assert not np.any(np.isnan(simple_sim.state.positions))
    assert not np.any(np.isnan(simple_sim.state.velocities))


def test_goal_deactivation():
    """Agent placed very close to goal gets deactivated."""
    walls = [Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0]))]
    world = World(walls)
    # Place agent right next to goal
    state = AgentState(
        positions=np.array([[4.8, 5.0]]),
        velocities=np.array([[1.34, 0.0]]),
        goals=np.array([[5.0, 5.0]]),
        radii=np.array([0.25]),
        desired_speeds=np.array([1.34]),
        masses=np.array([80.0]),
        taus=np.array([0.5]),
        active=np.array([True]),
    )
    params = {"dt": 0.01, "neighbor_radius": 3.0, "max_time": 300.0, "goal_reached_dist": 0.5}
    sim = Simulation(world, state, steering_model=None, integrator=EulerIntegrator(), params=params)

    # Run enough steps for agent to reach goal (0.2m at ~1.34 m/s)
    for _ in range(100):
        sim.step()

    assert state.n_active == 0


def test_step_returns_metrics(simple_sim):
    """step() returns a dict with time, n_active, mean_speed."""
    metrics = simple_sim.step()
    assert "time" in metrics
    assert "n_active" in metrics
    assert "mean_speed" in metrics
    assert metrics["n_active"] > 0
    assert metrics["time"] > 0


def test_run_terminates(simple_sim):
    """run() terminates and returns results."""
    results = simple_sim.run(max_steps=500, max_time=5.0)
    assert "n_steps" in results
    assert "evacuation_time" in results
    assert "agents_exited" in results
    assert "mean_speed" in results
    assert results["n_steps"] > 0


def test_metrics_log_grows(simple_sim):
    """metrics_log accumulates one entry per step."""
    for _ in range(50):
        simple_sim.step()
    assert len(simple_sim.metrics_log) == 50


def test_speed_clamping():
    """Agents don't exceed 2x desired speed."""
    walls = [Wall(np.array([0.0, 0.0]), np.array([100.0, 0.0]))]
    world = World(walls)
    state = AgentState.create(
        5,
        spawn_area=(0.5, 1.0, 0.5, 1.0),
        goals=np.array([100.0, 0.75]),
        seed=42,
        heterogeneous=False,
    )
    params = {"dt": 0.01, "neighbor_radius": 3.0, "max_time": 300.0, "goal_reached_dist": 0.5}
    sim = Simulation(world, state, steering_model=None, integrator=EulerIntegrator(), params=params)

    for _ in range(500):
        sim.step()

    speeds = np.linalg.norm(state.velocities[state.active], axis=1)
    max_allowed = 2.0 * state.desired_speeds[state.active]
    assert np.all(speeds <= max_allowed + 1e-8)


def test_simulation_accepts_voronoi_density_estimator():
    """Simulation with VoronoiDensityEstimator runs without crashing and
    produces different densities than the default grid estimator."""
    from sim.density.voronoi import VoronoiDensityEstimator
    from sim.scenarios.crush_room import CrushRoomScenario

    # Crush room with 40 agents packed -> strong local density peaks
    scenario_default = CrushRoomScenario(n_agents=40, exit_width=0.6)
    scenario_vor = CrushRoomScenario(n_agents=40, exit_width=0.6)

    sim_default = Simulation.from_scenario(scenario_default, "C1", seed=42)
    sim_vor = Simulation.from_scenario(
        scenario_vor, "C1", seed=42,
        density_estimator=VoronoiDensityEstimator(),
    )

    # Run a few steps
    for _ in range(30):
        sim_default.step()
        sim_vor.step()

    # Extract recorded max densities from metrics_log
    grid_max = max(m["max_density"] for m in sim_default.metrics_log)
    vor_max = max(m["max_density"] for m in sim_vor.metrics_log)
    assert vor_max > grid_max, \
        f"Voronoi max density ({vor_max:.2f}) should exceed grid max ({grid_max:.2f})"
