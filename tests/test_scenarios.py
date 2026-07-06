"""Tests for the retained simulation scenarios."""

import numpy as np
import pytest

from sim.core.integrator import EulerIntegrator
from sim.core.simulation import Simulation
from sim.experiments.configs import CONFIGS
from sim.scenarios.corridor import CorridorScenario
from sim.scenarios.bottleneck import BottleneckScenario
from sim.scenarios.funnel import FunnelScenario
from sim.scenarios.crush_room import CrushRoomScenario
from sim.steering.hybrid import HybridSteeringModel


_SCENARIOS = [
    ("Corridor", CorridorScenario(n_agents=10)),
    ("Bottleneck", BottleneckScenario(n_agents=15, exit_width=1.2)),
    ("Funnel", FunnelScenario(n_agents=15)),
    ("CrushRoom", CrushRoomScenario(n_agents=20, exit_width=0.6)),
]

_DEFAULT_PARAMS = {
    "A": 2000.0, "B": 0.08, "k": 120000.0, "kappa": 240000.0,
    "k_ttc": 1.5, "tau_0": 3.0, "tau_max": 8.0,
    "time_horizon": 5.0, "tau_orca": 0.5, "dt": 0.01,
    "k_crush": 360000.0, "kappa_crush": 480000.0,
    "rho_orca_fade": 4.0, "k_orca_fade": 2.0,
    "rho_crit": 5.5, "k_crit": 3.0,
    "neighbor_radius": 3.0, "goal_reached_dist": 0.5, "max_time": 300.0,
}


@pytest.mark.parametrize("name,scenario", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_scenario_builds(name, scenario):
    """Each scenario builds with correct agent count and valid positions."""
    world, state = scenario.build(seed=42)

    assert state.n > 0
    assert state.positions.shape == (state.n, 2)
    assert state.goals.shape == (state.n, 2)
    assert not np.any(np.isnan(state.positions))
    assert len(world.walls) > 0


def test_bottleneck_exit_width():
    """Bottleneck exit gap is at the correct width."""
    for width in [0.8, 1.2, 2.4]:
        scenario = BottleneckScenario(n_agents=10, exit_width=width)
        world, _ = scenario.build()

        # Find the two right-wall segments
        right_walls = [w for w in world.walls if w.start[0] == 10.0 and w.end[0] == 10.0]
        assert len(right_walls) == 2

        # Gap between them
        y_ends = sorted([w.end[1] for w in right_walls] + [w.start[1] for w in right_walls])
        # The gap is between the top of the lower segment and bottom of upper segment
        lower_top = max(w.end[1] for w in right_walls if w.start[1] < 5.0)
        upper_bottom = min(w.start[1] for w in right_walls if w.start[1] > 5.0)
        gap = upper_bottom - lower_top
        np.testing.assert_allclose(gap, width, atol=0.01)


def test_funnel_walls_converge():
    """Funnel scenario walls taper from 10m at x=0 to 3m at x=15."""
    scenario = FunnelScenario(n_agents=10)
    world, _ = scenario.build()

    # Bottom wall: (0,0) -> (15, 3.5)
    # Top wall: (0,10) -> (15, 6.5)
    # Width at x=0: 10-0 = 10m, at x=15: 6.5-3.5 = 3m
    bottom = None
    top = None
    for w in world.walls:
        if np.allclose(w.start, [0, 0]) and np.allclose(w.end, [15, 3.5]):
            bottom = w
        if np.allclose(w.start, [0, 10]) and np.allclose(w.end, [15, 6.5]):
            top = w

    assert bottom is not None, "Bottom angled wall not found"
    assert top is not None, "Top angled wall not found"

    # Width at entrance (x=0)
    width_entrance = top.start[1] - bottom.start[1]
    np.testing.assert_allclose(width_entrance, 10.0)

    # Width at exit (x=15)
    width_exit = top.end[1] - bottom.end[1]
    np.testing.assert_allclose(width_exit, 3.0)


@pytest.mark.parametrize("name,scenario", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_scenario_runs_50_steps(name, scenario):
    """Each scenario runs 50 steps with C1 without crashing."""
    world, state = scenario.build(seed=42)
    hybrid = HybridSteeringModel(CONFIGS["C1"], _DEFAULT_PARAMS)
    sim = Simulation(world, state, hybrid, EulerIntegrator(), _DEFAULT_PARAMS)

    for _ in range(50):
        sim.step()

    assert not np.any(np.isnan(state.positions)), f"{name}: NaN positions"
    assert not np.any(np.isnan(state.velocities)), f"{name}: NaN velocities"


def test_corridor_spawn_bounds():
    """Corridor agents spawn within the specified area."""
    scenario = CorridorScenario(n_agents=50)
    _, state = scenario.build(seed=42)

    assert np.all(state.positions[:, 0] >= 0.5)
    assert np.all(state.positions[:, 0] <= 2.0)
    assert np.all(state.positions[:, 1] >= 0.3)
    assert np.all(state.positions[:, 1] <= 3.3)


def test_funnel_exit_width_parameter():
    """FunnelScenario exit_width parameter produces correctly sized gap."""
    for w in [0.6, 0.8, 1.0, 3.0]:
        scenario = FunnelScenario(n_agents=10, exit_width=w)
        world, _ = scenario.build()
        # Find the two angled walls ending at x=15
        bottom = next(wl for wl in world.walls
                      if np.allclose(wl.start, [0, 0]) and wl.end[0] == 15.0)
        top = next(wl for wl in world.walls
                   if np.allclose(wl.start, [0, 10]) and wl.end[0] == 15.0)
        gap = top.end[1] - bottom.end[1]
        np.testing.assert_allclose(gap, w, atol=0.01)


def test_crush_room_geometry():
    """CrushRoomScenario has correct exit gap on right wall."""
    scenario = CrushRoomScenario(n_agents=50, width=5.0, height=5.0, exit_width=0.6)
    world, state = scenario.build(seed=42)

    assert state.n == 50
    # Right wall should have two segments with gap of 0.6m
    right_walls = [w for w in world.walls
                   if w.start[0] == 5.0 and w.end[0] == 5.0]
    assert len(right_walls) == 2
    lower_top = max(w.end[1] for w in right_walls if w.start[1] < 2.5)
    upper_bottom = min(w.start[1] for w in right_walls if w.start[1] > 2.5)
    gap = upper_bottom - lower_top
    np.testing.assert_allclose(gap, 0.6, atol=0.01)
    # Agents are inside the room
    assert np.all(state.positions[:, 0] > 0) and np.all(state.positions[:, 0] < 5.0)
    assert np.all(state.positions[:, 1] > 0) and np.all(state.positions[:, 1] < 5.0)
