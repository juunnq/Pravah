"""Tests for the Phase-2 timed-release surge mechanism.

Stepped tests use config "C1" (no ORCA) and short horizons to stay fast. Crush/C4
behaviour is Phase 3, not exercised here.
"""

import numpy as np
import pytest

from sim.core.simulation import Simulation
from sim.scenarios.railway_platform import RailwayPlatformScenario
from sim.scenarios.railway_release import (
    BASELINE_SCHEDULE,
    INTERVENTION_SCHEDULE,
    RAILWAY_PARAM_OVERRIDES,
    ReleaseSchedule,
    run_surge,
)

CORRIDOR_BOX = (0.3, 2.0, 0.3, 3.3)
CORRIDOR_GOAL = np.array([26.0, 1.8])


# 1. Frozen schedules total 150.
def test_schedule_totals():
    assert BASELINE_SCHEDULE.total == 150
    assert INTERVENTION_SCHEDULE.total == 150


# 2. Validation rejects bad input.
@pytest.mark.parametrize("bad", [
    ((0.0, 10), (5.0, 10), (3.0, 10)),   # unsorted
    ((0.0, 10), (5.0, 10), (5.0, 10)),   # duplicate time (not strictly ascending)
    ((-1.0, 10),),                       # negative time
    ((0.0, 0),),                         # n == 0
    ((0.0, -5),),                        # n < 0
    (),                                  # empty
])
def test_schedule_validation_rejects(bad):
    with pytest.raises(ValueError):
        ReleaseSchedule(bad)


# 3. batches_due / cumulative count (pure, no sim).
def test_batches_due_pure_logic():
    # Baseline: t=0 -> 1 batch (50); t=3 -> 2 (100); t=6 -> 3 (150).
    assert len(BASELINE_SCHEDULE.batches_due(0.0)) == 1
    assert BASELINE_SCHEDULE.n_due(0.0) == 50
    assert len(BASELINE_SCHEDULE.batches_due(3.0)) == 2
    assert BASELINE_SCHEDULE.n_due(3.0) == 100
    assert len(BASELINE_SCHEDULE.batches_due(6.0)) == 3
    assert BASELINE_SCHEDULE.n_due(6.0) == 150
    # Intervention: t=79 -> 4 (120); t=80 -> 5 (150).
    assert len(INTERVENTION_SCHEDULE.batches_due(79.0)) == 4
    assert INTERVENTION_SCHEDULE.n_due(79.0) == 120
    assert len(INTERVENTION_SCHEDULE.batches_due(80.0)) == 5
    assert INTERVENTION_SCHEDULE.n_due(80.0) == 150
    # release_steps at dt=0.01
    assert INTERVENTION_SCHEDULE.release_steps(0.01) == [0, 2000, 4000, 6000, 8000]


def _railway_sim(n_agents: int = 10, config: str = "C1") -> tuple:
    scenario = RailwayPlatformScenario(n_agents=n_agents)
    sim = Simulation.from_scenario(scenario, config, seed=42)
    return scenario, sim


# 4. inject_agents backward compatibility (default corridor path unchanged).
def test_inject_agents_backward_compatible():
    _, sim = _railway_sim(n_agents=10)
    n0 = sim.state.n
    sim.inject_agents(5, seed=0)
    assert sim.state.n == n0 + 5
    new_pos = sim.state.positions[n0:]
    new_goals = sim.state.goals[n0:]
    x0, x1, y0, y1 = CORRIDOR_BOX
    assert np.all(new_pos[:, 0] >= x0) and np.all(new_pos[:, 0] <= x1)
    assert np.all(new_pos[:, 1] >= y0) and np.all(new_pos[:, 1] <= y1)
    assert np.allclose(new_goals, CORRIDOR_GOAL)


# 5. inject_agents generalized into the railway holding area.
def test_inject_agents_generalized():
    scenario, sim = _railway_sim(n_agents=10)
    n0 = sim.state.n
    sim.inject_agents(7, seed=0, spawn_area=scenario.spawn_area, goal=scenario.goal)
    assert sim.state.n == n0 + 7
    new_pos = sim.state.positions[n0:]
    new_goals = sim.state.goals[n0:]
    new_speeds = sim.state.desired_speeds[n0:]
    new_radii = sim.state.radii[n0:]
    assert np.all(new_pos[:, 0] >= 2.0) and np.all(new_pos[:, 0] <= 14.0)
    assert np.all(new_pos[:, 1] >= 4.0) and np.all(new_pos[:, 1] <= 16.0)
    assert np.allclose(new_goals, np.array([53.0, 10.0]))
    assert new_speeds.std() > 0
    assert new_radii.std() > 0


# 6. avoid_overlap=True yields no spawn-overlap among active agents.
def test_inject_avoid_overlap():
    scenario, sim = _railway_sim(n_agents=20)
    for _ in range(20):
        sim.step()
    sim.inject_agents(
        20, seed=1, spawn_area=scenario.spawn_area, goal=scenario.goal,
        avoid_overlap=True,
    )
    idx = sim.state.active_indices
    pos = sim.state.positions[idx]
    rad = sim.state.radii[idx]
    m = len(idx)
    for i in range(m):
        for j in range(i + 1, m):
            d = np.linalg.norm(pos[i] - pos[j])
            assert d >= rad[i] + rad[j], f"overlap between active agents {i},{j}: d={d}"


# 7. Cumulative equal-N — baseline (C1, bounded just past t=6).
def test_baseline_reaches_150():
    scenario = RailwayPlatformScenario()
    sim = run_surge(scenario, BASELINE_SCHEDULE, config="C1", seed=42, max_steps=650)
    assert sim.state.n == 150


# 8. Cumulative equal-N — intervention via pure logic + bounded injection check.
def test_intervention_release_logic_and_bounded():
    # Pure logic: all 5 batches sum to 150 and release at the right steps.
    assert INTERVENTION_SCHEDULE.total == 150
    assert INTERVENTION_SCHEDULE.n_due(80.0) == 150
    assert INTERVENTION_SCHEDULE.release_steps(0.01) == [0, 2000, 4000, 6000, 8000]
    # Bounded run: just past the 2nd batch (t=20 -> step 2000); n == 30 + 30.
    scenario = RailwayPlatformScenario()
    sim = run_surge(
        scenario, INTERVENTION_SCHEDULE, config="C1", seed=42, max_steps=2050
    )
    assert sim.state.n == 60


# 9. Determinism: same seed -> identical state.
def test_determinism_same_seed():
    s1 = run_surge(RailwayPlatformScenario(), BASELINE_SCHEDULE,
                   config="C1", seed=7, max_steps=350)
    s2 = run_surge(RailwayPlatformScenario(), BASELINE_SCHEDULE,
                   config="C1", seed=7, max_steps=350)
    assert s1.state.n == s2.state.n
    assert np.array_equal(s1.state.positions, s2.state.positions)


# 10. Bounded smoke (C1): no NaN, never zero-agent, reaches n==150.
def test_baseline_bounded_smoke():
    scenario = RailwayPlatformScenario()
    sim = run_surge(scenario, BASELINE_SCHEDULE, config="C1", seed=42, max_steps=650)
    assert not np.any(np.isnan(sim.state.positions))
    assert not np.any(np.isnan(sim.state.velocities))
    assert sim.state.n == 150
    assert sim.state.n_active > 0
    assert min(m["n_active"] for m in sim.metrics_log) > 0


# 11. Calibrated Weidmann (SIMULTECH Table 1) is applied by run_surge by default.
def test_calibrated_weidmann_applied():
    assert RAILWAY_PARAM_OVERRIDES == {
        "weidmann_gamma": 0.833, "weidmann_rho_max": 5.98,
    }
    sim = run_surge(RailwayPlatformScenario(), BASELINE_SCHEDULE,
                    config="C1", seed=42, max_steps=5)
    assert sim.params["weidmann_gamma"] == pytest.approx(0.833)
    assert sim.params["weidmann_rho_max"] == pytest.approx(5.98)


# 12. Explicit param_overrides win over the calibrated defaults.
def test_param_overrides_win_over_calibration():
    sim = run_surge(
        RailwayPlatformScenario(), BASELINE_SCHEDULE, config="C1", seed=42,
        max_steps=5, param_overrides={"weidmann_gamma": 1.913},
    )
    assert sim.params["weidmann_gamma"] == pytest.approx(1.913)   # explicit wins
    assert sim.params["weidmann_rho_max"] == pytest.approx(5.98)  # default kept


# 13. Non-railway paths are untouched: plain from_scenario carries no weidmann keys.
def test_calibration_does_not_leak_to_other_scenarios():
    scenario = RailwayPlatformScenario(n_agents=5)
    sim = Simulation.from_scenario(scenario, "C1", seed=42)
    assert "weidmann_gamma" not in sim.params
    assert "weidmann_rho_max" not in sim.params


# 13b. spawn_area override: default preserves the frozen box; a proximity box
#      places agents (and injected batches, via scenario.spawn_area) inside it.
def test_spawn_area_override():
    # Default = frozen spec box.
    assert RailwayPlatformScenario().spawn_area == (2.0, 14.0, 4.0, 16.0)
    # Override: initial population lands inside the proximity box.
    box = (12.0, 19.5, 3.0, 17.0)
    scenario = RailwayPlatformScenario(spawn_area=box)
    assert scenario.spawn_area == box
    _, state = scenario.build(seed=42)
    assert np.all(state.positions[:, 0] >= box[0])
    assert np.all(state.positions[:, 0] <= box[1])
    assert np.all(state.positions[:, 1] >= box[2])
    assert np.all(state.positions[:, 1] <= box[3])


# 14. speed_mean flows to BOTH the initial population and injected batches;
#     default 1.34 preserves prior behavior.
def test_speed_mean_flows_through_surge():
    # Default: calm walking.
    _, state = RailwayPlatformScenario().build(seed=42)
    assert state.desired_speeds.mean() == pytest.approx(1.34, abs=0.1)

    # Hurried crowd: initial batch + injected batches all at the surge mean.
    scenario = RailwayPlatformScenario(speed_mean=2.0)
    sim = run_surge(scenario, BASELINE_SCHEDULE, config="C1", seed=42,
                    max_steps=650)
    assert sim.state.n == 150
    assert sim.state.desired_speeds[:50].mean() == pytest.approx(2.0, abs=0.15)
    assert sim.state.desired_speeds[50:].mean() == pytest.approx(2.0, abs=0.15)
    # Heterogeneity preserved (sampled, not constant).
    assert sim.state.desired_speeds.std() > 0
