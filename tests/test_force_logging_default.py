"""Regression test: log_forces=False preserves bit-identical default path.

Uses the same reference as test_logging_flags_regression.py:
Bottleneck w=1.0m C1 seed=42.
"""

import math
import pytest

from sim.core.simulation import Simulation
from sim.scenarios.bottleneck import BottleneckScenario

REFERENCE = {
    "n_steps": 5844,
    "evacuation_time": 58.43999999999694,
    "mean_speed": 0.3339698210842611,
    "max_density": 1.5915494309189535,
    "collision_count": 54,
    "flow_rate": 0.8555783709788264,
    "agents_exited": 50,
    "time_above_critical": 0,
}


def _run(log_forces=False):
    scenario = BottleneckScenario(n_agents=50, exit_width=1.0)
    sim = Simulation.from_scenario(scenario, "C1", seed=42)
    sim.log_forces = log_forces
    return sim.run(max_steps=100000, max_time=300.0), sim


class TestForceLoggingDefault:
    def test_forces_off_matches_reference(self):
        """log_forces=False must be bit-identical to stored reference."""
        result, _ = _run(False)
        for key, ref_val in REFERENCE.items():
            actual = result[key]
            if isinstance(ref_val, float) and not math.isinf(ref_val):
                assert actual == pytest.approx(ref_val, rel=1e-12), \
                    f"{key}: expected {ref_val}, got {actual}"
            else:
                assert actual == ref_val, f"{key}: expected {ref_val}, got {actual}"

    def test_forces_on_matches_reference(self):
        """log_forces=True must still be bit-identical (observer only)."""
        result, sim = _run(True)
        for key, ref_val in REFERENCE.items():
            actual = result[key]
            if isinstance(ref_val, float) and not math.isinf(ref_val):
                assert actual == pytest.approx(ref_val, rel=1e-12), \
                    f"{key}: expected {ref_val}, got {actual}"
            else:
                assert actual == ref_val, f"{key}: expected {ref_val}, got {actual}"

    def test_force_log_populated_when_on(self):
        """log_forces=True should produce non-empty _force_log."""
        _, sim = _run(True)
        assert len(sim._force_log) > 0
        entry = sim._force_log[0]
        assert "t" in entry
        assert "agent_id" in entry
        assert "density" in entry
        assert "mag_des" in entry
        assert "mag_sfm" in entry

    def test_force_log_empty_when_off(self):
        """log_forces=False should leave _force_log empty."""
        _, sim = _run(False)
        assert len(sim._force_log) == 0
