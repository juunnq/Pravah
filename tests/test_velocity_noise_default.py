"""Regression test: velocity_noise_std=0.0 preserves bit-identical default path.

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


def _run(noise_std=0.0):
    scenario = BottleneckScenario(n_agents=50, exit_width=1.0)
    sim = Simulation.from_scenario(scenario, "C1", seed=42)
    sim.velocity_noise_std = noise_std
    return sim.run(max_steps=100000, max_time=300.0), sim


class TestVelocityNoiseDefault:
    def test_noise_zero_matches_reference(self):
        """velocity_noise_std=0.0 must be bit-identical to stored reference."""
        result, _ = _run(0.0)
        for key, ref_val in REFERENCE.items():
            actual = result[key]
            if isinstance(ref_val, float) and not math.isinf(ref_val):
                assert actual == pytest.approx(ref_val, rel=1e-12), \
                    f"{key}: expected {ref_val}, got {actual}"
            else:
                assert actual == ref_val, f"{key}: expected {ref_val}, got {actual}"

    def test_noise_positive_differs(self):
        """velocity_noise_std > 0 should produce different results."""
        result, _ = _run(0.05)
        # The noise changes the trajectory so metrics will differ
        # (not necessarily all, but at least evacuation_time or collision_count)
        differs = (
            result["evacuation_time"] != REFERENCE["evacuation_time"]
            or result["collision_count"] != REFERENCE["collision_count"]
            or result["mean_speed"] != pytest.approx(REFERENCE["mean_speed"], rel=1e-6)
        )
        assert differs, "Noise should alter at least one metric"
