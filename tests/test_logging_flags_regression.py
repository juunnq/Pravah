"""Regression test: logging flags preserve bit-identical default path.

Runs Bottleneck w=1.0m C1 seed=42 with both flags False and compares
every numeric field to the stored reference in results/Bottleneck_w1.0_C1.csv.
This guarantees the logging additions do not perturb the simulation.
"""

import math

import numpy as np
import pandas as pd
import pytest

from sim.core.simulation import Simulation
from sim.scenarios.bottleneck import BottleneckScenario


# Reference values from the frozen results/ CSV
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


def _run_bottleneck_w10_c1_seed42(log_positions=False, log_collisions=False):
    """Run the reference simulation and return the compiled result dict."""
    scenario = BottleneckScenario(n_agents=50, exit_width=1.0)
    sim = Simulation.from_scenario(scenario, "C1", seed=42)
    sim.log_positions = log_positions
    sim.log_collisions = log_collisions
    result = sim.run(max_steps=100000, max_time=300.0)
    return result, sim


class TestLoggingFlagsRegression:
    """Ensure logging flags don't change simulation results."""

    def test_default_flags_false_match_reference(self):
        """Both flags False: result must be bit-identical to stored CSV."""
        result, _ = _run_bottleneck_w10_c1_seed42(False, False)
        for key, ref_val in REFERENCE.items():
            actual = result[key]
            if isinstance(ref_val, float) and not math.isinf(ref_val):
                assert actual == pytest.approx(ref_val, rel=1e-12), \
                    f"{key}: expected {ref_val}, got {actual}"
            else:
                assert actual == ref_val, f"{key}: expected {ref_val}, got {actual}"

    def test_flags_true_match_reference(self):
        """Both flags True: result must still be bit-identical."""
        result, _ = _run_bottleneck_w10_c1_seed42(True, True)
        for key, ref_val in REFERENCE.items():
            actual = result[key]
            if isinstance(ref_val, float) and not math.isinf(ref_val):
                assert actual == pytest.approx(ref_val, rel=1e-12), \
                    f"{key}: expected {ref_val}, got {actual}"
            else:
                assert actual == ref_val, f"{key}: expected {ref_val}, got {actual}"

    def test_position_log_populated_when_enabled(self):
        """With log_positions=True, _position_log is non-empty."""
        _, sim = _run_bottleneck_w10_c1_seed42(True, False)
        assert len(sim._position_log) > 0
        # Each entry: (t, agent_ids, positions, velocities)
        t, ids, pos, vel = sim._position_log[0]
        assert isinstance(t, float)
        assert pos.shape[1] == 2
        assert vel.shape[1] == 2
        assert len(ids) == pos.shape[0]

    def test_collision_log_populated_when_enabled(self):
        """With log_collisions=True, _collision_log is non-empty (w=1.0 has collisions)."""
        _, sim = _run_bottleneck_w10_c1_seed42(False, True)
        assert len(sim._collision_log) > 0
        # Each entry: (t, i, j, x_i, y_i, x_j, y_j)
        entry = sim._collision_log[0]
        assert len(entry) == 7

    def test_position_log_empty_when_disabled(self):
        """With log_positions=False, _position_log stays empty."""
        _, sim = _run_bottleneck_w10_c1_seed42(False, False)
        assert len(sim._position_log) == 0

    def test_collision_log_empty_when_disabled(self):
        """With log_collisions=False, _collision_log stays empty."""
        _, sim = _run_bottleneck_w10_c1_seed42(False, False)
        assert len(sim._collision_log) == 0

    def test_write_logs_produces_parquet(self, tmp_path):
        """write_logs produces valid parquet files."""
        _, sim = _run_bottleneck_w10_c1_seed42(True, True)
        traj_path = str(tmp_path / "traj.parquet")
        coll_path = str(tmp_path / "coll.parquet")
        sim.write_logs(trajectory_path=traj_path, collision_path=coll_path)

        traj_df = pd.read_parquet(traj_path)
        assert set(traj_df.columns) == {"t", "agent_id", "x", "y", "vx", "vy"}
        assert len(traj_df) > 0

        coll_df = pd.read_parquet(coll_path)
        assert set(coll_df.columns) == {"t", "i", "j", "x_i", "y_i", "x_j", "y_j"}
        assert len(coll_df) > 0
