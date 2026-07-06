"""Experiment runner: batch execution of scenario x config x seeds -> CSV."""

import os
import time

import numpy as np
import pandas as pd

from sim.core.simulation import Simulation


class ExperimentRunner:
    """Batch experiment runner producing CSV results.

    Args:
        output_dir: Directory for CSV output files.
    """

    def __init__(self, output_dir: str = "results/"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def run(
        self,
        scenario_class,
        config_name: str,
        n_replications: int = 30,
        max_time: float = 120.0,
        max_steps: int = 100000,
        density_estimator=None,
        **scenario_kwargs,
    ) -> pd.DataFrame:
        """Run scenario x config x seeds and save to CSV.

        Args:
            scenario_class: Scenario class to instantiate.
            config_name: Configuration name (C1-C4).
            n_replications: Number of random seeds.
            max_time: Maximum simulation time per run (s).
            max_steps: Maximum steps per run.
            density_estimator: Optional density estimator override (default grid).
            **scenario_kwargs: Passed to scenario constructor.

        Returns:
            DataFrame with one row per replication.
        """
        rows = []
        for rep in range(n_replications):
            seed = 42 + rep
            scenario = scenario_class(**scenario_kwargs)
            sim = Simulation.from_scenario(
                scenario, config_name, seed=seed,
                density_estimator=density_estimator,
            )

            t0 = time.perf_counter()
            result = sim.run(max_steps=max_steps, max_time=max_time)
            wall_time = time.perf_counter() - t0

            row = {
                "scenario": scenario_class.__name__,
                "config": config_name,
                "seed": seed,
                "wall_time_s": wall_time,
                **result,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        fname = f"{scenario_class.__name__}_{config_name}.csv"
        df.to_csv(os.path.join(self.output_dir, fname), index=False)
        return df

    def run_fundamental_diagram(
        self,
        config_name: str,
        agent_counts: list[int] | None = None,
        n_replications: int = 5,
        corridor_length: float = 18.0,
        corridor_width: float = 1.8,
    ) -> pd.DataFrame:
        """Produce speed-density FD using periodic corridor.

        Args:
            config_name: Steering config (C1-C4).
            agent_counts: Agent counts (each gives a density point).
            n_replications: Reps per agent count.
            corridor_length: Corridor length (m).
            corridor_width: Corridor width (m). 1.8 for single-lane.

        Returns:
            DataFrame with config, n_agents, rep, density, speed columns.
        """
        from sim.scenarios.corridor import PeriodicCorridorScenario

        if agent_counts is None:
            area = corridor_length * corridor_width
            target_rhos = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            agent_counts = [max(3, int(rho * area)) for rho in target_rhos]

        rows = []
        for n_agents in agent_counts:
            for rep in range(n_replications):
                seed = 42 + rep
                scenario = PeriodicCorridorScenario(
                    n_agents=n_agents,
                    corridor_length=corridor_length,
                    corridor_width=corridor_width,
                    warmup_steps=500,
                    measure_steps=1000,
                )
                sim = Simulation.from_scenario(scenario, config_name, seed=seed)

                # Warmup
                for _ in range(scenario.warmup_steps):
                    sim.step()

                # Measure
                area = corridor_length * corridor_width
                speeds = []
                for _ in range(scenario.measure_steps):
                    sim.step()
                    active = sim.state.active
                    v = np.linalg.norm(sim.state.velocities[active], axis=1)
                    speeds.append(float(np.mean(v)))

                density = float(np.sum(sim.state.active)) / area
                mean_speed = float(np.mean(speeds))
                rows.append({
                    "config": config_name,
                    "n_agents": n_agents,
                    "rep": rep,
                    "density": density,
                    "speed": mean_speed,
                })
                print(f"  FD {config_name} n={n_agents} rep={rep}: "
                      f"rho={density:.2f}, v={mean_speed:.2f}")

        df = pd.DataFrame(rows)
        fname = f"fd_{config_name}.csv"
        df.to_csv(os.path.join(self.output_dir, fname), index=False)
        return df

    def run_scaling(
        self,
        config_name: str = "C4",
        agent_counts: list[int] | None = None,
        n_steps: int = 50,
    ) -> pd.DataFrame:
        """Measure ms/step at various agent counts.

        Args:
            config_name: Configuration to test.
            agent_counts: List of agent counts.
            n_steps: Steps per measurement.

        Returns:
            DataFrame with n_agents and ms_per_step columns.
        """
        if agent_counts is None:
            agent_counts = [50, 100, 200, 500, 1000]

        from sim.scenarios.bottleneck import BottleneckScenario

        rows = []
        for n in agent_counts:
            # Use 10x10 bottleneck (100 m^2) to keep density realistic
            # N=1000 -> 10 ped/m^2 (high but physically possible)
            scenario = BottleneckScenario(n_agents=n, exit_width=3.6)
            sim = Simulation.from_scenario(scenario, config_name)

            t0 = time.perf_counter()
            sim.run(max_steps=n_steps, max_time=1000.0)
            elapsed_ms = (time.perf_counter() - t0) / n_steps * 1000

            rows.append({"n_agents": n, "ms_per_step": elapsed_ms, "config": config_name})

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.output_dir, f"scaling_{config_name}.csv"), index=False)
        return df
