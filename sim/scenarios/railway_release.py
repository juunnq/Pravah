"""Timed-release surge mechanism for the railway-platform scenario (Phase 2).

Releases agents into the holding area in timed batches per two frozen schedules
(frozen at design time), both totalling N=150:

  BASELINE_SCHEDULE     = [(0, 50), (3, 50), (6, 50)]                       (crush)
  INTERVENTION_SCHEDULE = [(0, 30), (20, 30), (40, 30), (60, 30), (80, 30)] (safe)

The t=0 batch is the simulation's initial agent set (built via the scenario); later
batches are injected by ``Simulation.inject_agents``. This module owns its own step
loop (``run_surge``) and does NOT use or alter ``Simulation.run``'s corridor path.

Phase 2 is the release mechanism ONLY — no density detector, no risk thresholds,
no FOB-zone measurement.
"""

from dataclasses import dataclass

import numpy as np

from sim.core.simulation import Simulation


@dataclass(frozen=True)
class ReleaseSchedule:
    """An ordered set of timed release batches.

    Args:
        batches: List of (time_s, n_agents) tuples. Times must be non-negative and
            strictly ascending (sorted, no duplicate times); each n_agents > 0.

    Raises:
        ValueError: If times are negative, not sorted ascending, duplicated, or any
            batch count is non-positive, or the list is empty.
    """

    batches: tuple[tuple[float, int], ...]

    def __post_init__(self):
        if len(self.batches) == 0:
            raise ValueError("ReleaseSchedule must have at least one batch")
        prev_t = -1.0
        for t, n in self.batches:
            if t < 0:
                raise ValueError(f"batch time must be non-negative, got {t}")
            if t <= prev_t:
                raise ValueError(
                    f"batch times must be sorted strictly ascending, got {t} after {prev_t}"
                )
            if n <= 0:
                raise ValueError(f"batch count must be positive, got {n}")
            prev_t = t

    @property
    def total(self) -> int:
        """Total agents released across all batches."""
        return int(sum(n for _, n in self.batches))

    def batches_due(self, t: float) -> list[tuple[float, int]]:
        """Batches whose release time is at or before ``t``.

        Args:
            t: Query time (s).

        Returns:
            The sub-list of (time_s, n_agents) batches with time <= t, in order.
        """
        return [(bt, bn) for (bt, bn) in self.batches if bt <= t]

    def n_due(self, t: float) -> int:
        """Cumulative agent count released at or before time ``t``.

        Args:
            t: Query time (s).

        Returns:
            Sum of n_agents over all batches with time <= t.
        """
        return int(sum(bn for (bt, bn) in self.batches if bt <= t))

    def release_steps(self, dt: float) -> list[int]:
        """Step index at which each batch releases, ``round(time/dt)`` per batch.

        Args:
            dt: Simulation timestep (s).

        Returns:
            List of integer step indices, one per batch (same order as ``batches``).
        """
        return [int(round(bt / dt)) for (bt, _) in self.batches]


# Frozen schedules; equal-N is a HARD requirement.
BASELINE_SCHEDULE = ReleaseSchedule(((0.0, 50), (3.0, 50), (6.0, 50)))
INTERVENTION_SCHEDULE = ReleaseSchedule(
    ((0.0, 30), (20.0, 30), (40.0, 30), (60.0, 30), (80.0, 30))
)

# Calibrated Weidmann speed-density coefficients from the SIMULTECH 2026 paper
# (Table 1: gamma*=0.833, rho_max*=5.98; FD RMSE 0.235 -> 0.099 m/s vs 4776 FZJ
# points). The engine's desired-force defaults are the UNCALIBRATED textbook
# values (1.913, 5.4) and config/params.yaml has no weidmann section, so the
# railway path must inject these or it runs on uncalibrated physics.
RAILWAY_PARAM_OVERRIDES: dict[str, float] = {
    "weidmann_gamma": 0.833,
    "weidmann_rho_max": 5.98,
}

assert BASELINE_SCHEDULE.total == 150, "BASELINE_SCHEDULE must total 150"
assert INTERVENTION_SCHEDULE.total == 150, "INTERVENTION_SCHEDULE must total 150"


def _batch_seeds(master_seed: int, n_batches: int) -> list[int]:
    """Derive deterministic per-batch integer seeds from a master seed.

    Args:
        master_seed: Master random seed.
        n_batches: Number of batches.

    Returns:
        One non-negative int seed per batch, reproducible for a given master_seed.
    """
    children = np.random.SeedSequence(master_seed).spawn(n_batches)
    return [int(child.generate_state(1)[0]) for child in children]


def run_surge(
    scenario,
    schedule: ReleaseSchedule,
    config: str = "C4",
    seed: int = 42,
    density_estimator=None,
    max_steps: int = 20000,
    max_time: float = 300.0,
    param_overrides: dict | None = None,
    log_positions: bool = False,
) -> Simulation:
    """Run a railway scenario with a timed agent-release schedule.

    The first (t=0) batch is the simulation's initial state; later batches are
    injected into the scenario's holding area on schedule. Owns its own step loop;
    does not call ``Simulation.run``. Computes no density/risk metrics (Phase 2).

    Applies the calibrated Weidmann coefficients (``RAILWAY_PARAM_OVERRIDES``) by
    default; explicit ``param_overrides`` entries win over them.

    Args:
        scenario: A railway scenario exposing ``spawn_area``, ``goal``, ``build``,
            ``is_complete``, and a settable ``n_agents`` attribute. Its ``n_agents``
            is set to the t=0 batch size for this run.
        schedule: The release schedule (the first batch is the initial population).
        config: Steering config name (e.g. "C1", "C4").
        seed: Master random seed (initial state + per-batch injection seeds).
        density_estimator: Optional density estimator passed to ``from_scenario``.
        max_steps: Hard cap on simulation steps.
        max_time: Hard cap on simulation time (s).
        param_overrides: Optional params.yaml overrides; merged on top of the
            calibrated Weidmann defaults (explicit entries win).
        log_positions: When True, enable the engine's opt-in per-step position
            logging (``Simulation.log_positions``) for post-hoc analysis.

    Returns:
        The Simulation after the run, with ``metrics_log`` intact.
    """
    batches = schedule.batches
    initial_n = batches[0][1]

    merged_overrides = dict(RAILWAY_PARAM_OVERRIDES)
    if param_overrides:
        merged_overrides.update(param_overrides)

    # The t=0 batch is the initial agent set.
    scenario.n_agents = initial_n
    sim = Simulation.from_scenario(
        scenario, config, seed=seed, density_estimator=density_estimator,
        param_overrides=merged_overrides,
    )
    sim.log_positions = log_positions

    dt = sim.params.get("dt", 0.01)
    release_steps = schedule.release_steps(dt)
    batch_seeds = _batch_seeds(seed, len(batches))

    released = {0}  # batch 0 is the initial state, never injected

    while sim.step_count < max_steps and sim.time < max_time:
        if scenario.is_complete(sim.state, sim.time):
            break
        # Inject any due-but-unreleased batch BEFORE stepping.
        for bi in range(1, len(batches)):
            if bi not in released and sim.step_count >= release_steps[bi]:
                sim.inject_agents(
                    batches[bi][1],
                    seed=batch_seeds[bi],
                    spawn_area=scenario.spawn_area,
                    goal=scenario.goal,
                    avoid_overlap=True,
                    speed_mean=getattr(scenario, "speed_mean", 1.34),
                )
                released.add(bi)
        sim.step()

    return sim
