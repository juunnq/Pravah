"""Main simulation loop: neighbor search, force computation, integration, metrics."""

import numpy as np
from scipy.spatial import KDTree

from sim.core.agent import AgentState
from sim.core.helpers import clamp_speed
from sim.core.integrator import EulerIntegrator
from sim.core.world import World
from sim.steering.base import SteeringModel
from sim.steering.desired import compute_desired_force
from sim.steering.walls import WallForces


class Simulation:
    """Orchestrates the simulation loop.

    Each step: KDTree neighbor search -> density estimation -> force computation
    -> integration -> speed clamping -> goal deactivation -> metrics recording.

    Args:
        world: World geometry (walls, obstacles).
        agent_state: Initial agent state.
        steering_model: Steering model for force computation (None = desired only).
        integrator: Numerical integrator (default: EulerIntegrator).
        params: Dict of simulation parameters.
        density_estimator: Optional density estimator. When None (default),
            uses grid-counting density (neighbor count / pi*R^2). Pass a
            VoronoiDensityEstimator or KDEDensityEstimator to override.
    """

    def __init__(
        self,
        world: World,
        agent_state: AgentState,
        steering_model: SteeringModel | None,
        integrator: EulerIntegrator | None = None,
        params: dict | None = None,
        density_estimator=None,
        log_positions: bool = False,
        log_collisions: bool = False,
        velocity_noise_std: float = 0.0,
        log_forces: bool = False,
        seed: int = 42,
    ):
        self.world = world
        self.state = agent_state
        self.steering = steering_model
        self.integrator = integrator or EulerIntegrator()
        self.params = params or {
            "dt": 0.01,
            "neighbor_radius": 3.0,
            "max_time": 300.0,
            "goal_reached_dist": 0.5,
        }
        self.density_estimator = density_estimator
        self.time = 0.0
        self.step_count = 0
        self.metrics_log: list[dict] = []
        self.periodic_length: float | None = None  # set for periodic corridors

        # Opt-in logging (default off — preserves bit-identical default path)
        self.log_positions = log_positions
        self.log_collisions = log_collisions
        self._position_log: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self._collision_log: list[tuple[float, int, int, float, float, float, float]] = []

        # R3.1: velocity noise for symmetry-breaking control (default 0 = off)
        self.velocity_noise_std = velocity_noise_std
        self._rng = np.random.Generator(np.random.PCG64(seed))

        # R3.2: per-component force logging (default off)
        self.log_forces = log_forces
        self._force_log: list[dict] = []

    def step(self) -> dict:
        """Execute one simulation timestep.

        Returns:
            Dict with step metrics (time, n_active, mean_speed).
        """
        active = self.state.active_indices
        if len(active) == 0:
            return {"time": self.time, "n_active": 0}

        # 1. KDTree neighbor search (active agents only, self excluded)
        active_idx = self.state.active_indices
        active_pos = self.state.positions[active_idx]
        N_active = len(active_idx)
        # Map from local (active-only) index to global index
        neighbor_lists: list[list[int]] = [[] for _ in range(self.state.n)]

        if self.periodic_length is not None:
            L = self.periodic_length
            ghosts_r = active_pos.copy(); ghosts_r[:, 0] += L
            ghosts_l = active_pos.copy(); ghosts_l[:, 0] -= L
            extended = np.vstack([active_pos, ghosts_r, ghosts_l])
            tree = KDTree(extended)
            raw_nbs = tree.query_ball_point(active_pos, r=self.params["neighbor_radius"])
            for local_i, nbrs in enumerate(raw_nbs):
                global_i = active_idx[local_i]
                real = list({active_idx[j % N_active] for j in nbrs} - {global_i})
                neighbor_lists[global_i] = real
        else:
            tree = KDTree(active_pos)
            raw_nbs = tree.query_ball_point(active_pos, r=self.params["neighbor_radius"])
            for local_i, nbrs in enumerate(raw_nbs):
                global_i = active_idx[local_i]
                neighbor_lists[global_i] = [active_idx[j] for j in nbrs if active_idx[j] != global_i]

        # 2. Density estimation: custom estimator or default grid count
        if self.density_estimator is not None:
            # Custom estimator (Voronoi/KDE): compute on active agents only,
            # then scatter back to full-size array (inactive agents get 0).
            active_densities = self.density_estimator.estimate(active_pos)
            densities = np.zeros(self.state.n, dtype=float)
            densities[active_idx] = active_densities
        else:
            r = self.params["neighbor_radius"]
            area = np.pi * r * r
            densities = np.array(
                [len(n) / area for n in neighbor_lists], dtype=float
            )

        # 3. Compute forces
        if self.steering is not None:
            forces = self.steering.compute_forces(
                self.state, neighbor_lists, self.world.walls, densities
            )
            # 3b. Force logging (R3.2, default off, every 10th step)
            if (self.log_forces
                    and self.step_count % 10 == 0
                    and hasattr(self.steering, 'compute_forces_decomposed')):
                _, mags = self.steering.compute_forces_decomposed(
                    self.state, neighbor_lists, self.world.walls, densities
                )
                for ai in active_idx:
                    self._force_log.append({
                        "t": self.time + self.params.get("dt", 0.01),
                        "agent_id": int(ai),
                        "density": float(densities[ai]),
                        "mag_des": float(mags["des"][ai]),
                        "mag_sfm": float(mags["sfm"][ai]),
                        "mag_ttc": float(mags["ttc"][ai]),
                        "mag_orca": float(mags["orca"][ai]),
                    })
        else:
            forces = compute_desired_force(
                self.state.positions,
                self.state.velocities,
                self.state.goals,
                self.state.desired_speeds,
                self.state.masses,
                self.state.taus,
                local_densities=densities,
            )
            if self.world.walls:
                forces += WallForces().compute_wall_forces(self.state, self.world.walls)

        # 4. Integrate
        dt = self.params["dt"]
        new_pos, new_vel = self.integrator.integrate(
            self.state.positions,
            self.state.velocities,
            forces,
            self.state.masses,
            dt,
        )

        # 5. Clamp speed to 2x desired
        max_speeds = 2.0 * self.state.desired_speeds
        new_vel = clamp_speed(new_vel, max_speeds)

        # 5b. Velocity noise injection (R3.1, default 0 = off, no-op)
        if self.velocity_noise_std > 0:
            noise = self._rng.normal(0, self.velocity_noise_std, new_vel.shape)
            active_mask = self.state.active
            new_vel[active_mask] += noise[active_mask]
            new_vel = clamp_speed(new_vel, max_speeds)  # re-clamp after noise

        # 6. Update state
        self.state.positions = new_pos
        self.state.velocities = new_vel

        # 6b. Periodic boundary: wrap x-coordinate, keep goals ahead
        if self.periodic_length is not None:
            L = self.periodic_length
            self.state.positions[:, 0] = self.state.positions[:, 0] % L
            self.state.goals[:, 0] = self.state.positions[:, 0] + 5.0
            reached = np.array([], dtype=int)  # never deactivate
        else:
            # 7. Deactivate agents that reached their goal
            dists_to_goal = np.linalg.norm(
                self.state.goals - self.state.positions, axis=1
            )
            reached = np.where(
                self.state.active & (dists_to_goal < self.params["goal_reached_dist"])
            )[0]
            self.state.deactivate(reached)

        # 8. Record metrics
        self.time += dt
        self.step_count += 1
        active_now = self.state.active_indices
        n_exited_this_step = len(reached)

        # 8a. Position logging (opt-in, default off)
        if self.log_positions and len(active_now) > 0:
            self._position_log.append((
                self.time,
                active_now.copy(),
                self.state.positions[active_now].copy(),
                self.state.velocities[active_now].copy(),
            ))

        if len(active_now) > 0:
            mean_speed = float(
                np.mean(np.linalg.norm(self.state.velocities[active_now], axis=1))
            )
            max_density = float(np.max(densities[active_now]))
        else:
            mean_speed = 0.0
            max_density = 0.0

        # Collision count: pairs with distance < sum of radii
        collision_count = 0
        for i in active_now:
            for j in neighbor_lists[i]:
                if j > i and self.state.active[j]:
                    d = np.linalg.norm(self.state.positions[i] - self.state.positions[j])
                    if d < self.state.radii[i] + self.state.radii[j]:
                        collision_count += 1
                        # 8b. Collision logging (opt-in, default off)
                        if self.log_collisions:
                            self._collision_log.append((
                                self.time, i, j,
                                float(self.state.positions[i, 0]),
                                float(self.state.positions[i, 1]),
                                float(self.state.positions[j, 0]),
                                float(self.state.positions[j, 1]),
                            ))

        metrics = {
            "time": self.time,
            "n_active": self.state.n_active,
            "mean_speed": mean_speed,
            "max_density": max_density,
            "collision_count": collision_count,
            "agents_exited_step": n_exited_this_step,
        }
        self.metrics_log.append(metrics)
        return metrics

    def inject_agents(
        self,
        n: int,
        seed: int,
        spawn_area: tuple[float, float, float, float] | None = None,
        goal: np.ndarray | None = None,
        avoid_overlap: bool = False,
        speed_mean: float = 1.34,
    ) -> None:
        """Inject new agents and append them to the active state.

        Backward-compatible: when both ``spawn_area`` and ``goal`` are None this
        reproduces the original corridor injection exactly (same spawn box, same
        goal, same RNG draws), so existing callers are unaffected.

        Args:
            n: Number of agents to inject.
            seed: Random seed for placement (deterministic PCG64 stream).
            spawn_area: Optional (x_min, x_max, y_min, y_max) box. None ⇒ the
                default corridor box (0.3, 2.0, 0.3, 3.3).
            goal: Optional goal position, shape (2,). None ⇒ the default corridor
                goal (26.0, 1.8).
            avoid_overlap: When True, reject candidate spawn positions within
                (r_new + r_existing + margin) of any active agent (and of already
                placed new agents), resampling up to a bounded number of attempts
                before best-effort placement. Default False preserves behavior.
            speed_mean: Mean desired speed (m/s) for the injected agents. Default
                1.34 (calm walking) preserves prior behavior.
        """
        if spawn_area is None and goal is None:
            # Original corridor injection (unchanged default code path).
            spawn_area = (0.3, 2.0, 0.3, 3.3)
            goal = np.array([26.0, 1.8])

        new = AgentState.create(
            n,
            spawn_area=spawn_area,
            goals=goal,
            seed=seed,
            heterogeneous=True,
            speed_mean=speed_mean,
        )

        if avoid_overlap:
            # Rejection-sample new positions away from existing active agents and
            # from one another. Deterministic in `seed`.
            x0, x1, y0, y1 = spawn_area
            margin = 0.05
            max_attempts = 50
            s = self.state
            if np.any(s.active):
                ex_pos = s.positions[s.active]
                ex_rad = s.radii[s.active]
            else:
                ex_pos = np.empty((0, 2))
                ex_rad = np.empty((0,))
            rng = np.random.Generator(np.random.PCG64(seed))
            placed = np.empty((n, 2))
            new_rad = new.radii
            for i in range(n):
                ri = new_rad[i]
                cand = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                for _ in range(max_attempts):
                    ok = True
                    if len(ex_pos):
                        d = np.linalg.norm(ex_pos - cand, axis=1)
                        if np.any(d < ex_rad + ri + margin):
                            ok = False
                    if ok and i > 0:
                        d2 = np.linalg.norm(placed[:i] - cand, axis=1)
                        if np.any(d2 < new_rad[:i] + ri + margin):
                            ok = False
                    if ok:
                        break
                    cand = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                placed[i] = cand
            new.positions = placed

        # Append to existing state
        s = self.state
        s.positions = np.vstack([s.positions, new.positions])
        s.velocities = np.vstack([s.velocities, new.velocities])
        s.goals = np.vstack([s.goals, new.goals])
        s.radii = np.concatenate([s.radii, new.radii])
        s.desired_speeds = np.concatenate([s.desired_speeds, new.desired_speeds])
        s.masses = np.concatenate([s.masses, new.masses])
        s.taus = np.concatenate([s.taus, new.taus])
        s.active = np.concatenate([s.active, new.active])

    def run(
        self, max_steps: int = 10000, max_time: float | None = None
    ) -> dict:
        """Run the simulation until completion.

        Args:
            max_steps: Maximum number of timesteps.
            max_time: Maximum simulation time in seconds.

        Returns:
            Summary dict with n_steps, time, agents_exited, mean_speed.
        """
        if max_time is None:
            max_time = self.params.get("max_time", 300.0)

        # Check if scenario uses continuous injection
        scenario = getattr(self, '_scenario', None)
        inj_rate = getattr(scenario, 'injection_rate', 0) if scenario else 0
        inj_accum = 0.0
        inj_seed_counter = 10000

        while (
            self.step_count < max_steps
            and self.time < max_time
            and self.state.n_active > 0
        ):
            # Inject agents if configured
            if inj_rate > 0:
                dt = self.params.get("dt", 0.01)
                inj_accum += inj_rate * dt
                if inj_accum >= 1.0:
                    n_inject = int(inj_accum)
                    inj_accum -= n_inject
                    self.inject_agents(n_inject, seed=inj_seed_counter)
                    inj_seed_counter += 1

            self.step()

            # Deactivate agents past corridor exit
            if inj_rate > 0:
                past_exit = np.where(
                    self.state.active & (self.state.positions[:, 0] > 24.5)
                )[0]
                self.state.deactivate(past_exit)

        return self._compile_results()

    @classmethod
    def from_scenario(
        cls,
        scenario,
        config_name: str = "C1",
        seed: int = 42,
        param_overrides: dict | None = None,
        density_estimator=None,
    ) -> "Simulation":
        """Build a Simulation from a scenario object and config name.

        Args:
            scenario: Scenario with a build(seed) method returning (World, AgentState).
            config_name: One of C1-C4.
            seed: Random seed.
            param_overrides: Optional dict to override params.yaml values.
            density_estimator: Optional density estimator override. When None,
                uses grid-counting density (default). Pass a VoronoiDensityEstimator
                to improve local-hotspot detection (e.g. for crush scenarios).

        Returns:
            Configured Simulation instance.
        """
        import yaml

        from sim.experiments.configs import get_config, get_param_overrides
        from sim.steering.hybrid import HybridSteeringModel

        world, agent_state = scenario.build(seed=seed)
        with open("config/params.yaml") as f:
            params = yaml.safe_load(f)
        flat: dict = {}
        for v in params.values():
            if isinstance(v, dict):
                flat.update(v)
        # Apply D-config overrides first, then explicit overrides
        flat.update(get_param_overrides(config_name))
        if param_overrides:
            flat.update(param_overrides)
        config = get_config(config_name)
        steering = HybridSteeringModel(config, flat)
        sim = cls(world, agent_state, steering, EulerIntegrator(), flat,
                  density_estimator=density_estimator, seed=seed)
        sim._scenario = scenario
        sim.periodic_length = getattr(scenario, 'periodic_length', None)
        return sim

    def write_logs(self, trajectory_path: str | None = None,
                   collision_path: str | None = None,
                   force_path: str | None = None) -> None:
        """Write accumulated logs to parquet files.

        Args:
            trajectory_path: Output path for trajectory parquet.
            collision_path: Output path for collision parquet.
            force_path: Output path for force-component parquet.
        """
        import os
        import pandas as pd

        if trajectory_path and self._position_log:
            os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
            rows = []
            for t, agent_ids, pos, vel in self._position_log:
                for k, aid in enumerate(agent_ids):
                    rows.append((t, int(aid), pos[k, 0], pos[k, 1],
                                 vel[k, 0], vel[k, 1]))
            df = pd.DataFrame(rows, columns=["t", "agent_id", "x", "y", "vx", "vy"])
            df.to_parquet(trajectory_path, index=False, engine="pyarrow")

        if collision_path and self._collision_log:
            os.makedirs(os.path.dirname(collision_path), exist_ok=True)
            df = pd.DataFrame(self._collision_log,
                              columns=["t", "i", "j", "x_i", "y_i", "x_j", "y_j"])
            df.to_parquet(collision_path, index=False, engine="pyarrow")

        if force_path and self._force_log:
            os.makedirs(os.path.dirname(force_path), exist_ok=True)
            df = pd.DataFrame(self._force_log)
            df.to_parquet(force_path, index=False, engine="pyarrow")

    def _compile_results(self) -> dict:
        """Compile summary statistics from the simulation run.

        Returns:
            Dict with the project's standard metrics set.
        """
        if not self.metrics_log:
            return {
                "n_steps": 0, "evacuation_time": 0.0, "mean_speed": 0.0,
                "max_density": 0.0, "collision_count": 0, "flow_rate": 0.0,
                "agents_exited": 0, "mean_risk": 0.0, "max_risk": 0.0,
                "time_above_critical": 0.0,
            }

        agents_exited = self.state.n - self.state.n_active
        evac_time = self.time if self.state.n_active == 0 else float('inf')

        mean_speed = float(np.mean([m["mean_speed"] for m in self.metrics_log]))
        max_density = float(np.max([m["max_density"] for m in self.metrics_log]))
        total_collisions = int(np.sum([m["collision_count"] for m in self.metrics_log]))

        dt = self.params.get("dt", 0.01)
        flow_rate = agents_exited / max(self.time, dt)

        # Risk and time above critical (density > 5.5)
        critical_threshold = self.params.get("rho_crit", 5.5)
        time_above_critical = sum(
            dt for m in self.metrics_log if m["max_density"] > critical_threshold
        )

        return {
            "n_steps": self.step_count,
            "evacuation_time": evac_time,
            "mean_speed": mean_speed,
            "max_density": max_density,
            "collision_count": total_collisions,
            "flow_rate": flow_rate,
            "agents_exited": agents_exited,
            "mean_risk": 0.0,  # populated by runner when density estimators used
            "max_risk": 0.0,
            "time_above_critical": time_above_critical,
        }
