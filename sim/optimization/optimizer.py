"""Barrier placement optimizer for crowd safety."""

import numpy as np
from scipy.optimize import minimize

from sim.core.simulation import Simulation
from sim.core.world import Wall
from sim.optimization.barrier import BarrierConfig


class BarrierOptimizer:
    """Optimize barrier placement to minimize evacuation time.

    Args:
        scenario_class: Scenario class to use.
        config_name: Steering config (C1-C4).
        n_agents: Agents per run.
        n_reps: Replications per evaluation.
        max_time: Max simulation time per run.
    """

    def __init__(
        self,
        scenario_class,
        config_name: str = "C4",
        n_agents: int = 100,
        n_reps: int = 3,
        max_time: float = 60.0,
    ):
        self.scenario_class = scenario_class
        self.config_name = config_name
        self.n_agents = n_agents
        self.n_reps = n_reps
        self.max_time = max_time
        self.history: list[dict] = []

    def evaluate(self, params: np.ndarray) -> float:
        """Evaluate a barrier configuration.

        Args:
            params: [x, y, length, angle] for the barrier.

        Returns:
            Cost (mean evacuation time; lower is better).
        """
        barrier = BarrierConfig(
            x=float(params[0]),
            y=float(params[1]),
            length=float(params[2]),
            angle=float(params[3]),
        )
        obstacle = barrier.to_obstacle()

        costs = []
        for rep in range(self.n_reps):
            seed = 42 + rep
            scenario = self.scenario_class(n_agents=self.n_agents)
            sim = Simulation.from_scenario(scenario, self.config_name, seed=seed)
            # Add barrier as wall segments (obstacle edges)
            verts = obstacle.vertices
            for k in range(len(verts)):
                wall = Wall(verts[k].copy(), verts[(k + 1) % len(verts)].copy())
                sim.world.walls.append(wall)

            result = sim.run(max_time=self.max_time)
            # Cost: evacuation time (cap inf to max_time, penalize incomplete)
            evac = result["evacuation_time"]
            if np.isinf(evac):
                evac = self.max_time
            if result["agents_exited"] < self.n_agents:
                evac += (self.n_agents - result["agents_exited"]) * 10.0
            costs.append(evac)

        mean_cost = float(np.mean(costs))
        self.history.append({"params": params.tolist(), "cost": mean_cost})
        return mean_cost

    def optimize(
        self,
        method: str = "nelder-mead",
        max_evals: int = 200,
    ) -> dict:
        """Run optimization.

        Args:
            method: "nelder-mead" or "cma-es".
            max_evals: Maximum function evaluations.

        Returns:
            Dict with best params, cost, and history.
        """
        bounds = BarrierConfig.bounds()
        x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])
        self.history = []

        if method == "cma-es":
            return self._optimize_cma(x0, bounds, max_evals)
        else:
            return self._optimize_nelder_mead(x0, bounds, max_evals)

    def _optimize_nelder_mead(self, x0, bounds, max_evals):
        """Nelder-Mead with bounds enforcement."""
        bounds_arr = np.array(bounds)

        def bounded_eval(params):
            clipped = np.clip(params, bounds_arr[:, 0], bounds_arr[:, 1])
            return self.evaluate(clipped)

        res = minimize(
            bounded_eval, x0, method="Nelder-Mead",
            options={"maxfev": max_evals, "adaptive": True},
        )
        best = np.clip(res.x, bounds_arr[:, 0], bounds_arr[:, 1])
        return {"params": best.tolist(), "cost": float(res.fun), "history": self.history}

    def _optimize_cma(self, x0, bounds, max_evals):
        """CMA-ES optimization (falls back to Nelder-Mead if cma not installed)."""
        try:
            import cma
            bounds_lo = [b[0] for b in bounds]
            bounds_hi = [b[1] for b in bounds]
            opts = {
                "maxfevals": max_evals,
                "bounds": [bounds_lo, bounds_hi],
                "verbose": -9,
                "popsize": 15,
            }
            es = cma.CMAEvolutionStrategy(x0.tolist(), 1.0, opts)
            while not es.stop():
                solutions = es.ask()
                fitnesses = [self.evaluate(np.array(s)) for s in solutions]
                es.tell(solutions, fitnesses)
            best = es.result.xbest
            return {"params": list(best), "cost": float(es.result.fbest), "history": self.history}
        except ImportError:
            return self._optimize_nelder_mead(x0, bounds, max_evals)

    def sweep(self, param_index: int = 0, n_points: int = 10) -> list[dict]:
        """1D parameter sweep.

        Args:
            param_index: Which parameter to sweep (0=x, 1=y, 2=length, 3=angle).
            n_points: Number of sweep points.

        Returns:
            List of {param_value, cost} dicts.
        """
        bounds = BarrierConfig.bounds()
        base = np.array([(lo + hi) / 2 for lo, hi in bounds])
        lo, hi = bounds[param_index]
        values = np.linspace(lo, hi, n_points)

        results = []
        for val in values:
            params = base.copy()
            params[param_index] = val
            cost = self.evaluate(params)
            results.append({"param_value": float(val), "cost": cost})
        return results
