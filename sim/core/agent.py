"""AgentState dataclass for pedestrian crowd simulation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class AgentState:
    """State container for all agents in the simulation.

    Attributes:
        positions: Agent positions, shape (N, 2).
        velocities: Agent velocities, shape (N, 2).
        goals: Agent goal positions, shape (N, 2).
        radii: Agent body radii, shape (N,).
        desired_speeds: Preferred walking speeds, shape (N,).
        masses: Agent masses in kg, shape (N,).
        taus: Relaxation times in seconds, shape (N,).
        active: Boolean mask of active agents, shape (N,).
    """

    positions: np.ndarray
    velocities: np.ndarray
    goals: np.ndarray
    radii: np.ndarray
    desired_speeds: np.ndarray
    masses: np.ndarray
    taus: np.ndarray
    active: np.ndarray

    @property
    def n(self) -> int:
        """Total number of agents (active + inactive)."""
        return len(self.positions)

    @property
    def n_active(self) -> int:
        """Number of currently active agents."""
        return int(np.sum(self.active))

    @property
    def active_indices(self) -> np.ndarray:
        """Indices of active agents."""
        return np.where(self.active)[0]

    def deactivate(self, indices: np.ndarray) -> None:
        """Mark agents at given indices as inactive.

        Args:
            indices: Array or list of agent indices to deactivate.
        """
        self.active[indices] = False

    @classmethod
    def create(
        cls,
        n: int,
        spawn_area: tuple[float, float, float, float],
        goals: np.ndarray,
        seed: int = 42,
        heterogeneous: bool = True,
        speed_mean: float = 1.34,
        speed_std: float = 0.26,
        radius_mean: float = 0.25,
        radius_std: float = 0.03,
        mass: float = 80.0,
        tau: float = 0.5,
    ) -> "AgentState":
        """Factory method to create agents with randomized attributes.

        Args:
            n: Number of agents.
            spawn_area: Bounding box (x_min, x_max, y_min, y_max).
            goals: Goal position(s), shape (2,) or (N, 2).
            seed: Random seed for reproducibility.
            heterogeneous: If True, sample speeds/radii/taus from distributions.
            speed_mean: Mean desired speed (m/s).
            speed_std: Std dev of desired speed.
            radius_mean: Mean body radius (m).
            radius_std: Std dev of body radius.
            mass: Agent mass (kg).
            tau: Relaxation time (s).

        Returns:
            AgentState instance with n agents.
        """
        rng = np.random.Generator(np.random.PCG64(seed))
        x0, x1, y0, y1 = spawn_area
        pos = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])

        goals = np.asarray(goals, dtype=float)
        if goals.ndim == 1:
            goals = np.tile(goals, (n, 1))

        if heterogeneous:
            spd = np.maximum(rng.normal(speed_mean, speed_std, n), 0.5)
            rad = np.maximum(rng.normal(radius_mean, radius_std, n), 0.15)
            tau_arr = rng.uniform(0.4, 0.6, n)
        else:
            spd = np.full(n, speed_mean)
            rad = np.full(n, radius_mean)
            tau_arr = np.full(n, tau)

        return cls(
            pos,
            np.zeros((n, 2)),
            goals,
            rad,
            spd,
            np.full(n, mass),
            tau_arr,
            np.ones(n, dtype=bool),
        )
