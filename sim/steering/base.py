"""Abstract base class for steering models."""

from abc import ABC, abstractmethod

import numpy as np

from sim.core.agent import AgentState
from sim.core.world import Wall


class SteeringModel(ABC):
    """Base class that all steering models must implement."""

    @abstractmethod
    def compute_forces(
        self,
        agent_state: AgentState,
        neighbor_lists: list[list[int]],
        walls: list[Wall],
        local_densities: np.ndarray,
    ) -> np.ndarray:
        """Compute steering forces for all agents.

        Args:
            agent_state: Current state of all agents.
            neighbor_lists: Per-agent list of neighbor indices.
            walls: List of wall segments in the world.
            local_densities: Per-agent local density estimate, shape (N,).

        Returns:
            Forces array of shape (N, 2).
        """
        ...
