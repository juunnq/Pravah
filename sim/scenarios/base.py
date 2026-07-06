"""Abstract base class for simulation scenarios."""

from abc import ABC, abstractmethod

from sim.core.agent import AgentState
from sim.core.world import World


class Scenario(ABC):
    """Base class for all simulation scenarios.

    Subclasses define the world geometry and agent placement.
    """

    @abstractmethod
    def build(self, seed: int = 42) -> tuple[World, AgentState]:
        """Construct the world and initial agent state.

        Args:
            seed: Random seed for agent placement.

        Returns:
            Tuple of (World, AgentState).
        """
        ...

    @abstractmethod
    def is_complete(self, agent_state: AgentState, time: float) -> bool:
        """Check if the scenario is finished.

        Args:
            agent_state: Current agent state.
            time: Current simulation time.

        Returns:
            True if the scenario is complete.
        """
        ...
