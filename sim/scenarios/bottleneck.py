"""Bottleneck scenario: 10x10m room with variable-width exit on right wall."""

import numpy as np

from sim.core.agent import AgentState
from sim.core.world import Wall, World
from sim.scenarios.base import Scenario


class BottleneckScenario(Scenario):
    """10x10m room with a gap on the right wall.

    Args:
        n_agents: Number of agents.
        exit_width: Width of exit gap (m), centered at y=5.
    """

    def __init__(self, n_agents: int = 100, exit_width: float = 1.2):
        self.n_agents = n_agents
        self.exit_width = exit_width

    def build(self, seed: int = 42) -> tuple[World, AgentState]:
        """Build bottleneck world and agents.

        Walls: 10x10 room. Right wall has gap centered at y=5.
        Spawn: x in [1, 8], y in [1, 9].
        Goal: (11, 5).
        """
        half = self.exit_width / 2.0
        walls = [
            Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0])),     # bottom
            Wall(np.array([0.0, 10.0]), np.array([0.0, 0.0])),     # left
            Wall(np.array([10.0, 10.0]), np.array([0.0, 10.0])),   # top
            # right wall with gap
            Wall(np.array([10.0, 0.0]), np.array([10.0, 5.0 - half])),
            Wall(np.array([10.0, 5.0 + half]), np.array([10.0, 10.0])),
        ]
        world = World(walls)

        state = AgentState.create(
            self.n_agents,
            spawn_area=(1.0, 8.0, 1.0, 9.0),
            goals=np.array([11.0, 5.0]),
            seed=seed,
        )
        return world, state

    def is_complete(self, agent_state: AgentState, time: float) -> bool:
        """Complete when all agents have exited."""
        return agent_state.n_active == 0
