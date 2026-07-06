"""Funnel scenario: 10m-wide entrance tapering to 3m exit over 15m."""

import numpy as np

from sim.core.agent import AgentState
from sim.core.world import Wall, World
from sim.scenarios.base import Scenario


class FunnelScenario(Scenario):
    """Funnel with angled walls tapering from 10m to a configurable exit width.

    Top wall:    (0, 10) -> (15, 5 + exit_width/2).
    Bottom wall: (0, 0)  -> (15, 5 - exit_width/2).
    Left wall:   (0, 0)  -> (0, 10).

    Args:
        n_agents: Number of agents.
        exit_width: Exit gap at x=15 (m). Default 3.0. For crush scenarios
            use 0.8-1.0m to force sustained bottleneck density.
    """

    def __init__(self, n_agents: int = 400, exit_width: float = 3.0):
        self.n_agents = n_agents
        self.exit_width = exit_width

    def build(self, seed: int = 42) -> tuple[World, AgentState]:
        """Build funnel world and agents.

        Spawn: x in [0.5, 5], y in [1, 9].
        Goal: (16, 5).
        """
        half = self.exit_width / 2.0
        y_bottom = 5.0 - half  # bottom wall endpoint at exit
        y_top = 5.0 + half     # top wall endpoint at exit
        walls = [
            Wall(np.array([0.0, 0.0]), np.array([15.0, y_bottom])),   # bottom (angled)
            Wall(np.array([0.0, 10.0]), np.array([15.0, y_top])),     # top (angled)
            Wall(np.array([0.0, 0.0]), np.array([0.0, 10.0])),        # left
            # Exit frame: short vertical segments extending from funnel endpoints
            Wall(np.array([15.0, y_bottom]), np.array([15.0, max(0.0, y_bottom - 0.5)])),
            Wall(np.array([15.0, y_top]), np.array([15.0, min(10.0, y_top + 0.5)])),
        ]
        world = World(walls)

        state = AgentState.create(
            self.n_agents,
            spawn_area=(0.5, 5.0, 1.0, 9.0),
            goals=np.array([16.0, 5.0]),
            seed=seed,
        )
        return world, state

    def is_complete(self, agent_state: AgentState, time: float) -> bool:
        """Complete when all agents have exited."""
        return agent_state.n_active == 0
