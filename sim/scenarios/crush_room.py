"""Crush room scenario: packed room with a single narrow exit.

Classical crush geometry (concerts, Hajj, nightclub exits): agents packed
into a rectangular room converge on one narrow exit, producing sustained
high-density conditions at the exit throat. Used to validate the crush
regime activation at densities > 5.5 ped/m^2.
"""

import numpy as np

from sim.core.agent import AgentState
from sim.core.world import Wall, World
from sim.scenarios.base import Scenario


class CrushRoomScenario(Scenario):
    """Packed room with a single narrow exit centred on the right wall.

    Geometry: `width` x `height` room (default 5 x 5 m).
    Exit: centred on the right wall (x=width), of width `exit_width`.
    Goal: just outside the exit at (width+1, height/2).

    Args:
        n_agents: Number of agents packed into the room.
        width: Room width (m).
        height: Room height (m).
        exit_width: Exit gap width (m).
    """

    def __init__(
        self,
        n_agents: int = 300,
        width: float = 5.0,
        height: float = 5.0,
        exit_width: float = 0.6,
    ):
        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.exit_width = exit_width

    def build(self, seed: int = 42) -> tuple[World, AgentState]:
        """Build crush room world and agents.

        Spawn: x in [0.3, width-0.3], y in [0.3, height-0.3].
        Goal: (width+1, height/2).
        """
        half = self.exit_width / 2.0
        y_gap_bottom = self.height / 2.0 - half
        y_gap_top = self.height / 2.0 + half

        walls = [
            # bottom wall
            Wall(np.array([0.0, 0.0]), np.array([self.width, 0.0])),
            # top wall
            Wall(np.array([0.0, self.height]), np.array([self.width, self.height])),
            # left wall
            Wall(np.array([0.0, 0.0]), np.array([0.0, self.height])),
            # right wall with exit gap
            Wall(np.array([self.width, 0.0]), np.array([self.width, y_gap_bottom])),
            Wall(np.array([self.width, y_gap_top]), np.array([self.width, self.height])),
        ]
        world = World(walls)

        state = AgentState.create(
            self.n_agents,
            spawn_area=(0.3, self.width - 0.3, 0.3, self.height - 0.3),
            goals=np.array([self.width + 1.0, self.height / 2.0]),
            seed=seed,
            heterogeneous=True,
        )
        return world, state

    def is_complete(self, agent_state: AgentState, time: float) -> bool:
        """Complete when all agents have exited."""
        return agent_state.n_active == 0
