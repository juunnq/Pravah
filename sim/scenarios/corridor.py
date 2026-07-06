"""Corridor scenario: 10x3.6m unidirectional flow."""

import numpy as np

from sim.core.agent import AgentState
from sim.core.world import Wall, World
from sim.scenarios.base import Scenario


class CorridorScenario(Scenario):
    """10x3.6m corridor with agents walking left to right.

    Args:
        n_agents: Number of agents.
        heterogeneous: Whether to use heterogeneous agent parameters.
    """

    def __init__(self, n_agents: int = 50, heterogeneous: bool = True):
        self.n_agents = n_agents
        self.heterogeneous = heterogeneous

    def build(self, seed: int = 42) -> tuple[World, AgentState]:
        """Build corridor world and agents.

        Walls: 10x3.6 rectangle.
        Spawn: x in [0.5, 2], y in [0.3, 3.3].
        Goal: (10.5, y_i) — each agent's own y-coordinate.
        """
        walls = [
            Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0])),   # bottom
            # right wall removed: open exit for agents to flow through
            Wall(np.array([10.0, 3.6]), np.array([0.0, 3.6])),   # top
            Wall(np.array([0.0, 3.6]), np.array([0.0, 0.0])),    # left
        ]
        world = World(walls)

        state = AgentState.create(
            self.n_agents,
            spawn_area=(0.5, 2.0, 0.3, 3.3),
            goals=np.array([10.5, 1.8]),  # placeholder, overwritten below
            seed=seed,
            heterogeneous=self.heterogeneous,
        )
        # Set per-agent goals to maintain their y-coordinate
        state.goals[:, 0] = 10.5
        state.goals[:, 1] = state.positions[:, 1]

        return world, state

    def is_complete(self, agent_state: AgentState, time: float) -> bool:
        """Complete when all agents have exited."""
        return agent_state.n_active == 0


class CorridorFDScenario(Scenario):
    """Corridor with continuous agent injection for fundamental diagram measurement.

    Agents are injected at a configurable rate at the left boundary.
    Density-speed pairs are measured in the area x in [2, 8] after warmup.

    Args:
        injection_rate: Agents per second to inject.
        warmup_time: Seconds before measurement starts.
        measure_time: Seconds of data collection.
    """

    injection_rate: float
    warmup_time: float
    measure_time: float

    def __init__(
        self,
        injection_rate: float = 5.0,
        warmup_time: float = 5.0,
        measure_time: float = 15.0,
    ):
        self.injection_rate = injection_rate
        self.warmup_time = warmup_time
        self.measure_time = measure_time

    def build(self, seed: int = 42) -> tuple[World, AgentState]:
        """Build 25m x 3.6m corridor with 1m exit gap for FD measurement.

        The exit bottleneck at x=24 creates back-pressure that forces
        density up in the measurement zone x in [8, 18]. At high injection
        rates, the queue fills the measurement area producing high density
        and reduced speed.
        """
        w = 3.6
        walls = [
            Wall(np.array([0.0, 0.0]), np.array([25.0, 0.0])),
            Wall(np.array([25.0, 0.0]), np.array([25.0, w])),
            Wall(np.array([25.0, w]), np.array([0.0, w])),
            Wall(np.array([0.0, w]), np.array([0.0, 0.0])),
            # Exit bottleneck at x=24: 0.6m gap from y=1.5 to y=2.1
            Wall(np.array([24.0, 0.0]), np.array([24.0, 1.5])),
            Wall(np.array([24.0, 2.1]), np.array([24.0, w])),
        ]
        world = World(walls)

        state = AgentState.create(
            5,
            spawn_area=(0.3, 2.0, 0.3, w - 0.3),
            goals=np.array([26.0, w / 2]),
            seed=seed,
        )
        return world, state

    def is_complete(self, agent_state: AgentState, time: float) -> bool:
        """Complete after warmup + measure time."""
        return time >= self.warmup_time + self.measure_time


class PeriodicCorridorScenario(Scenario):
    """Periodic boundary corridor for fundamental diagram measurement.

    Fixed number of agents in a corridor with wrap-around x boundaries.
    Density = n_agents / (length * width). Speed measured at steady state.
    Matches FZJ experimental protocol: controlled pedestrian count in a loop.

    Args:
        n_agents: Number of agents (controls density directly).
        corridor_length: Length in meters.
        corridor_width: Width in meters (1.8 for single-lane FZJ).
        warmup_steps: Steps before measurement starts.
        measure_steps: Steps to measure.
    """

    def __init__(
        self,
        n_agents: int = 50,
        corridor_length: float = 18.0,
        corridor_width: float = 1.8,
        warmup_steps: int = 500,
        measure_steps: int = 1000,
    ):
        self.n_agents = n_agents
        self.length = corridor_length
        self.width = corridor_width
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.periodic_length = corridor_length

    def build(self, seed: int = 42) -> tuple[World, AgentState]:
        """Build corridor with top/bottom walls only (x is periodic)."""
        walls = [
            Wall(np.array([0.0, 0.0]), np.array([self.length, 0.0])),
            Wall(np.array([0.0, self.width]), np.array([self.length, self.width])),
        ]
        world = World(walls)

        state = AgentState.create(
            self.n_agents,
            spawn_area=(0.5, self.length - 0.5, 0.3, self.width - 0.3),
            goals=np.array([self.length + 10.0, self.width / 2]),
            seed=seed,
            heterogeneous=True,
        )
        return world, state

    def is_complete(self, agent_state: AgentState, time: float) -> bool:
        """Never completes on its own (run manually with step counts)."""
        return False
