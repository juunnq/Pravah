import pytest
import numpy as np
from sim.core.agent import AgentState
from sim.core.world import World, Wall


@pytest.fixture
def box_world():
    """10x10 box with 4 walls."""
    walls = [
        Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0])),
        Wall(np.array([10.0, 0.0]), np.array([10.0, 10.0])),
        Wall(np.array([10.0, 10.0]), np.array([0.0, 10.0])),
        Wall(np.array([0.0, 10.0]), np.array([0.0, 0.0])),
    ]
    return World(walls)


@pytest.fixture
def corridor_world():
    """10x3.6 corridor."""
    walls = [
        Wall(np.array([0.0, 0.0]), np.array([10.0, 0.0])),
        Wall(np.array([10.0, 0.0]), np.array([10.0, 3.6])),
        Wall(np.array([10.0, 3.6]), np.array([0.0, 3.6])),
        Wall(np.array([0.0, 3.6]), np.array([0.0, 0.0])),
    ]
    return World(walls)


@pytest.fixture
def two_agents():
    return AgentState.create(
        2, spawn_area=(0, 3, 0.5, 1.5), goals=np.array([8.0, 1.0]), seed=42, heterogeneous=False
    )


@pytest.fixture
def single_agent():
    return AgentState.create(
        1, spawn_area=(1, 1.5, 1.5, 2.0), goals=np.array([9.0, 1.8]), seed=42, heterogeneous=False
    )
