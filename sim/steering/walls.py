"""Wall forces: social repulsion, body compression, and friction against walls.

Same SFM structure as agent-agent but using agent-to-segment distance.
Wall is stationary so friction uses agent velocity directly.
"""

import numpy as np

from sim.core.agent import AgentState
from sim.core.helpers import check_forces
from sim.core.world import Wall, agents_to_walls


class WallForces:
    """Wall interaction forces using the SFM formulation.

    Forces per wall:
        f_iw = A*exp((r_i - d_iw)/B) * n_iw              [social]
             + k * g(r_i - d_iw) * n_iw                   [body]
             + kappa * g(r_i - d_iw) * dot(v_i, t_iw) * t_iw  [friction]

    Args:
        A: Social repulsion strength (N).
        B: Social repulsion range (m).
        k: Body compression coefficient (kg/s^2).
        kappa: Sliding friction coefficient (kg/(m*s)).
    """

    def __init__(
        self,
        A: float = 2000.0,
        B: float = 0.08,
        k: float = 120000.0,
        kappa: float = 240000.0,
    ):
        self.A = A
        self.B = B
        self.k = k
        self.kappa = kappa

    def compute_wall_forces(
        self,
        agent_state: AgentState,
        walls: list[Wall],
    ) -> np.ndarray:
        """Compute wall interaction forces for all agents.

        Vectorized over agents, loops over walls.

        Args:
            agent_state: Current state of all agents.
            walls: List of wall segments.

        Returns:
            Forces array of shape (N, 2).
        """
        n = agent_state.n
        if not walls:
            return check_forces(np.zeros((n, 2)), "walls")

        # Get distances and normals for all agents to all walls
        distances, normals = agents_to_walls(agent_state.positions, walls)
        # distances: (N, W), normals: (N, W, 2)

        forces = np.zeros((n, 2))
        radii = agent_state.radii
        vel = agent_state.velocities
        active = agent_state.active

        for w_idx in range(len(walls)):
            d_iw = distances[:, w_idx]              # (N,)
            n_iw = normals[:, w_idx]                # (N, 2)

            # Tangent: 90-degree rotation of normal
            t_iw = np.empty_like(n_iw)
            t_iw[:, 0] = -n_iw[:, 1]
            t_iw[:, 1] = n_iw[:, 0]

            # Overlap
            overlap = np.maximum(0.0, radii - d_iw)  # (N,)

            # Only compute for active agents
            amask = active.astype(float)  # (N,)

            # Social repulsion (always active)
            exp_term = np.exp((radii - d_iw) / self.B)  # (N,)
            f_social = (self.A * exp_term * amask)[:, None] * n_iw  # (N, 2)

            # Body compression (contact only)
            f_body = (self.k * overlap * amask)[:, None] * n_iw  # (N, 2)

            # Sliding friction: wall stationary, use -v_i (Helbing 2000 Eq.3)
            vt = np.sum(vel * t_iw, axis=1)  # dot(v_i, t_iw), (N,)
            f_friction = -(self.kappa * overlap * vt * amask)[:, None] * t_iw  # (N, 2)

            forces += f_social + f_body + f_friction

        return check_forces(forces, "walls")
