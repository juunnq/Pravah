"""Social Force Model: agent-agent repulsion, body compression, and friction.

Implements the Helbing-Molnar SFM with contact forces (Helbing et al. 2000).
Vectorized over neighbor pairs using np.add.at for O(K) memory where K = total pairs.
Parameters: A=2000N, B=0.08m, k=120000 kg/s^2, kappa=240000 kg/(m*s).
"""

import numpy as np

from sim.core.agent import AgentState
from sim.core.helpers import check_forces


class SocialForceModel:
    """Social Force Model for agent-agent interaction.

    Forces:
        f_ij = A*exp((r_ij - d_ij)/B) * n_ij                   [social, always]
             + k * g(r_ij - d_ij) * n_ij                        [body, contact]
             + kappa * g(r_ij - d_ij) * dot(dv_ji, t_ij) * t_ij [friction, contact]

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

    def compute_agent_forces(
        self,
        agent_state: AgentState,
        neighbor_lists: list[list[int]],
    ) -> np.ndarray:
        """Compute SFM agent-agent interaction forces.

        Vectorized over neighbor pairs (sparse) instead of full N*N grid.

        Args:
            agent_state: Current state of all agents.
            neighbor_lists: Per-agent list of neighbor indices from KDTree.

        Returns:
            Forces array of shape (N, 2).
        """
        n = agent_state.n
        pos = agent_state.positions
        vel = agent_state.velocities
        radii = agent_state.radii
        active = agent_state.active

        # Build flat arrays of valid (i, j) neighbor pairs
        pairs_i = []
        pairs_j = []
        for i in range(n):
            if not active[i]:
                continue
            for j in neighbor_lists[i]:
                if j == i or not active[j]:
                    continue
                pairs_i.append(i)
                pairs_j.append(j)

        forces = np.zeros((n, 2))
        if not pairs_i:
            return check_forces(forces, "SFM")

        pi = np.array(pairs_i)
        pj = np.array(pairs_j)

        # Pairwise vectors: diff = pos[i] - pos[j] (j->i direction)
        diff = pos[pi] - pos[pj]                       # (K, 2)
        dist_raw = np.linalg.norm(diff, axis=1)         # (K,)
        dist = np.maximum(dist_raw, 1e-6)               # (K,)

        # Normal and tangent — arbitrary unit normal for near-coincident
        n_ij = diff / dist[:, None]                      # (K, 2)
        near = dist_raw < 1e-6
        if np.any(near):
            n_ij[near] = np.array([1.0, 0.0])
        t_ij = np.column_stack([-n_ij[:, 1], n_ij[:, 0]])  # (K, 2)

        # Combined radii and overlap
        r_ij = radii[pi] + radii[pj]                    # (K,)
        overlap = np.maximum(0.0, r_ij - dist)           # (K,)

        # Social repulsion (always)
        social_mag = self.A * np.exp((r_ij - dist) / self.B)  # (K,)
        f_social = social_mag[:, None] * n_ij             # (K, 2)

        # Body compression (contact only)
        f_body = (self.k * overlap)[:, None] * n_ij       # (K, 2)

        # Sliding friction: dv_ji = vel[j] - vel[i]
        dv = vel[pj] - vel[pi]                           # (K, 2)
        dv_dot_t = np.sum(dv * t_ij, axis=1)             # (K,)
        f_friction = (self.kappa * overlap * dv_dot_t)[:, None] * t_ij  # (K, 2)

        # Accumulate per-pair forces into per-agent totals
        f_pair = f_social + f_body + f_friction           # (K, 2)
        np.add.at(forces, pi, f_pair)

        return check_forces(forces, "SFM")
