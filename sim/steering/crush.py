"""Crush regime: enhanced contact forces with no social repulsion.

Same structure as SFM contact forces but 3x body compression and 2x friction.
Vectorized over neighbor pairs using np.add.at for O(K) memory.
Parameters: k_crush=360000, kappa_crush=480000.
"""

import numpy as np

from sim.core.agent import AgentState
from sim.core.helpers import check_forces


class CrushRegime:
    """Crush-regime contact forces (no social repulsion).

    f_crush_ij = k_crush * g(r_ij - d_ij) * n_ij
              + kappa_crush * g(r_ij - d_ij) * dot(dv_ji, t_ij) * t_ij

    Args:
        k_crush: Enhanced body compression coefficient (kg/s^2).
        kappa_crush: Enhanced sliding friction coefficient (kg/(m*s)).
    """

    def __init__(
        self,
        k_crush: float = 360000.0,
        kappa_crush: float = 480000.0,
    ):
        self.k_crush = k_crush
        self.kappa_crush = kappa_crush

    def compute_crush_forces(
        self,
        agent_state: AgentState,
        neighbor_lists: list[list[int]],
    ) -> np.ndarray:
        """Compute crush-regime contact forces (sparse, vectorized over pairs).

        Args:
            agent_state: Current state of all agents.
            neighbor_lists: Per-agent list of neighbor indices.

        Returns:
            Forces array of shape (N, 2).
        """
        n = agent_state.n
        pos = agent_state.positions
        vel = agent_state.velocities
        radii = agent_state.radii
        active = agent_state.active

        # Build flat arrays of valid overlapping pairs
        pairs_i = []
        pairs_j = []
        for i in range(n):
            if not active[i]:
                continue
            for j in neighbor_lists[i]:
                if j == i or not active[j]:
                    continue
                # Pre-check overlap to skip non-contact pairs
                dx = pos[i] - pos[j]
                d = np.linalg.norm(dx)
                if d < radii[i] + radii[j]:
                    pairs_i.append(i)
                    pairs_j.append(j)

        forces = np.zeros((n, 2))
        if not pairs_i:
            return check_forces(forces, "crush")

        pi = np.array(pairs_i)
        pj = np.array(pairs_j)

        diff = pos[pi] - pos[pj]                        # (K, 2)
        dist_raw = np.linalg.norm(diff, axis=1)          # (K,)
        dist = np.maximum(dist_raw, 1e-6)

        n_ij = diff / dist[:, None]
        near = dist_raw < 1e-6
        if np.any(near):
            n_ij[near] = np.array([1.0, 0.0])
        t_ij = np.column_stack([-n_ij[:, 1], n_ij[:, 0]])

        r_ij = radii[pi] + radii[pj]
        overlap = np.maximum(0.0, r_ij - dist)

        # Body compression (enhanced, no social)
        f_body = (self.k_crush * overlap)[:, None] * n_ij

        # Sliding friction (enhanced)
        dv = vel[pj] - vel[pi]
        dv_dot_t = np.sum(dv * t_ij, axis=1)
        f_friction = (self.kappa_crush * overlap * dv_dot_t)[:, None] * t_ij

        f_pair = f_body + f_friction
        np.add.at(forces, pi, f_pair)

        return check_forces(forces, "crush")
