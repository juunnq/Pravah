"""Time-to-Collision (TTC) anticipatory force model.

Computes the earliest collision time between agent pairs and applies
an exponentially-weighted repulsive force in the avoidance direction.
Vectorized over neighbor pairs for O(K) performance.
Parameters: k_ttc=1.5, tau_0=3.0, tau_max=8.0.
"""

import numpy as np

from sim.core.agent import AgentState
from sim.core.helpers import check_forces, safe_normalize


class TTCForceModel:
    """Time-to-Collision anticipatory steering force.

    For each pair (i, j), computes the time tau at which they would first
    collide assuming constant velocities, then applies:
        F_mag = k_ttc * exp(-tau / tau_0) / tau^2
    in the avoidance direction (separating their predicted future positions).

    Args:
        k_ttc: Force magnitude scaling.
        tau_0: Exponential decay time constant (s).
        tau_max: Maximum lookahead horizon (s).
    """

    def __init__(
        self,
        k_ttc: float = 1.5,
        tau_0: float = 3.0,
        tau_max: float = 8.0,
    ):
        self.k_ttc = k_ttc
        self.tau_0 = tau_0
        self.tau_max = tau_max

    def compute_ttc_forces(
        self,
        agent_state: AgentState,
        neighbor_lists: list[list[int]],
    ) -> np.ndarray:
        """Compute TTC anticipatory forces for all agents.

        Vectorized over neighbor pairs with masked tau computation.

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

        # Build flat pair arrays
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
            return check_forces(forces, "TTC")

        pi = np.array(pairs_i)
        pj = np.array(pairs_j)

        # Vectorized tau computation
        dx = pos[pj] - pos[pi]                          # (K, 2)
        dv = vel[pi] - vel[pj]                          # (K, 2)
        r = radii[pi] + radii[pj]                       # (K,)

        a = np.sum(dv * dv, axis=1)                     # (K,)
        b = -np.sum(dx * dv, axis=1)                    # (K,)
        c = np.sum(dx * dx, axis=1) - r * r             # (K,)
        disc = b * b - a * c                            # (K,)

        # Valid pairs: relative motion, positive discriminant
        valid = (a >= 1e-8) & (disc >= 0)
        tau = np.full(len(a), np.inf)
        sqrt_disc = np.sqrt(np.maximum(disc[valid], 0.0))
        tau[valid] = (-b[valid] - sqrt_disc) / a[valid]

        # Further filter: positive tau within horizon
        valid &= (tau > 0) & (tau <= self.tau_max)

        if not np.any(valid):
            return check_forces(forces, "TTC")

        # Compute forces for valid pairs only
        vi = valid.nonzero()[0]
        tau_v = tau[vi]
        pi_v = pi[vi]
        pj_v = pj[vi]

        F_mag = self.k_ttc * np.exp(-tau_v / self.tau_0) / (tau_v * tau_v)

        # Avoidance direction
        xi_f = pos[pi_v] + vel[pi_v] * tau_v[:, None]
        xj_f = pos[pj_v] + vel[pj_v] * tau_v[:, None]
        diff_f = xi_f - xj_f
        n_avoid = safe_normalize(diff_f)

        # Fallback for coincident future positions
        zero_mask = np.linalg.norm(diff_f, axis=1) < 1e-8
        if np.any(zero_mask):
            fallback = safe_normalize(pos[pi_v[zero_mask]] - pos[pj_v[zero_mask]])
            n_avoid[zero_mask] = fallback

        f_pair = F_mag[:, None] * n_avoid
        np.add.at(forces, pi_v, f_pair)

        return check_forces(forces, "TTC")
