"""Desired (goal-seeking) force with optional density-dependent speed reduction.

F = m * (v0_eff * e_hat - v) / tau
where v0_eff = v0 * (1 - exp(-gamma * (1/rho - 1/rho_max))) for rho > 0
      v0_eff = v0 for rho = 0 (free flow)

This implements the Weidmann (1993) speed-density coupling in the desired force,
which is the standard way SFM produces the fundamental diagram relationship.
"""

import numpy as np

from sim.core.helpers import check_forces, safe_normalize


def compute_desired_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    desired_speeds: np.ndarray,
    masses: np.ndarray,
    taus: np.ndarray,
    local_densities: np.ndarray | None = None,
    weidmann_gamma: float = 1.913,
    weidmann_rho_max: float = 5.4,
) -> np.ndarray:
    """Compute the desired (goal-seeking) force for all agents.

    F_i = m_i * (v0_eff_i * e_hat_i - v_i) / tau_i

    When local_densities is provided, v0_eff follows Weidmann (1993):
        v0_eff = v0 * (1 - exp(-gamma * (1/max(rho, 0.01) - 1/rho_max)))

    Args:
        positions: Agent positions, shape (N, 2).
        velocities: Agent velocities, shape (N, 2).
        goals: Agent goal positions, shape (N, 2).
        desired_speeds: Preferred free-flow speeds, shape (N,).
        masses: Agent masses, shape (N,).
        taus: Relaxation times, shape (N,).
        local_densities: Per-agent local density, shape (N,). Optional.
        weidmann_gamma: Weidmann (1993) shape parameter. Default 1.913.
        weidmann_rho_max: Weidmann (1993) jam density (ped/m^2). Default 5.4.

    Returns:
        Desired forces, shape (N, 2).
    """
    direction = safe_normalize(goals - positions)

    if local_densities is not None:
        # Weidmann (1993) speed-density coupling
        rho = np.maximum(local_densities, 0.01)
        speed_factor = 1.0 - np.exp(-weidmann_gamma * (1.0 / rho - 1.0 / weidmann_rho_max))
        speed_factor = np.clip(speed_factor, 0.05, 1.0)
        effective_speeds = desired_speeds * speed_factor
    else:
        effective_speeds = desired_speeds

    desired_vel = effective_speeds[:, None] * direction
    forces = masses[:, None] * (desired_vel - velocities) / taus[:, None]
    return check_forces(forces, "desired")
