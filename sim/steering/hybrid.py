"""Density-weighted hybrid steering model combining SFM, TTC, ORCA, and crush.

F = (1 - w_crush) * F_desire + F_SFM + F_TTC + w_ORCA * F_ORCA
  + w_crush * F_crush + F_wall

Sigmoid weights:
    w_ORCA(rho) = 1 - sigma(rho; 4.0, 2.0)    high at low density
    w_crush(rho) = sigma(rho; 5.5, 3.0)        high at high density
"""

import numpy as np

from sim.core.agent import AgentState
from sim.core.helpers import check_forces
from sim.core.world import Wall
from sim.steering.base import SteeringModel
from sim.steering.crush import CrushRegime
from sim.steering.desired import compute_desired_force
from sim.steering.orca import ORCAModel
from sim.steering.sfm import SocialForceModel
from sim.steering.ttc import TTCForceModel
from sim.steering.walls import WallForces


class HybridSteeringModel(SteeringModel):
    """Density-weighted hybrid of SFM, TTC, ORCA, crush, and wall forces.

    Args:
        config: Dict with boolean keys "sfm", "ttc", "orca", "crush".
        params: Flat parameter dict from params.yaml.
    """

    def __init__(self, config: dict, params: dict):
        self.config = config
        self.params = params

        self.sfm = SocialForceModel(
            A=params.get("A", 2000.0),
            B=params.get("B", 0.08),
            k=params.get("k", 120000.0),
            kappa=params.get("kappa", 240000.0),
        )

        self.ttc = (
            TTCForceModel(
                k_ttc=params.get("k_ttc", 1.5),
                tau_0=params.get("tau_0", 3.0),
                tau_max=params.get("tau_max", 8.0),
            )
            if config.get("ttc")
            else None
        )

        self.orca = (
            ORCAModel(
                time_horizon=params.get("time_horizon", 5.0),
                tau_orca=params.get("tau_orca", 0.5),
                dt=params.get("dt", 0.01),
            )
            if config.get("orca")
            else None
        )

        self.crush = (
            CrushRegime(
                k_crush=params.get("k_crush", 360000.0),
                kappa_crush=params.get("kappa_crush", 480000.0),
            )
            if config.get("crush")
            else None
        )

        self.wall_forces = WallForces(
            A=params.get("A", 2000.0),
            B=params.get("B", 0.08),
            k=params.get("k", 120000.0),
            kappa=params.get("kappa", 240000.0),
        )

    def compute_forces(
        self,
        agent_state: AgentState,
        neighbor_lists: list[list[int]],
        walls: list[Wall],
        local_densities: np.ndarray,
    ) -> np.ndarray:
        """Compute hybrid steering forces for all agents.

        Args:
            agent_state: Current state of all agents.
            neighbor_lists: Per-agent neighbor indices.
            walls: Wall segments.
            local_densities: Per-agent density, shape (N,).

        Returns:
            Forces array of shape (N, 2).
        """
        n = agent_state.n

        # Sigmoid weights
        if self.orca is not None:
            w_orca = 1.0 - self._sigmoid(
                local_densities,
                self.params.get("rho_orca_fade", 4.0),
                self.params.get("k_orca_fade", 2.0),
            )
        else:
            w_orca = np.zeros(n)

        if self.crush is not None:
            w_crush = self._sigmoid(
                local_densities,
                self.params.get("rho_crit", 5.5),
                self.params.get("k_crit", 3.0),
            )
        else:
            w_crush = np.zeros(n)

        # Desired force with density-dependent speed reduction
        F_desire = compute_desired_force(
            agent_state.positions,
            agent_state.velocities,
            agent_state.goals,
            agent_state.desired_speeds,
            agent_state.masses,
            agent_state.taus,
            local_densities=local_densities,
            weidmann_gamma=self.params.get("weidmann_gamma", 1.913),
            weidmann_rho_max=self.params.get("weidmann_rho_max", 5.4),
        )
        F = (1.0 - w_crush)[:, None] * F_desire

        # SFM agent-agent (always on)
        F += self.sfm.compute_agent_forces(agent_state, neighbor_lists)

        # TTC
        if self.ttc is not None:
            F += self.ttc.compute_ttc_forces(agent_state, neighbor_lists)

        # ORCA (weighted by density)
        if self.orca is not None:
            F += w_orca[:, None] * self.orca.compute_orca_forces(
                agent_state, neighbor_lists
            )

        # Crush (weighted by density)
        if self.crush is not None:
            F += w_crush[:, None] * self.crush.compute_crush_forces(
                agent_state, neighbor_lists
            )

        # Wall forces
        F += self.wall_forces.compute_wall_forces(agent_state, walls)

        return check_forces(F, "hybrid")

    def compute_forces_decomposed(
        self,
        agent_state: AgentState,
        neighbor_lists: list[list[int]],
        walls: list[Wall],
        local_densities: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute forces and return per-component magnitudes.

        Returns:
            (total_forces, component_magnitudes) where component_magnitudes
            is a dict mapping component name to per-agent magnitude arrays.
        """
        n = agent_state.n
        mags: dict[str, np.ndarray] = {}

        # Desired force
        F_desire = compute_desired_force(
            agent_state.positions, agent_state.velocities,
            agent_state.goals, agent_state.desired_speeds,
            agent_state.masses, agent_state.taus,
            local_densities=local_densities,
            weidmann_gamma=self.params.get("weidmann_gamma", 1.913),
            weidmann_rho_max=self.params.get("weidmann_rho_max", 5.4),
        )
        mags["des"] = np.linalg.norm(F_desire, axis=1)

        # SFM
        F_sfm = self.sfm.compute_agent_forces(agent_state, neighbor_lists)
        mags["sfm"] = np.linalg.norm(F_sfm, axis=1)

        # TTC
        if self.ttc is not None:
            F_ttc = self.ttc.compute_ttc_forces(agent_state, neighbor_lists)
            mags["ttc"] = np.linalg.norm(F_ttc, axis=1)
        else:
            F_ttc = np.zeros((n, 2))
            mags["ttc"] = np.zeros(n)

        # ORCA
        if self.orca is not None:
            w_orca = 1.0 - self._sigmoid(
                local_densities,
                self.params.get("rho_orca_fade", 4.0),
                self.params.get("k_orca_fade", 2.0),
            )
            F_orca = self.orca.compute_orca_forces(agent_state, neighbor_lists)
            mags["orca"] = np.linalg.norm(F_orca, axis=1)
        else:
            w_orca = np.zeros(n)
            F_orca = np.zeros((n, 2))
            mags["orca"] = np.zeros(n)

        # Wall
        F_wall = self.wall_forces.compute_wall_forces(agent_state, walls)
        mags["wall"] = np.linalg.norm(F_wall, axis=1)

        # Crush (skip if not present — already cut from paper)
        w_crush = np.zeros(n)
        if self.crush is not None:
            w_crush = self._sigmoid(
                local_densities,
                self.params.get("rho_crit", 5.5),
                self.params.get("k_crit", 3.0),
            )

        # Total (same formula as compute_forces)
        F = (1.0 - w_crush)[:, None] * F_desire + F_sfm + F_ttc
        F += w_orca[:, None] * F_orca
        if self.crush is not None:
            F += w_crush[:, None] * self.crush.compute_crush_forces(
                agent_state, neighbor_lists)
        F += F_wall

        return check_forces(F, "hybrid"), mags

    @staticmethod
    def _sigmoid(x: np.ndarray, x0: float, k: float) -> np.ndarray:
        """Logistic sigmoid: 1 / (1 + exp(-k*(x - x0)))."""
        return 1.0 / (1.0 + np.exp(-k * (np.asarray(x, dtype=float) - x0)))
