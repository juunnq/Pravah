"""Numerical integrators for the simulation: Euler and RK4."""

from typing import Callable

import numpy as np

from sim.core.helpers import clamp_speed


class EulerIntegrator:
    """Forward Euler integration scheme.

    Computes:
        accel = F / m
        v_new = v + accel * dt
        x_new = x + v_new * dt
    """

    def integrate(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray,
        masses: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one Euler integration step.

        Args:
            positions: Agent positions, shape (N, 2).
            velocities: Agent velocities, shape (N, 2).
            forces: Net forces on agents, shape (N, 2).
            masses: Agent masses, shape (N,).
            dt: Timestep in seconds.

        Returns:
            Tuple of (new_positions, new_velocities), each shape (N, 2).
        """
        accel = forces / masses[:, None]
        new_vel = velocities + accel * dt
        new_pos = positions + new_vel * dt
        return new_pos, new_vel


class RK4Integrator:
    """Fourth-order Runge-Kutta integration scheme.

    Uses four force evaluations per step for higher accuracy.
    Requires a callable force function rather than precomputed forces.
    """

    def integrate(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        force_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        masses: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one RK4 integration step.

        The system is dx/dt = v, dv/dt = F(x,v)/m.

        Args:
            positions: Agent positions, shape (N, 2).
            velocities: Agent velocities, shape (N, 2).
            force_fn: Callable(positions, velocities) -> forces (N, 2).
            masses: Agent masses, shape (N,).
            dt: Timestep in seconds.

        Returns:
            Tuple of (new_positions, new_velocities), each shape (N, 2).
        """
        m = masses[:, None]

        k1v = force_fn(positions, velocities) / m * dt
        k1x = velocities * dt

        k2v = force_fn(positions + k1x / 2, velocities + k1v / 2) / m * dt
        k2x = (velocities + k1v / 2) * dt

        k3v = force_fn(positions + k2x / 2, velocities + k2v / 2) / m * dt
        k3x = (velocities + k2v / 2) * dt

        k4v = force_fn(positions + k3x, velocities + k3v) / m * dt
        k4x = (velocities + k3v) * dt

        new_vel = velocities + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        new_pos = positions + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        return new_pos, new_vel
