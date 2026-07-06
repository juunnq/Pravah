"""Tests for Euler and RK4 integrators."""

import numpy as np
import pytest

from sim.core.integrator import EulerIntegrator, RK4Integrator


class TestEulerIntegrator:
    def test_constant_force(self):
        """F=(1,0), m=1, 100 steps dt=0.01 -> vel ~(1,0), pos ~(0.505,0)."""
        euler = EulerIntegrator()
        pos = np.array([[0.0, 0.0]])
        vel = np.array([[0.0, 0.0]])
        forces = np.array([[1.0, 0.0]])
        masses = np.array([1.0])
        dt = 0.01

        for _ in range(100):
            pos, vel = euler.integrate(pos, vel, forces, masses, dt)

        # After 100 steps: v = a*t = 1.0*1.0 = 1.0 m/s
        np.testing.assert_allclose(vel[0], [1.0, 0.0], atol=1e-10)
        # x = 0.5*a*t^2 + extra from Euler (v_new*dt not v_old*dt)
        # Euler: x accumulates v_new*dt each step, so x > 0.5
        assert pos[0, 0] > 0.5
        assert pos[0, 0] < 0.51
        np.testing.assert_allclose(pos[0, 1], 0.0, atol=1e-10)

    def test_no_nan(self):
        """50 agents with random forces produce no NaN."""
        euler = EulerIntegrator()
        rng = np.random.Generator(np.random.PCG64(123))
        pos = rng.uniform(-10, 10, (50, 2))
        vel = rng.uniform(-2, 2, (50, 2))
        forces = rng.uniform(-100, 100, (50, 2))
        masses = rng.uniform(50, 100, 50)
        dt = 0.01

        for _ in range(100):
            pos, vel = euler.integrate(pos, vel, forces, masses, dt)

        assert not np.any(np.isnan(pos))
        assert not np.any(np.isnan(vel))

    def test_zero_force(self):
        """Zero force preserves velocity, updates position linearly."""
        euler = EulerIntegrator()
        pos = np.array([[1.0, 2.0]])
        vel = np.array([[3.0, -1.0]])
        forces = np.array([[0.0, 0.0]])
        masses = np.array([80.0])
        dt = 0.1

        new_pos, new_vel = euler.integrate(pos, vel, forces, masses, dt)
        np.testing.assert_allclose(new_vel, vel, atol=1e-12)
        np.testing.assert_allclose(new_pos, pos + vel * dt, atol=1e-12)

    def test_multi_agent(self):
        """Multiple agents with different masses integrate independently."""
        euler = EulerIntegrator()
        pos = np.zeros((3, 2))
        vel = np.zeros((3, 2))
        forces = np.array([[10.0, 0.0], [0.0, 20.0], [5.0, 5.0]])
        masses = np.array([1.0, 2.0, 5.0])
        dt = 0.1

        new_pos, new_vel = euler.integrate(pos, vel, forces, masses, dt)
        # v = F/m * dt
        np.testing.assert_allclose(new_vel[0], [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(new_vel[1], [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(new_vel[2], [0.1, 0.1], atol=1e-12)


class TestRK4Integrator:
    def test_harmonic_oscillator(self):
        """F=-x, m=1: one period (2*pi s) should return close to start."""
        rk4 = RK4Integrator()
        pos = np.array([[1.0, 0.0]])
        vel = np.array([[0.0, 0.0]])
        masses = np.array([1.0])
        dt = 0.01
        period = 2 * np.pi
        n_steps = int(period / dt)

        def force_fn(p, v):
            return -p  # harmonic: F = -x

        for _ in range(n_steps):
            pos, vel = rk4.integrate(pos, vel, force_fn, masses, dt)

        # Should return close to (1, 0) after one period
        np.testing.assert_allclose(pos[0, 0], 1.0, atol=0.01)
        np.testing.assert_allclose(pos[0, 1], 0.0, atol=0.01)
        np.testing.assert_allclose(vel[0, 0], 0.0, atol=0.01)

    def test_no_nan(self):
        """50 agents with random force_fn produce no NaN."""
        rk4 = RK4Integrator()
        rng = np.random.Generator(np.random.PCG64(456))
        pos = rng.uniform(-10, 10, (50, 2))
        vel = rng.uniform(-2, 2, (50, 2))
        masses = rng.uniform(50, 100, 50)
        dt = 0.01

        # Force proportional to position difference from origin
        def force_fn(p, v):
            return -0.5 * p - 0.1 * v

        for _ in range(100):
            pos, vel = rk4.integrate(pos, vel, force_fn, masses, dt)

        assert not np.any(np.isnan(pos))
        assert not np.any(np.isnan(vel))

    def test_constant_force(self):
        """Constant force: RK4 should give exact result (polynomial trajectory)."""
        rk4 = RK4Integrator()
        pos = np.array([[0.0, 0.0]])
        vel = np.array([[0.0, 0.0]])
        masses = np.array([1.0])
        dt = 0.01

        def force_fn(p, v):
            return np.array([[1.0, 0.0]])

        for _ in range(100):
            pos, vel = rk4.integrate(pos, vel, force_fn, masses, dt)

        t = 1.0
        # Exact: x = 0.5*a*t^2 = 0.5, v = a*t = 1.0
        np.testing.assert_allclose(vel[0], [1.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(pos[0, 0], 0.5, atol=1e-8)

    def test_rk4_more_accurate_than_euler(self):
        """RK4 should be more accurate than Euler on a nonlinear problem."""
        pos0 = np.array([[1.0, 0.0]])
        vel0 = np.array([[0.0, 1.0]])
        masses = np.array([1.0])
        dt = 0.05
        n_steps = 50

        def force_fn(p, v):
            # Circular orbit: F = -r/|r|^3 (unit gravity)
            r = np.linalg.norm(p, axis=1, keepdims=True)
            return -p / np.maximum(r, 1e-8) ** 3

        # Run Euler
        euler = EulerIntegrator()
        pos_e, vel_e = pos0.copy(), vel0.copy()
        for _ in range(n_steps):
            forces = force_fn(pos_e, vel_e)
            pos_e, vel_e = euler.integrate(pos_e, vel_e, forces, masses, dt)

        # Run RK4
        rk4 = RK4Integrator()
        pos_r, vel_r = pos0.copy(), vel0.copy()
        for _ in range(n_steps):
            pos_r, vel_r = rk4.integrate(pos_r, vel_r, force_fn, masses, dt)

        # Energy conservation check: E = 0.5*v^2 - 1/r, should be -0.5
        E_euler = 0.5 * np.sum(vel_e**2) - 1.0 / np.linalg.norm(pos_e)
        E_rk4 = 0.5 * np.sum(vel_r**2) - 1.0 / np.linalg.norm(pos_r)
        assert abs(E_rk4 - (-0.5)) < abs(E_euler - (-0.5))
