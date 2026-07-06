"""Optimal Reciprocal Collision Avoidance (ORCA) velocity optimization.

Constructs half-plane velocity constraints from neighboring agents, then
solves a 2D linear program to find the closest feasible velocity to the
agent's preferred velocity.  Force = m * (v_orca - v) / tau_orca.

Per-agent loops are expected — ORCA's LP is inherently sequential.
"""

import numpy as np
from scipy.optimize import minimize

from sim.core.agent import AgentState
from sim.core.helpers import check_forces, safe_normalize


# ---------------------------------------------------------------------------
# Half-plane construction
# ---------------------------------------------------------------------------

def _halfplane_collision(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    vel_i: np.ndarray,
    vel_j: np.ndarray,
    r_sum: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Half-plane for the COLLISION case (agents already overlap).

    Section 10.1: push apart along the separating direction.

    Returns:
        (point, normal) defining the half-plane.
    """
    x_rel = pos_j - pos_i
    dist = np.linalg.norm(x_rel)
    direction = x_rel / dist if dist > 1e-6 else np.array([1.0, 0.0])

    # Push-apart velocity correction
    u = -(r_sum - dist) / dt * direction
    normal = safe_normalize(-direction)
    point = vel_i + 0.5 * u
    return point, normal


def _halfplane_normal(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    vel_i: np.ndarray,
    vel_j: np.ndarray,
    r_sum: float,
    tau_h: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Half-plane for the NON-COLLISION case (truncated VO).

    Section 10.2: determines cap vs leg region, projects onto boundary.

    Returns:
        (point, normal) or None if the pair can be skipped.
    """
    x_rel = pos_j - pos_i
    v_rel = vel_i - vel_j
    dist_sq = np.dot(x_rel, x_rel)
    dist = np.sqrt(dist_sq)
    r_sum_sq = r_sum * r_sum

    # w = v_rel - x_rel / tau_h (vector from truncation center to v_rel)
    w = v_rel - x_rel / tau_h
    w_len_sq = np.dot(w, w)
    w_dot_x = np.dot(w, x_rel)

    r_trunc = r_sum / tau_h

    # Determine cap vs leg
    if w_dot_x < 0 and w_dot_x * w_dot_x > r_sum_sq * w_len_sq:
        # Cap region: project onto truncation circle
        w_len = np.sqrt(w_len_sq)
        if w_len < 1e-10:
            return None
        w_unit = w / w_len
        u = (r_trunc - w_len) * w_unit
        normal = safe_normalize(w_unit)
    else:
        # Leg region
        leg_sq = dist_sq - r_sum_sq
        if leg_sq < 0:
            # Shouldn't happen in non-collision, but guard
            return None
        leg = np.sqrt(leg_sq)

        # Cross product determines left/right leg
        cross = x_rel[0] * v_rel[1] - x_rel[1] * v_rel[0]

        if cross > 0:
            # Left leg
            direction = np.array([
                x_rel[0] * leg - x_rel[1] * r_sum,
                x_rel[0] * r_sum + x_rel[1] * leg,
            ]) / dist_sq
        else:
            # Right leg
            direction = np.array([
                x_rel[0] * leg + x_rel[1] * r_sum,
                -x_rel[0] * r_sum + x_rel[1] * leg,
            ]) / dist_sq

        dot_v_dir = np.dot(v_rel, direction)
        u = dot_v_dir * direction - v_rel
        u_len = np.linalg.norm(u)
        if u_len < 1e-10:
            return None
        normal = safe_normalize(u)

    point = vel_i + 0.5 * u
    return point, normal


# ---------------------------------------------------------------------------
# 2D Linear Program — incremental solver
# ---------------------------------------------------------------------------

def solve_2d_lp(
    halfplanes: list[tuple[np.ndarray, np.ndarray]],
    v_pref: np.ndarray,
    max_speed: float,
) -> np.ndarray:
    """Incremental 2D linear program (Section 10.3).

    Finds the velocity closest to v_pref satisfying all half-plane
    constraints and the speed-circle constraint.

    Args:
        halfplanes: List of (point, normal) tuples.
        v_pref: Preferred velocity, shape (2,).
        max_speed: Maximum allowed speed.

    Returns:
        Optimal velocity, shape (2,).
    """
    # Start with v_pref clamped to speed circle
    pref_speed = np.linalg.norm(v_pref)
    if pref_speed > max_speed:
        result = v_pref * (max_speed / pref_speed)
    else:
        result = v_pref.copy()

    for k, (point_k, normal_k) in enumerate(halfplanes):
        # Check if current result already satisfies this constraint
        if np.dot(result - point_k, normal_k) >= 0:
            continue

        # Project onto constraint line
        line_dir = np.array([-normal_k[1], normal_k[0]])

        # Find valid t range from previous constraints
        t_left = -1e9
        t_right = 1e9
        infeasible = False

        for j in range(k):
            point_j, normal_j = halfplanes[j]
            denom = np.dot(line_dir, normal_j)
            numer = np.dot(point_j - point_k, normal_j)

            if abs(denom) < 1e-10:
                if numer > 0:
                    infeasible = True
                    break
                continue

            t = numer / denom
            if denom > 0:
                t_left = max(t_left, t)
            else:
                t_right = min(t_right, t)

        if infeasible or t_left > t_right:
            # Constraints are infeasible — keep best so far
            continue

        # Clamp to speed circle: |point_k + t * line_dir|^2 <= max_speed^2
        # Quadratic: a*t^2 + b*t + c <= 0
        a_q = np.dot(line_dir, line_dir)
        b_q = 2.0 * np.dot(point_k, line_dir)
        c_q = np.dot(point_k, point_k) - max_speed * max_speed
        disc = b_q * b_q - 4.0 * a_q * c_q

        if disc < 0:
            # Line doesn't intersect speed circle — keep best
            continue

        sqrt_disc = np.sqrt(disc)
        t_circle_left = (-b_q - sqrt_disc) / (2.0 * a_q)
        t_circle_right = (-b_q + sqrt_disc) / (2.0 * a_q)
        t_left = max(t_left, t_circle_left)
        t_right = min(t_right, t_circle_right)

        if t_left > t_right:
            continue

        # Find t that minimizes distance to v_pref
        t_opt = np.dot(v_pref - point_k, line_dir)
        t_opt = np.clip(t_opt, t_left, t_right)
        result = point_k + t_opt * line_dir

    return result


# ---------------------------------------------------------------------------
# Scipy fallback
# ---------------------------------------------------------------------------

def solve_lp_scipy(
    halfplanes: list[tuple[np.ndarray, np.ndarray]],
    v_pref: np.ndarray,
    max_speed: float,
) -> np.ndarray:
    """Fallback LP solver using scipy SLSQP (Section 10.4).

    Args:
        halfplanes: List of (point, normal) tuples.
        v_pref: Preferred velocity, shape (2,).
        max_speed: Maximum allowed speed.

    Returns:
        Optimal velocity, shape (2,).
    """
    constraints = [
        {
            "type": "ineq",
            "fun": lambda v, p=pt, n=nm: np.dot(v - p, n),
        }
        for pt, nm in halfplanes
    ]
    constraints.append(
        {"type": "ineq", "fun": lambda v: max_speed**2 - np.dot(v, v)}
    )
    res = minimize(
        lambda v: np.sum((v - v_pref) ** 2),
        v_pref,
        method="SLSQP",
        constraints=constraints,
    )
    return res.x if res.success else v_pref


# ---------------------------------------------------------------------------
# ORCA steering model
# ---------------------------------------------------------------------------

class ORCAModel:
    """ORCA velocity-obstacle steering model.

    Builds half-plane constraints from each neighbor, solves a 2D LP
    for the optimal velocity, then converts to a force.

    Args:
        time_horizon: Lookahead time for non-collision half-planes (s).
        tau_orca: Relaxation time for velocity-to-force conversion (s).
        dt: Simulation timestep for collision half-planes (s).
    """

    def __init__(
        self,
        time_horizon: float = 5.0,
        tau_orca: float = 0.5,
        dt: float = 0.01,
    ):
        self.time_horizon = time_horizon
        self.tau_orca = tau_orca
        self.dt = dt

    def compute_orca_forces(
        self,
        agent_state: AgentState,
        neighbor_lists: list[list[int]],
    ) -> np.ndarray:
        """Compute ORCA steering forces for all agents (Section 10.5).

        Args:
            agent_state: Current state of all agents.
            neighbor_lists: Per-agent list of neighbor indices.

        Returns:
            Forces array of shape (N, 2).
        """
        n = agent_state.n
        forces = np.zeros((n, 2))
        pos = agent_state.positions
        vel = agent_state.velocities
        radii = agent_state.radii
        active = agent_state.active
        masses = agent_state.masses
        desired_speeds = agent_state.desired_speeds
        goals = agent_state.goals

        for i in range(n):
            if not active[i]:
                continue

            # Build half-planes from neighbors
            halfplanes: list[tuple[np.ndarray, np.ndarray]] = []
            for j in neighbor_lists[i]:
                if j == i or not active[j]:
                    continue

                x_rel = pos[j] - pos[i]
                dist = np.linalg.norm(x_rel)
                r_sum = radii[i] + radii[j]

                if dist < r_sum:
                    # Collision case
                    hp = _halfplane_collision(
                        pos[i], pos[j], vel[i], vel[j], r_sum, self.dt
                    )
                    halfplanes.append(hp)
                else:
                    # Non-collision case
                    hp = _halfplane_normal(
                        pos[i], pos[j], vel[i], vel[j],
                        r_sum, self.time_horizon
                    )
                    if hp is not None:
                        halfplanes.append(hp)

            # Preferred velocity
            goal_dir = safe_normalize(goals[i] - pos[i])
            v_pref = desired_speeds[i] * goal_dir
            max_speed = 2.0 * desired_speeds[i]

            # Solve LP: try incremental, fall back to scipy
            try:
                v_orca = solve_2d_lp(halfplanes, v_pref, max_speed)
            except Exception:
                v_orca = solve_lp_scipy(halfplanes, v_pref, max_speed)

            # Convert velocity to force
            forces[i] = masses[i] * (v_orca - vel[i]) / self.tau_orca

        return check_forces(forces, "ORCA")
