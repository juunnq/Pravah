"""Tests for RailwayPlatformScenario geometry (Phase 1).

Geometry tests are assertion-level and do NOT step the simulation. Exactly one
test (the bounded smoke test) advances the engine, and only briefly.
"""

import numpy as np
import pytest

from sim.core.simulation import Simulation
from sim.density.voronoi import VoronoiDensityEstimator
from sim.scenarios.railway_platform import RailwayPlatformScenario

# Expected landmark literals, pinned to the frozen geometry spec (independent of the module's
# own constants so the test genuinely checks the frozen values). Throat y-values are
# NOT pinned as literals — they are computed as Y_CENTER +/- throat_width/2.
X_HOLD_L = 0.0
X_HOLD_R = 20.0
X_THROAT_START = 24.0
X_THROAT_END = 32.0
X_PLATFORM_END = 52.0
X_GOAL = 53.0
Y_BOTTOM = 0.0
Y_TOP = 20.0
Y_PLATFORM_BOT = 3.0
Y_PLATFORM_TOP = 17.0
Y_CENTER = 10.0
SPAWN = (2.0, 14.0, 4.0, 16.0)


def _expected_walls(throat_width: float) -> list[tuple[tuple, tuple]]:
    """The frozen 11-wall geometry table, throat edges derived from W."""
    half = throat_width / 2.0
    tb = Y_CENTER - half  # throat bottom y
    tt = Y_CENTER + half  # throat top y
    return [
        ((X_HOLD_L, Y_BOTTOM), (X_HOLD_R, Y_BOTTOM)),            # 1 holding bottom
        ((X_HOLD_L, Y_TOP), (X_HOLD_R, Y_TOP)),                  # 2 holding top
        ((X_HOLD_L, Y_BOTTOM), (X_HOLD_L, Y_TOP)),               # 3 holding left
        ((X_HOLD_R, Y_BOTTOM), (X_THROAT_START, tb)),            # 4 taper bottom
        ((X_HOLD_R, Y_TOP), (X_THROAT_START, tt)),               # 5 taper top
        ((X_THROAT_START, tb), (X_THROAT_END, tb)),              # 6 throat bottom
        ((X_THROAT_START, tt), (X_THROAT_END, tt)),              # 7 throat top
        ((X_THROAT_END, tb), (X_THROAT_END, Y_PLATFORM_BOT)),    # 8 step-down
        ((X_THROAT_END, tt), (X_THROAT_END, Y_PLATFORM_TOP)),    # 9 step-up
        ((X_THROAT_END, Y_PLATFORM_BOT), (X_PLATFORM_END, Y_PLATFORM_BOT)),  # 10 platform bottom
        ((X_THROAT_END, Y_PLATFORM_TOP), (X_PLATFORM_END, Y_PLATFORM_TOP)),  # 11 platform top
    ]


def _same_segment(a_start, a_end, b_start, b_end, tol=1e-6) -> bool:
    """True if segment a equals segment b, ignoring direction."""
    fwd = (np.allclose(a_start, b_start, atol=tol) and np.allclose(a_end, b_end, atol=tol))
    rev = (np.allclose(a_start, b_end, atol=tol) and np.allclose(a_end, b_start, atol=tol))
    return fwd or rev


# 1. build() returns (World, AgentState); exactly 11 walls.
def test_build_returns_world_and_state_with_11_walls():
    scenario = RailwayPlatformScenario()
    world, state = scenario.build(seed=42)
    from sim.core.world import World
    from sim.core.agent import AgentState
    assert isinstance(world, World)
    assert isinstance(state, AgentState)
    assert len(world.walls) == 11


# 2. Each wall matches the table exactly (throat ys via Y_CENTER +/- W/2).
def test_walls_match_table_exactly():
    scenario = RailwayPlatformScenario(throat_width=3.66)
    world, _ = scenario.build()
    expected = _expected_walls(3.66)
    assert len(world.walls) == len(expected)
    for i, (w, (exp_start, exp_end)) in enumerate(zip(world.walls, expected), start=1):
        assert w.start == pytest.approx(np.array(exp_start)), f"wall {i} start"
        assert w.end == pytest.approx(np.array(exp_end)), f"wall {i} end"


# 3. Open exit: no wall is a vertical segment on x=52.
def test_no_far_wall_at_x52():
    scenario = RailwayPlatformScenario()
    world, _ = scenario.build()
    for w in world.walls:
        is_vertical_at_52 = (
            w.start[0] == pytest.approx(52.0)
            and w.end[0] == pytest.approx(52.0)
        )
        assert not is_vertical_at_52, "platform far edge must be an open exit, not a wall"


# 4. N==150 agents, all active at t=0; positions inside spawn_area.
def test_agents_count_active_and_within_spawn():
    scenario = RailwayPlatformScenario()
    _, state = scenario.build(seed=42)
    assert state.n == 150
    assert state.n_active == 150
    assert np.all(state.active)
    x_min, x_max, y_min, y_max = SPAWN
    assert np.all(state.positions[:, 0] >= x_min)
    assert np.all(state.positions[:, 0] <= x_max)
    assert np.all(state.positions[:, 1] >= y_min)
    assert np.all(state.positions[:, 1] <= y_max)


# 5. All goals equal (53, 10).
def test_goals_all_at_goal_point():
    scenario = RailwayPlatformScenario()
    _, state = scenario.build()
    assert state.goals.shape == (150, 2)
    assert np.allclose(state.goals, np.array([X_GOAL, Y_CENTER]))


# 6. Heterogeneity present.
def test_heterogeneous_agents():
    scenario = RailwayPlatformScenario()
    _, state = scenario.build(seed=42)
    assert state.desired_speeds.std() > 0
    assert state.radii.std() > 0


# 7. domain_polygon: valid, closed-able, simple, contains all initial agents.
def test_domain_polygon_valid_and_contains_agents():
    from shapely.geometry import Point, Polygon

    scenario = RailwayPlatformScenario(throat_width=3.66)
    domain = scenario.domain_polygon()
    assert domain.shape == (12, 2)

    # Expected vertex order (throat ys from W).
    half = 3.66 / 2.0
    tb, tt = Y_CENTER - half, Y_CENTER + half
    expected = np.array([
        [X_HOLD_L, Y_BOTTOM], [X_HOLD_R, Y_BOTTOM],
        [X_THROAT_START, tb], [X_THROAT_END, tb],
        [X_THROAT_END, Y_PLATFORM_BOT], [X_PLATFORM_END, Y_PLATFORM_BOT],
        [X_PLATFORM_END, Y_PLATFORM_TOP], [X_THROAT_END, Y_PLATFORM_TOP],
        [X_THROAT_END, tt], [X_THROAT_START, tt],
        [X_HOLD_R, Y_TOP], [X_HOLD_L, Y_TOP],
    ])
    assert np.allclose(domain, expected)

    poly = Polygon(domain)
    assert poly.is_valid          # simple (not self-intersecting), valid ring
    assert poly.is_simple

    _, state = scenario.build(seed=42)
    for p in state.positions:
        assert poly.contains(Point(p[0], p[1])), f"agent {p} outside domain"


# 8. Polygon-vs-walls consistency: only the far edge (52,3)-(52,17) is not a wall.
def test_polygon_edges_map_to_walls_except_far_edge():
    scenario = RailwayPlatformScenario(throat_width=3.66)
    world, _ = scenario.build()
    domain = scenario.domain_polygon()
    n = len(domain)

    far_edge = (np.array([X_PLATFORM_END, Y_PLATFORM_BOT]),
                np.array([X_PLATFORM_END, Y_PLATFORM_TOP]))

    non_wall_edges = 0
    for i in range(n):
        a = domain[i]
        b = domain[(i + 1) % n]
        matched = any(_same_segment(a, b, w.start, w.end) for w in world.walls)
        if not matched:
            non_wall_edges += 1
            # the only unmatched edge must be the far edge
            assert _same_segment(a, b, far_edge[0], far_edge[1]), \
                f"unexpected non-wall polygon edge {a}->{b}"
    assert non_wall_edges == 1, "exactly one polygon edge (the far edge) is not a wall"


# 9. Parametrization: throat_width=6.0 moves throat edges and dependent geometry.
def test_parametrization_throat_width_6():
    scenario = RailwayPlatformScenario(throat_width=6.0)
    world, _ = scenario.build()

    # throat edges -> 7.0 / 13.0
    tb, tt = Y_CENTER - 3.0, Y_CENTER + 3.0
    assert tb == pytest.approx(7.0)
    assert tt == pytest.approx(13.0)

    # walls #4-#9 use the new throat ys.
    expected = _expected_walls(6.0)
    for i in (3, 4, 5, 6, 7, 8):  # zero-based indices for walls #4-#9
        assert world.walls[i].start == pytest.approx(np.array(expected[i][0]))
        assert world.walls[i].end == pytest.approx(np.array(expected[i][1]))

    # domain throat vertices move.
    domain = scenario.domain_polygon()
    assert domain[2] == pytest.approx(np.array([X_THROAT_START, 7.0]))   # (24, 7)
    assert domain[3] == pytest.approx(np.array([X_THROAT_END, 7.0]))     # (32, 7)
    assert domain[8] == pytest.approx(np.array([X_THROAT_END, 13.0]))    # (32, 13)
    assert domain[9] == pytest.approx(np.array([X_THROAT_START, 13.0]))  # (24, 13)

    # fob_zone moves.
    assert scenario.fob_zone() == pytest.approx((24.0, 32.0, 7.0, 13.0))


# 10. fob_zone == (24, 32, Y_CENTER-half, Y_CENTER+half) for both W values.
@pytest.mark.parametrize("W", [3.66, 6.0])
def test_fob_zone_tracks_width(W):
    scenario = RailwayPlatformScenario(throat_width=W)
    half = W / 2.0
    assert scenario.fob_zone() == pytest.approx(
        (X_THROAT_START, X_THROAT_END, Y_CENTER - half, Y_CENTER + half)
    )


# 11. is_complete: False with active agents, True when none active.
def test_is_complete():
    scenario = RailwayPlatformScenario()
    _, state = scenario.build()
    assert scenario.is_complete(state, 0.0) is False
    state.deactivate(state.active_indices)
    assert scenario.is_complete(state, 1.0) is True


# 12. Density wiring: Voronoi with the domain returns finite (N,) densities.
def test_voronoi_density_wiring():
    scenario = RailwayPlatformScenario()
    _, state = scenario.build(seed=42)
    estimator = VoronoiDensityEstimator(domain=scenario.domain_polygon())
    dens = estimator.estimate(state.positions)
    assert dens.shape == (state.n,)
    assert np.all(np.isfinite(dens))


# 13. Bounded smoke: 25 agents, C4, 300 steps; agents move toward the goal, no NaN.
def test_bounded_smoke_run():
    scenario = RailwayPlatformScenario(n_agents=25)
    sim = Simulation.from_scenario(
        scenario, "C4", seed=42,
        density_estimator=VoronoiDensityEstimator(domain=scenario.domain_polygon()),
    )
    x_before = sim.state.positions[:, 0].mean()
    for _ in range(300):
        sim.step()
    assert not np.any(np.isnan(sim.state.positions))
    assert not np.any(np.isnan(sim.state.velocities))
    x_after = sim.state.positions[:, 0].mean()
    assert x_after > x_before, "agents should advance toward the goal (+x)"
