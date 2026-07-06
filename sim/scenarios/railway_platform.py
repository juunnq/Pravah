"""Railway-platform crowd-safety scenario: holding area -> FOB throat -> platform.

Geometry frozen at design time. Models a New-Delhi-like island platform fed by
a foot-overbridge (FOB) / staircase throat: a large holding area (concourse) tapers
into a narrow throat (the bottleneck / crush nucleation point) that empties onto a
platform. Agents spawn in the holding area and walk to a goal past the platform's
*open* far edge.

Phase 1 is GEOMETRY ONLY: build() spawns all N agents at t=0. Timed/batched release,
surge schedules, the density detector, and risk thresholds are later phases.

Provenance tags on constants:
  [OSM]        measured from OpenStreetMap, New Delhi station (NDLS)
  [CAG]        CAG 2016 minimum FOB-width recommendation (6 m), via Elphinstone reporting
  [LEGACY]     legacy 12-ft (~3.66 m) FOB class, pre-mandate
  [ASSUMPTION] stylized design choice, not surveyed (upgrade path: RTI to Northern Railway)
"""

import numpy as np

from sim.core.agent import AgentState
from sim.core.world import Wall, World
from sim.scenarios.base import Scenario

# --- Landmark coordinates (m). Single source of truth per landmark. ---
X_HOLD_L = 0.0        # [ASSUMPTION] holding-area left wall
X_HOLD_R = 20.0       # [ASSUMPTION] holding right = taper mouth
X_THROAT_START = 24.0  # [ASSUMPTION] taper end / throat start
X_THROAT_END = 32.0   # [ASSUMPTION] throat end / platform start
X_PLATFORM_END = 52.0  # [ASSUMPTION; truncated] real NDLS platforms ~580 m [OSM]; modelled as a 20 m working section
X_GOAL = 53.0         # [ASSUMPTION] goal beyond the open far edge
Y_BOTTOM = 0.0        # [ASSUMPTION] holding bottom
Y_TOP = 20.0          # [ASSUMPTION] holding top
Y_PLATFORM_BOT = 3.0  # platform bottom rail; platform width 14 m sits in real NDLS range 11-24 m [OSM]
Y_PLATFORM_TOP = 17.0  # platform top rail [OSM-supported width]
Y_CENTER = 10.0       # shared centerline (holding, throat, platform)

# --- Agent spawn (m). [ASSUMPTION] inside holding area, set back from the taper. ---
SPAWN_AREA = (2.0, 14.0, 4.0, 16.0)  # (x_min, x_max, y_min, y_max)


class RailwayPlatformScenario(Scenario):
    """Holding area -> taper -> FOB throat -> platform, with an open far edge.

    The throat width is the project's key dimension (the bottleneck aperture).
    All throat-dependent coordinates are DERIVED from ``throat_width`` so a future
    width change cannot desync the walls, the Voronoi domain polygon, or the FOB
    zone.

    Args:
        n_agents: Number of agents spawned at t=0. [ASSUMPTION] default 150,
            within the practical <=200-agent ORCA cost cap.
        throat_width: FOB throat aperture (m). Default 3.66 = the legacy 12-ft FOB
            class [LEGACY] (the unsafe baseline that crushes). 6.0 = the CAG 2016
            mandated minimum [CAG] (the safe-target comparison, used in later phases).
        speed_mean: Mean desired speed (m/s). Default 1.34 = FZJ-calibrated CALM
            walking. Surge values ~1.8-2.5 model a hurried/urgent crowd
            [ASSUMPTION - surge state; Helbing et al. 2000 use 1.5-5 m/s for
            panic]. Phase-3 finding: calm crowds self-limit near ~3 ped/m^2
            regardless of throat width, so urgency is the lever that reaches the
            crush band.
        spawn_area: Optional (x_min, x_max, y_min, y_max) spawn box override.
            Default None = the frozen spec box (2, 14, 4, 16), mid-
            concourse. A proximity box near the taper mouth (e.g. (12, 19.5,
            3, 17)) models the crowd already CONVERGED at the FOB approach —
            the actual Feb-2025 condition (crowd accumulated at the stairs)
            [ASSUMPTION - surge convergence]. Phase-3 finding: the mid-
            concourse walk disperses arrival pressure; convergence is a
            structural lever for crush onset.
    """

    def __init__(self, n_agents: int = 150, throat_width: float = 3.66,
                 speed_mean: float = 1.34,
                 spawn_area: tuple[float, float, float, float] | None = None):
        self.n_agents = n_agents
        self.throat_width = throat_width
        self.speed_mean = speed_mean
        self._spawn_area = SPAWN_AREA if spawn_area is None else spawn_area

    @property
    def spawn_area(self) -> tuple[float, float, float, float]:
        """Spawn box (x_min, x_max, y_min, y_max) for agent placement."""
        return self._spawn_area

    @property
    def goal(self) -> np.ndarray:
        """Goal position (m), beyond the platform's open far edge."""
        return np.array([X_GOAL, Y_CENTER])

    def build(self, seed: int = 42) -> tuple[World, AgentState]:
        """Build the 11-wall world and spawn all agents in the holding area.

        Throat edges are computed as ``Y_CENTER -/+ throat_width/2`` (8.17 / 11.83
        when throat_width=3.66). There is intentionally NO far wall at x=52: the
        platform far edge is an open exit (see ``domain_polygon`` for the clip-only
        boundary).

        Args:
            seed: Random seed for agent placement.

        Returns:
            Tuple of (World with 11 walls, AgentState with n_agents active agents).
        """
        half = self.throat_width / 2.0
        throat_bot_y = Y_CENTER - half  # = 8.17 when W=3.66
        throat_top_y = Y_CENTER + half  # = 11.83 when W=3.66

        walls = [
            # 1 holding bottom
            Wall(np.array([X_HOLD_L, Y_BOTTOM]), np.array([X_HOLD_R, Y_BOTTOM])),
            # 2 holding top
            Wall(np.array([X_HOLD_L, Y_TOP]), np.array([X_HOLD_R, Y_TOP])),
            # 3 holding left
            Wall(np.array([X_HOLD_L, Y_BOTTOM]), np.array([X_HOLD_L, Y_TOP])),
            # 4 taper bottom (angled)
            Wall(np.array([X_HOLD_R, Y_BOTTOM]), np.array([X_THROAT_START, throat_bot_y])),
            # 5 taper top (angled)
            Wall(np.array([X_HOLD_R, Y_TOP]), np.array([X_THROAT_START, throat_top_y])),
            # 6 throat bottom
            Wall(np.array([X_THROAT_START, throat_bot_y]), np.array([X_THROAT_END, throat_bot_y])),
            # 7 throat top
            Wall(np.array([X_THROAT_START, throat_top_y]), np.array([X_THROAT_END, throat_top_y])),
            # 8 step-down (bottom)
            Wall(np.array([X_THROAT_END, throat_bot_y]), np.array([X_THROAT_END, Y_PLATFORM_BOT])),
            # 9 step-up (top)
            Wall(np.array([X_THROAT_END, throat_top_y]), np.array([X_THROAT_END, Y_PLATFORM_TOP])),
            # 10 platform bottom
            Wall(np.array([X_THROAT_END, Y_PLATFORM_BOT]), np.array([X_PLATFORM_END, Y_PLATFORM_BOT])),
            # 11 platform top
            Wall(np.array([X_THROAT_END, Y_PLATFORM_TOP]), np.array([X_PLATFORM_END, Y_PLATFORM_TOP])),
            # NO #12 — platform far edge (x=52) is an OPEN EXIT, never a wall.
        ]
        world = World(walls)

        state = AgentState.create(
            self.n_agents,
            spawn_area=self._spawn_area,
            goals=np.array([X_GOAL, Y_CENTER]),
            seed=seed,
            heterogeneous=True,
            speed_mean=self.speed_mean,
        )
        return world, state

    def domain_polygon(self) -> np.ndarray:
        """Walkable-region polygon for Voronoi density clipping, shape (12, 2).

        Counter-clockwise vertex order tracing holding -> taper -> throat ->
        platform and back. The far-edge vertices (52, 3) and (52, 17) are present
        ONLY as the density-clipping boundary; there is no physical wall there
        (the platform far edge is an open exit). Throat vertices are derived from
        ``throat_width``.

        Returns:
            Polygon vertices as a float array of shape (12, 2). Pass to
            ``VoronoiDensityEstimator(domain=...)``.
        """
        half = self.throat_width / 2.0
        throat_bot_y = Y_CENTER - half
        throat_top_y = Y_CENTER + half

        return np.array([
            [X_HOLD_L, Y_BOTTOM],            # (0, 0)
            [X_HOLD_R, Y_BOTTOM],            # (20, 0)
            [X_THROAT_START, throat_bot_y],  # (24, 8.17)
            [X_THROAT_END, throat_bot_y],    # (32, 8.17)
            [X_THROAT_END, Y_PLATFORM_BOT],  # (32, 3)
            [X_PLATFORM_END, Y_PLATFORM_BOT],  # (52, 3)  clip boundary only
            [X_PLATFORM_END, Y_PLATFORM_TOP],  # (52, 17) clip boundary only
            [X_THROAT_END, Y_PLATFORM_TOP],  # (32, 17)
            [X_THROAT_END, throat_top_y],    # (32, 11.83)
            [X_THROAT_START, throat_top_y],  # (24, 11.83)
            [X_HOLD_R, Y_TOP],               # (20, 20)
            [X_HOLD_L, Y_TOP],               # (0, 20)
        ], dtype=float)

    def fob_zone(self) -> tuple[float, float, float, float]:
        """FOB-throat bounding box (metadata for later phases; unused in Phase 1).

        Invariant: the y-extent is ``Y_CENTER -/+ throat_width/2``, so the box
        tracks the throat walls whenever ``throat_width`` changes. The x-extent is
        the throat corridor [X_THROAT_START, X_THROAT_END].

        Returns:
            (x_min, x_max, y_min, y_max) = (24, 32, 8.17, 11.83) when W=3.66.
        """
        half = self.throat_width / 2.0
        return (X_THROAT_START, X_THROAT_END, Y_CENTER - half, Y_CENTER + half)

    def is_complete(self, agent_state: AgentState, time: float) -> bool:
        """Check whether all agents have exited.

        Args:
            agent_state: Current agent state.
            time: Current simulation time (s); unused.

        Returns:
            True when no agents remain active.
        """
        return agent_state.n_active == 0
