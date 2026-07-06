"""CrowdStateProvider seam: the brain reads density grids, never agent arrays.

This is the architectural boundary that lets a live CCTV feed replace the
simulator without rewriting the detector/forecaster. Providers
yield CrowdState snapshots at a decision cadence, decoupled from the sim's
dt and from any camera frame rate.

    CrowdStateProvider (interface)
      ├── SimulationStateProvider   # real: rasterize sim agents -> grid
      └── CCTVStateProvider         # STUB: proxies a SimulationStateProvider;
                                    #   TODOs = density CNN, homography, frames

Grid convention: density_grid[iy, ix] in ped/m², cell (ix, iy) covers
[origin_x + ix*cell, origin_x + (ix+1)*cell) × [origin_y + iy*cell, ...).
Agents outside the walkable domain polygon are EXCLUDED before rasterization
(Phase-3 lesson: out-of-domain agents corrupt density statistics).

# ponytail: plain count/cell_area rasterization — a CCTV density CNN yields
# the same form; upgrade to kernel-spread rasterization if 1 m cells prove
# too quantized for the forecaster.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CrowdState:
    """One crowd-state snapshot at a decision tick.

    Attributes:
        timestamp: Time (s) of the snapshot.
        density_grid: ped/m² per cell, shape (H, W), row iy = y-axis.
        cell_size: Cell edge length (m).
        origin: (x_min, y_min) of grid cell (0, 0).
        geometry_ref: The scenario object this grid is registered to.
    """

    timestamp: float
    density_grid: np.ndarray
    cell_size: float
    origin: tuple[float, float]
    geometry_ref: object


class SimulationStateProvider:
    """Rasterizes simulator agent positions into CrowdState grids.

    Args:
        scenario: Scenario exposing ``domain_polygon()`` (walkable region) —
            used for the grid extent and for excluding out-of-domain agents.
        cell_size: Grid cell edge (m). Default 1.0.
    """

    def __init__(self, scenario, cell_size: float = 1.0,
                 domain: "np.ndarray | None" = None):
        self.scenario = scenario
        self.cell_size = cell_size
        # Domain-neutrality: any venue can supply its walkable polygon
        # directly; scenarios with a domain_polygon() method remain the default.
        domain = domain if domain is not None else scenario.domain_polygon()
        self._x0 = float(domain[:, 0].min())
        self._y0 = float(domain[:, 1].min())
        self._nx = int(np.ceil((domain[:, 0].max() - self._x0) / cell_size))
        self._ny = int(np.ceil((domain[:, 1].max() - self._y0) / cell_size))
        from shapely.geometry import Polygon
        self._domain_poly = Polygon(domain)

    def sample(self, timestamp: float, positions: np.ndarray) -> CrowdState:
        """Build a CrowdState from active-agent positions.

        Args:
            timestamp: Current time (s).
            positions: Active agent positions, shape (N, 2).

        Returns:
            CrowdState with the rasterized density grid.
        """
        from shapely.geometry import Point

        grid = np.zeros((self._ny, self._nx))
        area = self.cell_size ** 2
        for p in positions:
            if not self._domain_poly.contains(Point(p[0], p[1])):
                continue  # out-of-domain agents excluded (Phase-3 lesson)
            ix = int((p[0] - self._x0) / self.cell_size)
            iy = int((p[1] - self._y0) / self.cell_size)
            if 0 <= ix < self._nx and 0 <= iy < self._ny:
                grid[iy, ix] += 1.0 / area
        return CrowdState(timestamp, grid, self.cell_size,
                          (self._x0, self._y0), self.scenario)

    def zone_cells(self, bbox: tuple[float, float, float, float]
                   ) -> tuple[slice, slice]:
        """Grid slices covering a world-coordinate bbox (x0, x1, y0, y1).

        Returns:
            (iy_slice, ix_slice) usable as density_grid[iy_slice, ix_slice].
        """
        x0, x1, y0, y1 = bbox
        ix0 = max(0, int((x0 - self._x0) / self.cell_size))
        ix1 = min(self._nx, int(np.ceil((x1 - self._x0) / self.cell_size)))
        iy0 = max(0, int((y0 - self._y0) / self.cell_size))
        iy1 = min(self._ny, int(np.ceil((y1 - self._y0) / self.cell_size)))
        return slice(iy0, iy1), slice(ix0, ix1)


class CCTVStateProvider:
    """STUB — proxies a simulation provider (kept for tests/back-compat).

    The REAL camera path is ``VideoCCTVProvider`` below. The interface is
    final: consumers receive CrowdState and cannot tell (and must not care)
    which provider produced it.
    """

    def __init__(self, proxy: SimulationStateProvider):
        self._proxy = proxy
        self.cell_size = proxy.cell_size

    def sample(self, timestamp: float, positions: np.ndarray) -> CrowdState:
        """Proxy to the simulation provider (stub)."""
        return self._proxy.sample(timestamp, positions)

    def zone_cells(self, bbox: tuple[float, float, float, float]
                   ) -> tuple[slice, slice]:
        """Proxy to the simulation provider (stub)."""
        return self._proxy.zone_cells(bbox)


class VideoCCTVProvider:
    """The REAL camera provider: video frame -> CrowdState density grid.

    Composes a crowd-density model (``sim.perception.CrowdDensityModel``) with
    a per-camera ``HomographyCalibrator``. Same contract as the other
    providers, so ThresholdDetector / ZoneForecaster run unchanged — this is
    the seam paying off.

    Remaining deployment TODOs (architected, not built): RTSP frame loop,
    multi-camera fusion into one grid.

    Args:
        density_model: Object with ``estimate(frame_bgr) -> density map``.
        calibrator: ``HomographyCalibrator`` for this camera (defines the
            world extent and cell size of the output grid).
        geometry_ref: Optional venue object (e.g. a Scenario) for consumers
            that want zone definitions.
    """

    def __init__(self, density_model, calibrator, geometry_ref=None):
        self.model = density_model
        self.cal = calibrator
        self.cell_size = calibrator.cell_m
        self.geometry_ref = geometry_ref

    def state_from_frame(self, timestamp: float,
                         frame_bgr: np.ndarray) -> CrowdState:
        """One video frame -> CrowdState (the camera-side ``sample``)."""
        dmap = self.model.estimate(frame_bgr)
        grid = self.cal.density_to_grid(dmap, frame_shape=frame_bgr.shape[:2])
        return CrowdState(timestamp, grid, self.cal.cell_m,
                          (self.cal.extent[0], self.cal.extent[2]),
                          self.geometry_ref)

    def zone_cells(self, bbox: tuple[float, float, float, float]
                   ) -> tuple[slice, slice]:
        """Grid slices for a world-coordinate bbox (same math as sim provider)."""
        x0, x1, y0, y1 = bbox
        ex0, _, ey0, _ = self.cal.extent
        cs = self.cal.cell_m
        ix0 = max(0, int((x0 - ex0) / cs))
        ix1 = min(self.cal.nx, int(np.ceil((x1 - ex0) / cs)))
        iy0 = max(0, int((y0 - ey0) / cs))
        iy1 = min(self.cal.ny, int(np.ceil((y1 - ey0) / cs)))
        return slice(iy0, iy1), slice(ix0, ix1)
