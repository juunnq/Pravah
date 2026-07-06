"""Zone density forecaster: predicts near-future crowd state from grid history.

Consumes the provider seam's CrowdState sequence — never agent arrays — so it
is CCTV-ready by construction (same contract as the detector). Two models:

  RateOfRiseForecaster  — the mandatory baseline: linear extrapolation of the
                          zone signal over a trailing window. Any learned model
                          must beat this on held-out windows or it doesn't ship
                          (ML-discipline ladder).
  ZoneForecaster        — sklearn regressor over trailing-window features.
                          Trained offline on the Phase-8 1-Hz trajectory
                          corpus; scored against the baseline.

The forecast target is deliberately simple for the stub: the zone's
pressure-column occupancy and grid-peak `horizon` seconds ahead. Both models
share the same `update(state) -> ZoneForecast` streaming interface.

Honesty bounds: trained on simulation, the forecaster inherits the engine's
validity ceiling (~5 ped/m² on the grid) and its OOD bias (over-predicts
congestion). Labels are sim-relative until per-site calibration.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sim.providers import CrowdState


@dataclass
class ZoneForecast:
    """Forecast for one decision tick.

    Attributes:
        timestamp: Tick time (s).
        horizon: Seconds ahead the prediction refers to.
        zone_peak_pred: Predicted zone peak density (ped/m²) at t + horizon.
        occupancy_pred: Predicted zone occupant count at t + horizon.
    """

    timestamp: float
    horizon: float
    zone_peak_pred: float
    occupancy_pred: float


@dataclass
class RateOfRiseForecaster:
    """Baseline: linear extrapolation of zone peak/occupancy over a window.

    Args:
        zone_bbox: World-coordinate (x0, x1, y0, y1) zone to watch.
        provider: Provider exposing ``zone_cells`` (grid registration).
        horizon: Forecast lead (s).
        window: Trailing window length in ticks used for the linear fit.
    """

    zone_bbox: tuple[float, float, float, float]
    provider: object
    horizon: float = 30.0
    window: int = 30

    _t: deque = field(default_factory=lambda: deque(maxlen=120), init=False)
    _peak: deque = field(default_factory=lambda: deque(maxlen=120), init=False)
    _occ: deque = field(default_factory=lambda: deque(maxlen=120), init=False)

    def _observe(self, state: CrowdState) -> tuple[float, float]:
        iy, ix = self.provider.zone_cells(self.zone_bbox)
        zone = state.density_grid[iy, ix]
        peak = float(zone.max()) if zone.size else 0.0
        occ = float(zone.sum() * state.cell_size ** 2)  # cells are ped/m²
        self._t.append(state.timestamp)
        self._peak.append(peak)
        self._occ.append(occ)
        return peak, occ

    @staticmethod
    def _extrapolate(ts: np.ndarray, ys: np.ndarray, t_pred: float) -> float:
        if len(ts) < 2:
            return float(ys[-1]) if len(ys) else 0.0
        slope, intercept = np.polyfit(ts, ys, 1)
        return float(max(0.0, slope * t_pred + intercept))

    def update(self, state: CrowdState) -> ZoneForecast:
        """Ingest one CrowdState tick; return the forecast for t + horizon."""
        self._observe(state)
        n = min(self.window, len(self._t))
        ts = np.array(self._t)[-n:]
        t_pred = state.timestamp + self.horizon
        return ZoneForecast(
            state.timestamp, self.horizon,
            round(self._extrapolate(ts, np.array(self._peak)[-n:], t_pred), 3),
            round(self._extrapolate(ts, np.array(self._occ)[-n:], t_pred), 3),
        )


@dataclass
class ZoneForecaster(RateOfRiseForecaster):
    """Learned forecaster: sklearn regressor over trailing-window features.

    Features per tick: the last `window` values of (peak, occupancy) plus
    their first differences. Targets: (peak, occupancy) at t + horizon.
    Falls back to the rate-of-rise prediction until `fit` has been called.

    Train offline with :meth:`fit` on windows harvested from the Phase-8
    corpus (see scripts/phase8_train.py), then stream via ``update``.
    """

    model: object = field(default=None, init=False)

    def features(self) -> np.ndarray | None:
        """Trailing-window feature vector, or None until the window fills."""
        if len(self._peak) < self.window:
            return None
        p = np.array(self._peak)[-self.window:]
        o = np.array(self._occ)[-self.window:]
        return np.concatenate([p, o, np.diff(p), np.diff(o)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ZoneForecaster":
        """Fit the regressor (y columns: peak, occupancy at t + horizon)."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        self.model = MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=0))
        self.model.fit(X, y)
        return self

    def update(self, state: CrowdState) -> ZoneForecast:
        """Ingest one tick; learned prediction, or baseline until ready."""
        base = super().update(state)
        feats = self.features()
        if self.model is None or feats is None:
            return base
        peak, occ = self.model.predict(feats.reshape(1, -1))[0]
        return ZoneForecast(state.timestamp, self.horizon,
                            round(max(0.0, float(peak)), 3),
                            round(max(0.0, float(occ)), 3))
