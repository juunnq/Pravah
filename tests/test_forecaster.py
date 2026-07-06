"""Tests for the zone forecaster stub (Phase 9): baseline + learned model."""

import numpy as np
import pytest

from sim.forecaster import RateOfRiseForecaster, ZoneForecaster
from sim.providers import CrowdState, SimulationStateProvider
from sim.scenarios.railway_platform import RailwayPlatformScenario

ZONE = (24.0, 32.0, 8.0, 12.0)


@pytest.fixture()
def provider():
    return SimulationStateProvider(RailwayPlatformScenario(), cell_size=1.0)


def make_state(provider, t, dens):
    """One CrowdState with a single zone cell at the given density."""
    grid = np.zeros((provider._ny, provider._nx))
    iy, ix = provider.zone_cells(ZONE)
    grid[iy.start, ix.start] = dens
    return CrowdState(t, grid, 1.0, (0, 0), provider.scenario)


# 1. Baseline extrapolates a linear ramp correctly.
def test_baseline_linear_ramp(provider):
    fc = RateOfRiseForecaster(ZONE, provider, horizon=10.0, window=30)
    for t in range(30):
        f = fc.update(make_state(provider, float(t), 0.1 * t))
    # ramp 0.1/s -> at t=29+10 expect peak ~3.9
    assert f.zone_peak_pred == pytest.approx(3.9, abs=0.05)


# 2. Baseline on a flat signal predicts the flat value (no spurious trend).
def test_baseline_flat(provider):
    fc = RateOfRiseForecaster(ZONE, provider, horizon=30.0)
    for t in range(40):
        f = fc.update(make_state(provider, float(t), 2.5))
    assert f.zone_peak_pred == pytest.approx(2.5, abs=0.01)


# 3. Predictions are clamped at zero on a falling signal.
def test_baseline_nonnegative(provider):
    fc = RateOfRiseForecaster(ZONE, provider, horizon=60.0, window=10)
    for t in range(10):
        f = fc.update(make_state(provider, float(t), max(0.0, 2.0 - 0.3 * t)))
    assert f.zone_peak_pred >= 0.0


# 4. Learned forecaster falls back to baseline until fitted + window full.
def test_learned_fallback(provider):
    fc = ZoneForecaster(ZONE, provider, horizon=10.0, window=30)
    f = fc.update(make_state(provider, 0.0, 1.0))
    assert f.zone_peak_pred >= 0.0  # baseline path, no crash


# 5. Learned forecaster beats the baseline on a saturating (nonlinear) signal.
def test_learned_beats_baseline_on_saturation(provider):
    # Signal: logistic saturation at 4.0 — linear extrapolation overshoots.
    def sig(t):
        return 4.0 / (1.0 + np.exp(-(t - 30) / 8.0))

    window, horizon = 30, 10
    # Harvest training windows from many phase-shifted copies.
    X, y = [], []
    for shift in range(0, 60, 3):
        fc = ZoneForecaster(ZONE, provider, horizon=horizon, window=window)
        vals = [sig(t + shift) for t in range(90)]
        for t in range(90 - horizon):
            fc.update(make_state(provider, float(t), vals[t]))
            feats = fc.features()
            if feats is not None:
                X.append(feats)
                y.append([vals[t + horizon], vals[t + horizon]])
    model = ZoneForecaster(ZONE, provider, horizon=horizon, window=window)
    model.fit(np.array(X), np.array(y))

    base = RateOfRiseForecaster(ZONE, provider, horizon=horizon, window=window)
    err_m, err_b = [], []
    vals = [sig(t + 1.5) for t in range(90)]  # held-out phase
    for t in range(90 - horizon):
        s = make_state(provider, float(t), vals[t])
        fm, fb = model.update(s), base.update(s)
        if t >= window:
            truth = vals[t + horizon]
            err_m.append(abs(fm.zone_peak_pred - truth))
            err_b.append(abs(fb.zone_peak_pred - truth))
    assert np.mean(err_m) < np.mean(err_b), \
        f"learned MAE {np.mean(err_m):.3f} vs baseline {np.mean(err_b):.3f}"


# 6. Source-agnostic: identical forecasts from sim provider and CCTV stub.
def test_forecaster_source_agnostic(provider):
    from sim.providers import CCTVStateProvider
    cctv = CCTVStateProvider(provider)
    f1 = RateOfRiseForecaster(ZONE, provider, horizon=10.0)
    f2 = RateOfRiseForecaster(ZONE, cctv, horizon=10.0)
    for t in range(10):
        s = make_state(provider, float(t), 0.5 * t)
        a, b = f1.update(s), f2.update(s)
    assert a.zone_peak_pred == b.zone_peak_pred
