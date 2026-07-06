"""Tests for the CrowdStateProvider seam and ThresholdDetector (Phase 4)."""

import numpy as np
import pytest

from sim.detector import ThresholdDetector
from sim.providers import CCTVStateProvider, CrowdState, SimulationStateProvider
from sim.scenarios.railway_platform import RailwayPlatformScenario


@pytest.fixture()
def provider():
    return SimulationStateProvider(RailwayPlatformScenario(), cell_size=1.0)


# 1. Rasterization: known positions land in the right cells at the right density.
def test_rasterization_counts(provider):
    pos = np.array([[5.5, 5.5], [5.7, 5.3], [10.5, 10.5]])  # 2 in one cell
    state = provider.sample(1.0, pos)
    assert state.density_grid[5, 5] == pytest.approx(2.0)   # 2 agents / 1 m²
    assert state.density_grid[10, 10] == pytest.approx(1.0)
    assert state.density_grid.sum() == pytest.approx(3.0)


# 2. Out-of-domain agents are excluded (Phase-3 lesson).
def test_out_of_domain_excluded(provider):
    pos = np.array([[5.5, 5.5], [52.5, 10.0], [30.0, 19.0]])
    # (52.5, 10) is past the open far edge; (30, 19) is outside the throat walls.
    state = provider.sample(0.0, pos)
    assert state.density_grid.sum() == pytest.approx(1.0)


# 3. Zone registration: fob_zone bbox maps to the correct grid slices.
def test_zone_cells(provider):
    scenario = provider.scenario
    iy, ix = provider.zone_cells(scenario.fob_zone())
    assert ix == slice(24, 32)
    # y in [8.17, 11.83] at 1 m cells -> rows 8..11 inclusive
    assert iy == slice(8, 12)


# 4. Detector bands and sustained-crush logic on a synthetic ramp.
def test_detector_bands_and_lead_time(provider):
    det = ThresholdDetector(provider.scenario.fob_zone(), provider)
    iy, ix = provider.zone_cells(provider.scenario.fob_zone())

    def tick(t, dens):
        grid = np.zeros((provider._ny, provider._nx))
        grid[iy.start, ix.start] = dens
        return det.update(CrowdState(t, grid, 1.0, (0, 0), provider.scenario))

    assert tick(0, 1.0).band == "clear"
    assert tick(1, 4.2).band == "watch"       # watch onset at t=1
    assert tick(2, 5.1).band == "amber"
    r3, r4 = tick(3, 5.6), tick(4, 5.7)
    assert r3.band == "critical" and not r3.crush   # streak 1
    assert not r4.crush                              # streak 2
    r5 = tick(5, 5.8)                                # streak 3 -> crush
    assert r5.crush
    assert r5.lead_time == pytest.approx(5 - 1)      # crush t=5, watch t=1
    # Band drop resets the streak but crush latches.
    r6 = tick(6, 3.0)
    assert r6.band == "clear" and r6.crush


# 5. Streak resets: 2 critical ticks, a dip, then 2 more != sustained.
def test_detector_streak_reset(provider):
    det = ThresholdDetector(provider.scenario.fob_zone(), provider)
    iy, ix = provider.zone_cells(provider.scenario.fob_zone())

    def tick(t, dens):
        grid = np.zeros((provider._ny, provider._nx))
        grid[iy.start, ix.start] = dens
        return det.update(CrowdState(t, grid, 1.0, (0, 0), provider.scenario))

    for t, d in [(0, 5.6), (1, 5.6), (2, 4.0), (3, 5.6), (4, 5.6)]:
        r = tick(t, d)
    assert not r.crush


# 6. CCTV stub is a transparent proxy: identical output, same interface.
def test_cctv_stub_proxies(provider):
    cctv = CCTVStateProvider(provider)
    pos = np.array([[5.5, 5.5], [10.5, 10.5], [11.5, 10.5], [12.5, 10.5]])
    a = provider.sample(2.0, pos)
    b = cctv.sample(2.0, pos)
    assert np.array_equal(a.density_grid, b.density_grid)
    assert cctv.zone_cells((24, 32, 8, 12)) == provider.zone_cells((24, 32, 8, 12))


# 7. Detector is source-agnostic: same readings from sim and CCTV providers.
def test_detector_source_agnostic(provider):
    cctv = CCTVStateProvider(provider)
    pos = np.tile(np.array([[28.5, 10.5]]), (6, 1))  # 6 agents, one FOB cell
    for src in (provider, cctv):
        det = ThresholdDetector(provider.scenario.fob_zone(), src)
        r = det.update(src.sample(0.0, pos))
        assert r.zone_peak == pytest.approx(6.0)
        assert r.band == "critical"
