"""Threshold detector: the warning brain. Consumes CrowdState grids only.

Watches a zone (grid cells inside a world-coordinate bbox, e.g. the FOB
throat) at the decision cadence and reports band crossings and warning lead
time. Thresholds are injected, never hard-coded — the detector is agnostic to
what density scale a venue/domain considers dangerous (spatial-risk-engine
neutrality).

Bands (frozen defaults):
  watch    >= 4.0 ped/m²  (ORCA-fade onset / Fruin LoS-E; lead-time anchor)
  amber    >= 5.0 ped/m²
  critical >= 5.5 ped/m²  (rho_crit; crush regime)

Lead time = t(critical first sustained) - t(watch first crossed): the window
an operator has between the earliest density-rise signal and crush onset.
"""

from dataclasses import dataclass, field

from sim.providers import CrowdState


@dataclass
class DetectorReading:
    """Detector output for one decision tick.

    Attributes:
        timestamp: Tick time (s).
        zone_peak: Max cell density (ped/m²) in the watched zone.
        band: "clear" | "watch" | "amber" | "critical".
        crush: True once critical has been sustained >= sustain_ticks.
        lead_time: t(critical, sustained) - t(watch onset); None until crush.
    """

    timestamp: float
    zone_peak: float
    band: str
    crush: bool
    lead_time: float | None


@dataclass
class ThresholdDetector:
    """Zone density detector with sustained-crossing logic.

    Args:
        zone_bbox: World-coordinate (x0, x1, y0, y1) to watch (e.g. fob_zone()).
        provider: Provider exposing ``zone_cells`` (grid registration).
        watch: Watch/lead-time-anchor threshold (ped/m²).
        amber: Amber pre-warning threshold (ped/m²).
        critical: Critical/crush threshold (ped/m²).
        sustain_ticks: Consecutive ticks >= critical to declare crush.
    """

    zone_bbox: tuple[float, float, float, float]
    provider: object
    watch: float = 4.0
    amber: float = 5.0
    critical: float = 5.5
    sustain_ticks: int = 3

    _t_watch: float | None = field(default=None, init=False)
    _t_crush: float | None = field(default=None, init=False)
    _streak: int = field(default=0, init=False)
    history: list = field(default_factory=list, init=False)

    def update(self, state: CrowdState) -> DetectorReading:
        """Process one CrowdState tick and return the reading.

        Args:
            state: Snapshot from any CrowdStateProvider.

        Returns:
            DetectorReading for this tick (also appended to ``history``).
        """
        iy, ix = self.provider.zone_cells(self.zone_bbox)
        zone = state.density_grid[iy, ix]
        peak = float(zone.max()) if zone.size else 0.0

        if peak >= self.critical:
            band = "critical"
            self._streak += 1
        else:
            self._streak = 0
            band = ("amber" if peak >= self.amber
                    else "watch" if peak >= self.watch else "clear")

        if self._t_watch is None and peak >= self.watch:
            self._t_watch = state.timestamp
        if self._t_crush is None and self._streak >= self.sustain_ticks:
            self._t_crush = state.timestamp

        lead = (self._t_crush - self._t_watch
                if self._t_crush is not None and self._t_watch is not None
                else None)
        reading = DetectorReading(state.timestamp, round(peak, 3), band,
                                  self._t_crush is not None, lead)
        self.history.append(reading)
        return reading
