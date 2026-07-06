"""SafetyAgent: the L3 decision layer — conditional autonomy with abstention.

Composes the detector reading and forecast into a specific, pre-approved
action, or explicitly abstains. This is deliberately NOT a learned policy:
actions come from a fixed, auditable playbook; the agent's intelligence is in
WHEN to invoke them and when to refuse to act (validity-bound abstention —
the property that makes autonomy certifiable).

Action set (all reversible, machine-actuatable — gate/turnstile metering):
  HOLD_RELEASE   pause pending crowd-release batches
  RESUME_RELEASE resume them once the zone has cleared
  NONE           no action needed
  ESCALATE       abstain: outside validated domain -> human judgment

Every decision is logged with its full evidence (reading, forecast, rule
fired) — the audit trail an override console and a certification review need.
"""

from dataclasses import dataclass, field

from sim.detector import DetectorReading
from sim.forecaster import ZoneForecast


@dataclass
class Decision:
    """One agent decision with its evidence.

    Attributes:
        timestamp: Decision time (s).
        action: "NONE" | "HOLD_RELEASE" | "RESUME_RELEASE" | "SOUND_ALARM" |
            "ESCALATE".
        reason: Human-readable rule trace (the audit line).
        instruction: Operator-facing advisory sentence (the spoken call) —
            empty for NONE.
        reading: The detector reading it acted on.
        forecast: The forecast it acted on.
    """

    timestamp: float
    action: str
    reason: str
    reading: DetectorReading
    forecast: ZoneForecast
    instruction: str = ""


@dataclass
class SafetyAgent:
    """Rule-based conditional-autonomy agent (L3) over the detector+forecaster.

    Args:
        act_threshold: Zone density (ped/m²) at/above which (measured OR
            forecast) the agent holds releases. Default = the watch band.
        clear_threshold: Density below which (measured AND forecast) it
            resumes. Hysteresis gap prevents oscillation.
        clear_ticks: Consecutive clear ticks required before resuming.
        validity_ceiling: Densities above this are OUTSIDE the perception/
            physics validated domain -> the agent abstains (ESCALATE) rather
            than trusting numbers it cannot certify.
    """

    act_threshold: float = 4.0
    alarm_threshold: float = 5.0   # amber: sound the venue alarm (advisory)
    clear_threshold: float = 2.5
    clear_ticks: int = 10
    validity_ceiling: float = 8.0
    sensor_timeout: float = 5.0    # max gap (s) between readings before the
                                   # evidence is STALE -> escalate. (For total
                                   # feed loss the deployment loop needs its
                                   # own heartbeat; this catches stream gaps.)
    # Venue-supplied playbook (the Domain-2 lesson: action vocabularies are
    # venue config, like geometry and thresholds — never baked in):
    actuation: bool = True         # False = gateless venue: advisory only,
                                   # HOLD/RESUME are never emitted.
    instruction_overrides: dict | None = None  # per-venue instruction text
                                               # by action name.

    holding: bool = field(default=False, init=False)
    alarmed: bool = field(default=False, init=False)   # alarm latches
    _clear_streak: int = field(default=0, init=False)
    _last_t: float | None = field(default=None, init=False)
    log: list = field(default_factory=list, init=False)

    def _advise(self, action: str, reading, forecast) -> str:
        """Compose the operator-facing call. Quotes the CLAMPED forecast —
        an instruction must never cite a density beyond the validated domain.
        Venue instruction_overrides take precedence over the defaults."""
        if self.instruction_overrides and action in self.instruction_overrides:
            return self.instruction_overrides[action].format(
                peak=reading.zone_peak,
                forecast=min(forecast.zone_peak_pred, self.validity_ceiling))
        fpeak = min(forecast.zone_peak_pred, self.validity_ceiling)
        lead = ""
        if fpeak > reading.zone_peak:
            more = ("+" if forecast.zone_peak_pred > self.validity_ceiling
                    else "")
            lead = (f" Forecast reaches {fpeak:.1f}{more}/m² "
                    f"within {forecast.horizon:.0f}s.")
        return {
            "HOLD_RELEASE": (
                f"HOLD all pending releases NOW — zone at "
                f"{reading.zone_peak:.1f}/m² and rising.{lead} "
                f"Phased holding is the highest-impact intervention "
                f"(physics playbook: −56% crowd-at-risk)."),
            "RESUME_RELEASE": (
                "RESUME releases, one metered batch — zone has stayed clear. "
                "Watching the probe batch; will re-hold on any rise."),
            "SOUND_ALARM": (
                f"SOUND VENUE ALARM — zone density {reading.zone_peak:.1f}/m² "
                f"in the amber band.{lead} Deploy staff to the throat; "
                f"prepare alternate routing."),
            "ESCALATE": (
                "ESCALATE TO OPERATOR — sensor evidence outside the validated "
                "domain; automated action suspended. Manual assessment "
                "required."),
        }.get(action, "")

    def decide(self, reading: DetectorReading, forecast: ZoneForecast) -> Decision:
        """One decision tick. Returns (and logs) the Decision."""
        t = reading.timestamp
        peak = reading.zone_peak
        fpeak = forecast.zone_peak_pred

        # Sensor watchdog: a gap in the evidence stream means the world moved
        # while we were blind — stale-evidence actions are uncertifiable.
        gap = (t - self._last_t) if self._last_t is not None else 0.0
        self._last_t = t
        if gap > self.sensor_timeout:
            self._clear_streak = 0  # stale evidence voids clearance progress
            d = Decision(t, "ESCALATE",
                         f"sensor gap {gap:.1f}s > {self.sensor_timeout}s — "
                         f"evidence stale, abstaining until stream recovers",
                         reading, forecast,
                         self._advise("ESCALATE", reading, forecast))
            self.log.append(d)
            return d

        # Abstention: MEASURED evidence outside the validated domain means the
        # perception itself cannot be trusted — abstain. A FORECAST overshoot,
        # by contrast, is an overconfident prediction of extreme danger, not
        # corrupt input: clamp it and act conservatively (holding a release is
        # low-regret). Freezing during a ramp is the one uncertifiable behavior.
        if peak > self.validity_ceiling:
            self._clear_streak = 0  # corrupt evidence voids clearance progress
            d = Decision(t, "ESCALATE",
                         f"MEASURED density {peak} beyond validated domain "
                         f"(>{self.validity_ceiling}) — perception "
                         f"untrustworthy, abstaining", reading, forecast,
                         self._advise("ESCALATE", reading, forecast))
            self.log.append(d)
            return d
        clamped = fpeak > self.validity_ceiling
        if clamped:
            fpeak = self.validity_ceiling

        # Venue alarm (advisory, latched once). ACTUATION outranks ADVISORY on
        # a shared tick: the gate closes now, the announcement follows next
        # tick (the state machine below takes precedence; alarm fills NONE
        # ticks only).
        alarm_due = (not self.alarmed
                     and (peak >= self.alarm_threshold
                          or fpeak >= self.alarm_threshold))

        if not self.actuation:
            # Gateless venue: advisory channel only.
            d = Decision(t, "NONE", "advisory-only venue", reading, forecast)
        elif not self.holding:
            if peak >= self.act_threshold or fpeak >= self.act_threshold:
                self.holding = True
                self._clear_streak = 0
                trigger = ("measured" if peak >= self.act_threshold
                           else f"forecast(+{forecast.horizon:.0f}s)"
                           + (" [overshoot clamped]" if clamped else ""))
                d = Decision(t, "HOLD_RELEASE",
                             f"{trigger} zone density "
                             f"{max(peak, fpeak):.1f} >= {self.act_threshold} "
                             f"— holding pending releases", reading, forecast,
                             self._advise("HOLD_RELEASE", reading, forecast))
            else:
                d = Decision(t, "NONE", "within safe bounds", reading, forecast)
        else:
            if peak < self.clear_threshold and fpeak < self.clear_threshold:
                self._clear_streak += 1
            else:
                self._clear_streak = 0
            if self._clear_streak >= self.clear_ticks:
                self.holding = False
                d = Decision(t, "RESUME_RELEASE",
                             f"zone clear (<{self.clear_threshold}) for "
                             f"{self.clear_ticks} ticks — resuming releases",
                             reading, forecast,
                             self._advise("RESUME_RELEASE", reading, forecast))
            else:
                d = Decision(t, "NONE",
                             f"holding (clear streak "
                             f"{self._clear_streak}/{self.clear_ticks})",
                             reading, forecast)
        if d.action == "NONE" and alarm_due:
            self.alarmed = True
            d = Decision(t, "SOUND_ALARM",
                         f"zone density {max(peak, fpeak):.1f} in the amber "
                         f"band (>= {self.alarm_threshold})", reading,
                         forecast,
                         self._advise("SOUND_ALARM", reading, forecast))
        self.log.append(d)
        return d

    @property
    def actions_taken(self) -> list:
        """Only the non-NONE decisions (the audit trail of interventions)."""
        return [d for d in self.log if d.action != "NONE"]
