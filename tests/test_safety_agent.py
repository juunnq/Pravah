"""Tests for the SafetyAgent decision layer (L3)."""

from sim.agent import SafetyAgent
from sim.detector import DetectorReading
from sim.forecaster import ZoneForecast


def tick(agent, t, peak, fpeak):
    r = DetectorReading(t, peak, "clear", False, None)
    f = ZoneForecast(t, 30.0, fpeak, 0.0)
    return agent.decide(r, f)


# 1. Quiet world -> no action.
def test_no_action_when_safe():
    a = SafetyAgent()
    for t in range(5):
        d = tick(a, t, 1.0, 1.2)
    assert d.action == "NONE" and not a.holding


# 2. Measured crossing triggers HOLD once (not repeatedly).
def test_hold_on_measured():
    a = SafetyAgent()
    d = tick(a, 0, 4.2, 2.0)
    assert d.action == "HOLD_RELEASE" and a.holding
    d = tick(a, 1, 4.5, 2.0)
    assert d.action == "NONE" and a.holding  # already holding


# 3. FORECAST crossing alone triggers HOLD — anticipatory autonomy.
def test_hold_on_forecast_only():
    a = SafetyAgent()
    d = tick(a, 0, 2.0, 4.5)
    assert d.action == "HOLD_RELEASE"
    assert "forecast" in d.reason


# 4. Resume requires sustained clearance (hysteresis), then fires once.
def test_resume_hysteresis():
    a = SafetyAgent(clear_ticks=3)
    tick(a, 0, 4.5, 4.5)
    assert a.holding
    tick(a, 1, 2.0, 2.0)      # streak 1
    tick(a, 2, 3.0, 2.0)      # dirty tick resets streak (peak >= clear_threshold)
    tick(a, 3, 2.0, 2.0)      # streak 1
    tick(a, 4, 2.0, 2.0)      # streak 2
    d = tick(a, 5, 2.0, 2.0)  # streak 3 -> resume
    assert d.action == "RESUME_RELEASE" and not a.holding


# 5. Abstention semantics: MEASURED overshoot -> ESCALATE (perception
#    untrustworthy); FORECAST overshoot -> clamped, act conservatively (HOLD).
def test_measured_overshoot_abstains():
    a = SafetyAgent(validity_ceiling=8.0)
    d = tick(a, 0, 9.5, 2.0)
    assert d.action == "ESCALATE" and not a.holding


def test_forecast_overshoot_holds_conservatively():
    a = SafetyAgent(validity_ceiling=8.0)
    d = tick(a, 0, 2.0, 12.0)   # ramp overshoot: prediction of extreme danger
    assert d.action == "HOLD_RELEASE" and a.holding
    assert "clamped" in d.reason


# 6. Full audit trail: every decision logged with evidence.
def test_audit_log():
    a = SafetyAgent()
    tick(a, 0, 1.0, 1.0)
    tick(a, 1, 4.5, 1.0)
    assert len(a.log) == 2
    assert a.actions_taken[0].action == "HOLD_RELEASE"
    assert a.actions_taken[0].reading.zone_peak == 4.5


# 7. The advisory voice: alarm sounds once (latched) at the amber band, and
#    every non-NONE decision carries an operator instruction.
def test_alarm_and_instructions():
    a = SafetyAgent()
    d = tick(a, 0, 4.2, 3.0)             # hold first
    assert d.action == "HOLD_RELEASE"
    assert "HOLD" in d.instruction and "/m²" in d.instruction
    d = tick(a, 1, 5.2, 5.0)              # amber -> alarm
    assert d.action == "SOUND_ALARM" and a.alarmed
    assert "ALARM" in d.instruction
    d = tick(a, 2, 5.4, 5.1)              # latched: no second alarm
    assert d.action == "NONE"


# 8. Forecast alone can sound the alarm (anticipatory advisory).
def test_alarm_on_forecast():
    a = SafetyAgent()
    d = tick(a, 0, 3.0, 5.5)
    # act threshold also crossed by forecast -> hold fires first...
    assert d.action == "HOLD_RELEASE"
    d = tick(a, 1, 3.0, 5.5)              # next tick: alarm
    assert d.action == "SOUND_ALARM"
    assert "Forecast" in d.instruction
