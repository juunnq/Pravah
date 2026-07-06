"""Property-based verification of the SafetyAgent decision logic.

Hypothesis generates thousands of arbitrary evidence streams; these
invariants must hold on EVERY one — the "systematically verified decision
logic" layer a certification review asks about. Each property is a sentence
a safety case can quote.
"""

from hypothesis import given, settings, strategies as st

from sim.agent import SafetyAgent
from sim.detector import DetectorReading
from sim.forecaster import ZoneForecast

APPROVED = {"NONE", "HOLD_RELEASE", "RESUME_RELEASE", "SOUND_ALARM", "ESCALATE"}

density = st.floats(min_value=0.0, max_value=20.0,
                    allow_nan=False, allow_infinity=False)
stream = st.lists(st.tuples(density, density), min_size=1, max_size=120)


def run_stream(agent, pairs, dt=1.0):
    out = []
    for i, (peak, fpeak) in enumerate(pairs):
        r = DetectorReading(i * dt, round(peak, 2), "clear", False, None)
        f = ZoneForecast(i * dt, 30.0, round(fpeak, 2), 0.0)
        out.append(agent.decide(r, f))
    return out


# P1: the agent only ever emits pre-approved actions, and every non-NONE
#     decision carries an instruction.
@settings(max_examples=300, deadline=None)
@given(stream)
def test_only_approved_actions(pairs):
    ds = run_stream(SafetyAgent(), pairs)
    for d in ds:
        assert d.action in APPROVED
        if d.action != "NONE":
            assert d.instruction


# P2: measured evidence beyond the validity ceiling NEVER produces an
#     actuation — abstention always wins over action on corrupt input.
@settings(max_examples=300, deadline=None)
@given(stream)
def test_corrupt_input_never_actuates(pairs):
    a = SafetyAgent()
    for d in run_stream(a, pairs):
        if d.reading.zone_peak > a.validity_ceiling:
            assert d.action == "ESCALATE"


# P3: RESUME only ever fires after clear_ticks consecutive genuinely-clear
#     ticks — never while the zone is (or is forecast) at/above the clear bar.
@settings(max_examples=300, deadline=None)
@given(stream)
def test_resume_requires_sustained_clearance(pairs):
    a = SafetyAgent(clear_ticks=3)
    ds = run_stream(a, pairs)
    for i, d in enumerate(ds):
        if d.action == "RESUME_RELEASE":
            window = ds[i - 2:i + 1] if i >= 2 else []
            assert len(window) == 3
            for w in window:
                assert w.reading.zone_peak < a.clear_threshold
                assert min(w.forecast.zone_peak_pred,
                           a.validity_ceiling) < a.clear_threshold


# P4: the alarm latches — at most one SOUND_ALARM per stream.
@settings(max_examples=300, deadline=None)
@given(stream)
def test_alarm_fires_at_most_once(pairs):
    ds = run_stream(SafetyAgent(), pairs)
    assert sum(1 for d in ds if d.action == "SOUND_ALARM") <= 1


# P5: every decision is logged; the log is complete and ordered.
@settings(max_examples=200, deadline=None)
@given(stream)
def test_audit_log_complete(pairs):
    a = SafetyAgent()
    ds = run_stream(a, pairs)
    assert len(a.log) == len(ds)
    assert [d.timestamp for d in a.log] == sorted(d.timestamp for d in a.log)


# P6: a gateless (advisory-only) venue NEVER receives an actuation command,
#     under any evidence whatsoever.
@settings(max_examples=300, deadline=None)
@given(stream)
def test_gateless_venue_never_actuated(pairs):
    ds = run_stream(SafetyAgent(actuation=False), pairs)
    for d in ds:
        assert d.action not in ("HOLD_RELEASE", "RESUME_RELEASE")


# P7: instructions never quote a density beyond the validated domain.
@settings(max_examples=300, deadline=None)
@given(stream)
def test_instructions_never_quote_invalid_densities(pairs):
    import re
    a = SafetyAgent()
    for d in run_stream(a, pairs):
        for num in re.findall(r"(\d+\.?\d*)\+?/m²", d.instruction):
            assert float(num) <= a.validity_ceiling


# P8: sensor gaps larger than the timeout always escalate (stale evidence
#     never drives an action).
@settings(max_examples=200, deadline=None)
@given(stream, st.floats(min_value=6.0, max_value=100.0, allow_nan=False))
def test_sensor_gap_escalates(pairs, gap):
    a = SafetyAgent()
    r = DetectorReading(0.0, 1.0, "clear", False, None)
    f = ZoneForecast(0.0, 30.0, 1.0, 0.0)
    a.decide(r, f)
    r2 = DetectorReading(gap + a.sensor_timeout, 1.0, "clear", False, None)
    f2 = ZoneForecast(gap + a.sensor_timeout, 30.0, 1.0, 0.0)
    d = a.decide(r2, f2)
    assert d.action == "ESCALATE"
    assert "gap" in d.reason
