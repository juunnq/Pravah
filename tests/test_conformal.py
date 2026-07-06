"""Tests for split-conformal intervals: the coverage guarantee must hold."""

import numpy as np
import pytest

from sim.conformal import SplitConformal


# 1. Coverage guarantee on exchangeable data: >= 1-alpha (within tolerance).
@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_coverage_guarantee(alpha):
    rng = np.random.default_rng(0)
    y = rng.normal(50, 10, 4000)
    pred = y + rng.normal(0, 5, 4000)   # predictor with noise-5 residuals
    cal_res = np.abs(y[:2000] - pred[:2000])
    sc = SplitConformal(alpha=alpha).fit(cal_res)
    cov = SplitConformal.empirical_coverage(y[2000:], pred[2000:], sc.q)
    assert cov >= (1 - alpha) - 0.03    # finite-sample slack


# 2. Intervals are symmetric and centered on the prediction.
def test_interval_shape():
    sc = SplitConformal(alpha=0.1).fit(np.array([1.0, 2.0, 3.0, 4.0]))
    lo, hi = sc.interval(10.0)
    assert lo == pytest.approx(10.0 - sc.q)
    assert hi == pytest.approx(10.0 + sc.q)
    assert sc.upper(10.0) == hi


# 3. Smaller alpha (more confidence demanded) -> wider intervals.
def test_alpha_monotonicity():
    rng = np.random.default_rng(1)
    res = np.abs(rng.normal(0, 3, 500))
    q90 = SplitConformal(alpha=0.1).fit(res).q
    q80 = SplitConformal(alpha=0.2).fit(res).q
    assert q90 >= q80


# 4. Guardrails: fit-before-use, empty calibration rejected.
def test_guardrails():
    sc = SplitConformal()
    with pytest.raises(RuntimeError):
        sc.interval(1.0)
    with pytest.raises(ValueError):
        sc.fit(np.array([]))
