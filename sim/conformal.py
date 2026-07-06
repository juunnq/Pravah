"""Split-conformal prediction intervals — distribution-free coverage bounds.

Wraps any point predictor with calibrated intervals: given absolute residuals
|y − ŷ| on a held-out calibration set, the (1−α) conformal quantile q makes
[ŷ − q, ŷ + q] cover the truth with probability ≥ 1−α on exchangeable data —
no distributional assumptions, no model retraining.

Why it matters here: the surrogate and forecaster stop giving point guesses
and start giving CALIBRATED bounds; the SafetyAgent can then act on the
worst-case-in-interval (principled risk aversion) instead of a best guess.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SplitConformal:
    """Split-conformal calibrator for symmetric absolute-residual intervals.

    Args:
        alpha: Miscoverage rate. alpha=0.1 -> 90% coverage target.
    """

    alpha: float = 0.1
    q: float = field(default=float("nan"), init=False)

    def fit(self, residuals: np.ndarray) -> "SplitConformal":
        """Calibrate on held-out absolute residuals |y - yhat|.

        Uses the finite-sample-valid quantile: the ceil((n+1)(1-alpha))-th
        order statistic of the residuals.
        """
        r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
        n = len(r)
        if n == 0:
            raise ValueError("need at least one calibration residual")
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        self.q = float(r[min(k, n) - 1])
        return self

    def interval(self, pred: float) -> tuple[float, float]:
        """Calibrated interval around a point prediction."""
        if np.isnan(self.q):
            raise RuntimeError("fit() before interval()")
        return (float(pred) - self.q, float(pred) + self.q)

    def upper(self, pred: float) -> float:
        """Worst-case-in-interval (the risk-averse decision input)."""
        return self.interval(pred)[1]

    @staticmethod
    def empirical_coverage(y: np.ndarray, pred: np.ndarray,
                           q: float) -> float:
        """Fraction of truths inside pred ± q (evaluation helper)."""
        y, pred = np.asarray(y, float), np.asarray(pred, float)
        return float(np.mean(np.abs(y - pred) <= q))
