"""Phase 8 surrogate training: sweep.csv -> baseline vs GBM -> validation scatter.

The honest-AI ladder: trivial baseline (global mean) -> linear regression ->
HistGradientBoosting. Validation is GroupKFold over PARAMETER COMBOS (whole
combos held out, never just seeds) — tests interpolation across the parameter
space and prevents seed leakage. Reports MAE with bootstrap CIs; saves the
surrogate-vs-simulator scatter (the Phase-8 deliverable figure).

Targets: max_pinned_upstream (crowd-at-risk), exposure_area_sec_3, grid_t_4.0
(watch onset; runs that never cross are excluded from that target and the
exclusion count reported).

Usage: python scripts/phase8_train.py [--csv results/phase8/sweep.csv]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from sim.viz.style import save_figure, set_style

TARGETS = ["max_pinned_upstream", "exposure_area_sec_3", "grid_t_4.0"]
NO_BLOCK = 10_000.0


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature frame from sweep params (schedule one-hot, rest numeric)."""
    X = pd.get_dummies(df["schedule"], prefix="sched")
    X["W"] = df["W"]
    X["surge"] = (df["crowd"] == "surge").astype(int)
    X["has_block"] = (df["t_block"] < NO_BLOCK).astype(int)
    X["t_block"] = df["t_block"].where(df["t_block"] < NO_BLOCK, 150.0)
    return X


def cv_mae(model_fn, X, y, groups, n_boot: int = 500) -> tuple[float, float, float, np.ndarray]:
    """GroupKFold CV MAE with bootstrap CI; returns (mae, lo, hi, oof_pred)."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        m = model_fn()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    err = np.abs(oof - y.to_numpy())
    rng = np.random.Generator(np.random.PCG64(0))
    boots = [np.mean(rng.choice(err, len(err))) for _ in range(n_boot)]
    return float(err.mean()), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5)), oof


class MeanBaseline:
    """Trivial baseline: predict the training mean."""

    def fit(self, X, y):
        self._mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mu)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("results", "phase8", "sweep.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"{len(df)} runs loaded; configs: {df.config.value_counts().to_dict()}")
    df = df[df.config == "C2"].copy()  # C4 anchors are held out for calibration
    combo = (df.schedule + "|" + df.W.astype(str) + "|" + df.crowd
             + "|" + df.t_block.astype(str))
    X = featurize(df)

    set_style()
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 3.6))
    results = []
    for ax, target in zip(np.atleast_1d(axes), TARGETS):
        y = df[target]
        mask = y.notna()
        excl = int((~mask).sum())
        Xm, ym, gm = X[mask], y[mask], combo[mask]
        row = {"target": target, "n": len(ym), "excluded": excl}
        for name, fn in [
            ("mean", MeanBaseline),
            ("linear", LinearRegression),
            ("gbm", lambda: HistGradientBoostingRegressor(random_state=0)),
        ]:
            mae, lo, hi, oof = cv_mae(fn, Xm, ym, gm)
            row[f"{name}_mae"] = round(mae, 2)
            row[f"{name}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
            if name == "gbm":
                ax.scatter(ym, oof, s=12, alpha=0.6, color="#0072b2")
                lim = (0, max(float(ym.max()), float(np.nanmax(oof))) * 1.05)
                ax.plot(lim, lim, color="gray", lw=1, ls="--")
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.set_xlabel(f"simulator: {target}")
                ax.set_ylabel("surrogate (held-out)")
                ax.set_title(f"GBM MAE {mae:.2f} {row['gbm_ci']}", fontsize=9)
        results.append(row)
        print(row, flush=True)

    print(save_figure(fig, "fig_surrogate_scatter"))
    out = os.path.join("results", "phase8", "surrogate_eval.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(out)
    gbm_beats = all(r["gbm_mae"] < r["mean_mae"] for r in results)
    print(f"GBM beats trivial baseline on all targets: {gbm_beats}")


if __name__ == "__main__":
    main()
